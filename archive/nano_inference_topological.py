import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

class RadialEncoding(nn.Module):
    def __init__(self, output_dim, vocab_size=32000):
        super().__init__()
        self.output_dim = output_dim
        self.phi = (1 + 5**0.5) / 2
        self.projection = nn.Linear(12, output_dim)
        
    def forward(self, x):
        device = x.device
        bits = []
        for i in range(8):
            bits.append((x >> i) & 1)
        bits = torch.stack(bits, dim=-1).float()
        n = 8
        indices = torch.arange(n, device=device).float()
        magnitudes = self.phi ** indices
        angles = 2 * np.pi * indices / n
        re = torch.sum(bits * magnitudes * torch.cos(angles), dim=-1)
        im = torch.sum(bits * magnitudes * torch.sin(angles), dim=-1)
        abs_v = torch.sqrt(re**2 + im**2)
        arg_v = torch.atan2(im, re)
        feat = torch.stack([re, im, abs_v, arg_v, re*0.5, im*0.5, abs_v*0.5, arg_v*0.5, re*0.2, im*0.2, abs_v*0.2, arg_v*0.2], dim=-1)
        weight = self.projection.weight.to(torch.float32)
        bias = self.projection.bias.to(torch.float32) if self.projection.bias is not None else None
        out = torch.nn.functional.linear(feat.to(torch.float32), weight, bias)
        out = torch.nan_to_num(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return out.to(self.projection.weight.dtype)

def unpack_weights(packed, bits, original_shape):
    """
    Physically unpacks NANO weights from uint8 containers.
    """
    device = packed.device
    num_el = np.prod(original_shape)
    
    if bits == 8:
        # Just signed int8
        return packed.to(torch.int8).to(torch.float32)
    
    if bits == 4:
        # Unpack 2 x 4-bit
        high = (packed >> 4).to(torch.int8)
        low = (packed & 0x0F).to(torch.int8)
        unpacked = torch.stack([high, low], dim=1).flatten()
        return (unpacked[:num_el].reshape(original_shape).to(torch.float32) - 8)

    if bits == 2:
        # Unpack 4 x 2-bit
        b1 = (packed >> 6) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 2) & 0x03
        b4 = (packed) & 0x03
        unpacked = torch.stack([b1, b2, b3, b4], dim=1).flatten()
        return (unpacked[:num_el].reshape(original_shape).to(torch.float32) - 2)
    
    return packed


def sanitize_live_tensors(model):
    for _, param in model.named_parameters():
        if param.is_floating_point() and not torch.isfinite(param).all():
            param.data.copy_(
                torch.nan_to_num(param.data.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0).to(param.dtype)
            )
    for _, buffer in model.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            buffer.data.copy_(
                torch.nan_to_num(buffer.data.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0).to(buffer.dtype)
            )


def reinitialize_rotary_buffers(model):
    for name, module in model.named_modules():
        if not hasattr(module, "inv_freq") or not hasattr(module, "original_inv_freq"):
            continue
        config = getattr(module, "config", None)
        if config is None:
            continue
        try:
            fresh_module = module.__class__(config, device=torch.device("cpu"))
        except Exception as exc:
            print(f"  [WARN] Failed to refresh rotary buffers for {name}: {exc}")
            continue
        module.inv_freq = fresh_module.inv_freq.to(torch.float32)
        module.original_inv_freq = fresh_module.original_inv_freq.to(torch.float32)
        for attr in ("attention_scaling", "max_seq_len_cached", "original_max_seq_len", "rope_type"):
            if hasattr(fresh_module, attr):
                setattr(module, attr, getattr(fresh_module, attr))


def maybe_restore_tied_lm_head(new_sd, config):
    if not getattr(config, "tie_word_embeddings", False):
        return
    if "lm_head.weight" not in new_sd or "model.embed_tokens.weight" not in new_sd:
        return
    if torch.count_nonzero(new_sd["lm_head.weight"]).item() != 0:
        return
    new_sd["lm_head.weight"] = new_sd["model.embed_tokens.weight"].clone()


def load_nano_model(model_path):
    print(f"Loading NANO Physical Bit-Packed Model: {model_path}")
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    topo_path = os.path.join(model_path, "nano_topology.json")
    with open(topo_path, "r", encoding="utf-8") as f:
        quant_info = json.load(f)
    st_dict = load_file(os.path.join(model_path, "model.safetensors"))

    has_saved_radial = any(key.startswith("model.embed_tokens.projection.") for key in st_dict.keys())
    if has_saved_radial and getattr(config, "tie_word_embeddings", False):
        print("Untying LM head for radial runtime compatibility...")
        config.tie_word_embeddings = False
    
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    
    model = model.to_empty(device="cpu")
    
    if has_saved_radial or hasattr(config, "nano_radial") or config.architectures[0] == "RadialFormer":
        print("Patching Radial Encoding...")
        model.set_input_embeddings(RadialEncoding(config.hidden_size, config.vocab_size))
    model_sd = model.state_dict()
    new_sd = {}
    
    print("Inference: Unpacking and dequantizing weights...")
    for name in model_sd.keys():
        if name in st_dict:
            packed = st_dict[name]
            if name in quant_info:
                info = quant_info[name]
                scale = info["scale"]
                bits = info["bits"]
                orig_shape = info["shape"]
                
                # Unpack -> Dequantize
                unpacked_f32 = unpack_weights(packed, bits, orig_shape)
                new_sd[name] = (unpacked_f32 * scale).to(torch.float16)
            else:
                new_sd[name] = packed.to(torch.float16)
        else:
            new_sd[name] = torch.zeros_like(model_sd[name])

    maybe_restore_tied_lm_head(new_sd, config)
    model.load_state_dict(new_sd, strict=False)
    model = model.to(torch.float16)
    reinitialize_rotary_buffers(model)
    sanitize_live_tensors(model)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="NanoLLM is a topological")
    args = parser.parse_args()

    model, tokenizer = load_nano_model(args.model_path)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    print("\nStarting generation...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    
    print(f"\nResponse: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
