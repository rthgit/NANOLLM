import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def audit_numeric_state(model):
    print("Post-Injection Parameter Audit:")
    bad_tensors = 0
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_tensors += 1
            print(f"  [AUDIT] Non-finite parameter: {name}")
    for name, buffer in model.named_buffers():
        if not torch.isfinite(buffer).all():
            bad_tensors += 1
            print(f"  [AUDIT] Non-finite buffer: {name}")
        if name.endswith(".scale"):
            print(
                f"  [SCALE] {name}: "
                f"min={buffer.min().item():.6g} "
                f"max={buffer.max().item():.6g} "
                f"mean={buffer.mean().item():.6g}"
            )
    if bad_tensors == 0:
        print("  [AUDIT] All parameters and buffers are finite.")


class RadialEncoding(nn.Module):
    def __init__(self, output_dim, vocab_size=32000, token_bits=8, normalize_features=False, bias=True):
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.token_bits = token_bits
        self.normalize_features = normalize_features
        self.phi = (1 + 5**0.5) / 2
        self.projection = nn.Linear(12, output_dim, bias=bias)

    def _radial_features(self, x):
        device = x.device
        x = x.to(torch.long)
        bits = torch.stack([(x >> i) & 1 for i in range(self.token_bits)], dim=-1).to(torch.float32)

        n = self.token_bits
        indices = torch.arange(n, device=device, dtype=torch.float32)
        magnitudes = self.phi ** indices
        angles = 2 * np.pi * indices / n
        real = torch.sum(bits * magnitudes * torch.cos(angles), dim=-1)
        imag = torch.sum(bits * magnitudes * torch.sin(angles), dim=-1)
        abs_val = torch.sqrt(real.square() + imag.square() + (1e-6 if self.normalize_features else 0.0))
        arg_val = torch.atan2(imag, real)

        features = torch.stack(
            [
                real,
                imag,
                abs_val,
                arg_val,
                real * 0.5,
                imag * 0.5,
                abs_val * 0.5,
                arg_val * 0.5,
                real * 0.2,
                imag * 0.2,
                abs_val * 0.2,
                arg_val * 0.2,
            ],
            dim=-1,
        )
        if self.normalize_features:
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)
        return features

    def forward(self, x):
        features = self._radial_features(x)
        weight = self.projection.weight.to(torch.float32)
        bias = self.projection.bias.to(torch.float32) if self.projection.bias is not None else None
        out = F.linear(features, weight, bias)
        out = torch.nan_to_num(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return out.to(self.projection.weight.dtype)


class NanoLinear(nn.Module):
    def __init__(self, name, packed_weight, scale, bits, original_shape, bias=None):
        super().__init__()
        self.name = name
        self.bits = bits
        self.original_shape = tuple(original_shape)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.bias = None

    def _get_discrete_weight(self):
        return unpack_discrete_tensor(self.packed_weight, self.bits, self.original_shape)

    def forward(self, x):
        target_dtype = x.dtype
        x_f32 = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
        x_f32 = x_f32.clamp(-65504.0, 65504.0)

        discrete_weight = self._get_discrete_weight().to(torch.float32)
        out = torch.matmul(x_f32, discrete_weight.t())
        out = out * self.scale
        if self.bias is not None:
            out = out + self.bias.to(torch.float32)

        out = torch.nan_to_num(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
        out = out.clamp(-65504.0, 65504.0)
        return out.to(target_dtype)


class MixedRowNanoLinear(nn.Module):
    def __init__(
        self,
        name,
        base_packed_weight,
        base_scale,
        base_bits,
        base_shape,
        base_row_indices,
        low_packed_weight,
        low_scales,
        low_bits,
        low_shape,
        low_packed_shape,
        low_group_size,
        low_row_indices,
        original_shape,
        residual_indices=None,
        residual_values=None,
        bias=None,
    ):
        super().__init__()
        self.name = name
        self.original_shape = tuple(original_shape)
        self.out_features = int(original_shape[0])
        self.base_bits = int(base_bits)
        self.low_bits = int(low_bits)
        self.base_shape = tuple(base_shape)
        self.low_shape = tuple(low_shape)
        self.low_packed_shape = tuple(low_packed_shape)
        self.low_group_size = int(low_group_size) if low_group_size else 0
        self.register_buffer("base_packed_weight", base_packed_weight)
        self.register_buffer("base_scale", torch.tensor(base_scale, dtype=torch.float32))
        self.register_buffer("low_packed_weight", low_packed_weight)
        self.register_buffer("low_scales", low_scales.to(torch.float32))
        self.register_buffer("low_row_indices", low_row_indices.to(torch.long))
        if base_row_indices is None:
            all_rows = torch.arange(self.out_features, dtype=torch.long)
            mask = torch.ones(self.out_features, dtype=torch.bool)
            mask[self.low_row_indices] = False
            base_row_indices = all_rows[mask]
        self.register_buffer("base_row_indices", base_row_indices.to(torch.long))
        if residual_indices is not None and residual_values is not None:
            self.register_buffer("residual_indices", residual_indices.to(torch.long))
            self.register_buffer("residual_values", residual_values.to(torch.float32))
        else:
            self.residual_indices = None
            self.residual_values = None
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.bias = None

    def _compute_rows(self, flat_x, packed_weight, bits, shape, scales, packed_shape=None, group_size=0):
        if shape[0] == 0:
            return None
        work_shape = packed_shape or shape
        discrete = unpack_discrete_tensor(packed_weight, bits, work_shape).to(torch.float32)
        if work_shape[1] != shape[1]:
            discrete = discrete[:, : shape[1]]
        if torch.is_tensor(scales) and scales.ndim == 2 and group_size > 0:
            n_rows = discrete.shape[0]
            padded_cols = ((shape[1] + group_size - 1) // group_size) * group_size
            if padded_cols != shape[1]:
                pad = torch.zeros((n_rows, padded_cols - shape[1]), dtype=discrete.dtype, device=discrete.device)
                discrete = torch.cat([discrete, pad], dim=1)
            grouped = discrete.reshape(n_rows, padded_cols // group_size, group_size)
            grouped = grouped * scales.to(torch.float32).unsqueeze(2)
            weight = grouped.reshape(n_rows, padded_cols)[:, : shape[1]]
            out = torch.matmul(flat_x, weight.t())
        elif torch.is_tensor(scales) and scales.ndim == 1:
            out = torch.matmul(flat_x, discrete.t())
            out = out * scales.to(torch.float32).unsqueeze(0)
        else:
            out = torch.matmul(flat_x, discrete.t())
            out = out * float(scales)
        return out

    def forward(self, x):
        target_dtype = x.dtype
        x_f32 = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
        x_f32 = x_f32.clamp(-65504.0, 65504.0)
        flat_x = x_f32.reshape(-1, x_f32.shape[-1])
        out = torch.zeros((flat_x.shape[0], self.out_features), dtype=torch.float32, device=flat_x.device)

        if self.base_shape[0] > 0:
            base_out = self._compute_rows(
                flat_x,
                self.base_packed_weight,
                self.base_bits,
                self.base_shape,
                self.base_scale,
                self.base_shape,
                0,
            )
            out.index_copy_(1, self.base_row_indices, base_out)

        if self.low_shape[0] > 0:
            low_out = self._compute_rows(
                flat_x,
                self.low_packed_weight,
                self.low_bits,
                self.low_shape,
                self.low_scales,
                self.low_packed_shape,
                self.low_group_size,
            )
            if self.residual_indices is not None and self.residual_values is not None:
                corr = (
                    flat_x[:, self.residual_indices] * self.residual_values.unsqueeze(0)
                ).sum(dim=-1)
                low_out = low_out + corr
            out.index_copy_(1, self.low_row_indices, low_out)

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)

        out = torch.nan_to_num(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
        out = out.clamp(-65504.0, 65504.0)
        out = out.reshape(*x.shape[:-1], self.out_features)
        return out.to(target_dtype)


def unpack_discrete_tensor(packed_weight, bits, original_shape):
    num_elements = int(np.prod(original_shape))
    if bits == 8:
        return packed_weight.to(torch.int8).reshape(original_shape)
    if bits == 6:
        flat = packed_weight.flatten().to(torch.uint8)
        if len(flat) % 3 != 0:
            padding = 3 - (len(flat) % 3)
            flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)])
        groups = flat.reshape(-1, 3)
        b0 = groups[:, 0]
        b1 = groups[:, 1]
        b2 = groups[:, 2]
        v0 = (b0 >> 2) & 0x3F
        v1 = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)
        v2 = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)
        v3 = b2 & 0x3F
        unpacked = torch.stack([v0, v1, v2, v3], dim=1).flatten()
        return (unpacked[:num_elements].reshape(original_shape).to(torch.int16) - 32).to(torch.int8)
    if bits == 4:
        high = (packed_weight >> 4).to(torch.int8)
        low = (packed_weight & 0x0F).to(torch.int8)
        unpacked = torch.stack([high, low], dim=1).flatten()
        return (unpacked[:num_elements].reshape(original_shape) - 8).to(torch.int8)
    if bits == 2:
        b1 = (packed_weight >> 6) & 0x03
        b2 = (packed_weight >> 4) & 0x03
        b3 = (packed_weight >> 2) & 0x03
        b4 = packed_weight & 0x03
        unpacked = torch.stack([b1, b2, b3, b4], dim=1).flatten()
        return (unpacked[:num_elements].reshape(original_shape) - 2).to(torch.int8)
    raise ValueError(f"Unsupported bit-width: {bits}")


def dequantize_tensor(packed_weight, scale, bits, original_shape, target_dtype):
    discrete = unpack_discrete_tensor(packed_weight, bits, original_shape)
    return (discrete.to(torch.float32) * scale).to(target_dtype)


def patch_norms_for_fp32(model):
    print("Patching norm layers for FP32 execution...")
    for module in model.modules():
        cls_name = module.__class__.__name__.lower()
        if "norm" not in cls_name:
            continue
        original_forward = module.forward

        def safe_forward(hidden_states, *args, _forward=original_forward, **kwargs):
            target_dtype = hidden_states.dtype
            hidden_states = torch.nan_to_num(
                hidden_states.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0
            )
            out = _forward(hidden_states, *args, **kwargs)
            if torch.is_tensor(out):
                out = torch.nan_to_num(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
                return out.to(target_dtype)
            return out

        module.forward = safe_forward


def build_radial_layer(config, st_dict, token_bits):
    has_bias = "model.embed_tokens.projection.bias" in st_dict
    return RadialEncoding(
        config.hidden_size,
        config.vocab_size,
        token_bits=token_bits,
        normalize_features=False,
        bias=has_bias,
    )


def apply_radial_override(model, radial_projection_path):
    if not radial_projection_path:
        return
    if not os.path.exists(radial_projection_path):
        print(f"  [WARN] Radial projection override not found: {radial_projection_path}")
        return

    print(f"  Applying radial projection override: {radial_projection_path}")
    override_state = torch.load(radial_projection_path, map_location="cpu")
    projection = model.get_input_embeddings().projection
    projection.load_state_dict(override_state, strict=False)


def maybe_restore_tied_lm_head(model_state, st_dict, config, loaded_names):
    if not getattr(config, "tie_word_embeddings", False):
        return
    if "lm_head.weight" in st_dict:
        return
    if "model.embed_tokens.weight" not in model_state or "lm_head.weight" not in model_state:
        return
    model_state["lm_head.weight"].copy_(model_state["model.embed_tokens.weight"])
    loaded_names.add("lm_head.weight")


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


def initialize_missing_state(model_state, loaded_names, expected_zero_names=None):
    expected_zero_names = expected_zero_names or set()
    for name, tensor in model_state.items():
        if name in loaded_names:
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            if name in expected_zero_names:
                print(f"  [INFO] Zero-initializing pruned tensor: {name}")
            else:
                print(f"  [WARN] Initializing missing tensor to zero: {name}")
            tensor.zero_()


def sanitize_live_tensors(model):
    for name, param in model.named_parameters():
        if param.is_floating_point() and not torch.isfinite(param).all():
            print(f"  [WARN] Sanitizing non-finite live parameter: {name}")
            param.data.copy_(
                torch.nan_to_num(param.data.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0).to(param.dtype)
            )
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            print(f"  [WARN] Sanitizing non-finite live buffer: {name}")
            buffer.data.copy_(
                torch.nan_to_num(buffer.data.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0).to(buffer.dtype)
            )


def load_nano_direct(model_path, radial_projection_path=None, token_bits=8, fp32_rmsnorm=False):
    print(f"Loading NANO Native-Bit Shell: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    topo_path = os.path.join(model_path, "nano_topology.json")
    with open(topo_path, "r", encoding="utf-8") as handle:
        quant_info = json.load(handle)
    pruned_path = os.path.join(model_path, "nano_pruned.json")
    if os.path.exists(pruned_path):
        with open(pruned_path, "r", encoding="utf-8") as handle:
            pruned_info = json.load(handle)
    else:
        pruned_info = {}
    st_dict = load_file(os.path.join(model_path, "model.safetensors"))
    has_saved_radial = any(key.startswith("model.embed_tokens.projection.") for key in st_dict.keys())

    if has_saved_radial and getattr(config, "tie_word_embeddings", False):
        print("  Untying LM head for radial runtime compatibility.")
        config.tie_word_embeddings = False

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model = model.to_empty(device="cpu")

    if has_saved_radial:
        model.set_input_embeddings(build_radial_layer(config, st_dict, token_bits))

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        weight_name = f"{name}.weight"
        if weight_name not in quant_info:
            continue

        info = quant_info[weight_name]

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        bias = st_dict.get(f"{name}.bias")
        if info.get("storage") == "row_mixed":
            replacement = MixedRowNanoLinear(
                name,
                st_dict[info["base_key"]],
                info["base_scale"],
                info["base_bits"],
                info["base_shape"],
                st_dict.get(info["base_index_key"]) if info.get("base_index_key") else None,
                st_dict[info["low_key"]],
                st_dict[info["low_scale_key"]],
                info["low_bits"],
                info["low_shape"],
                info.get("low_packed_shape", info["low_shape"]),
                info.get("low_group_size", 0),
                st_dict[info["low_index_key"]],
                info["shape"],
                st_dict.get(info["residual_index_key"]) if info.get("residual_index_key") else None,
                st_dict.get(info["residual_value_key"]) if info.get("residual_value_key") else None,
                bias=bias,
            )
        else:
            replacement = NanoLinear(
                name,
                st_dict[weight_name],
                info["scale"],
                info["bits"],
                info["shape"],
                bias=bias,
            )
        setattr(parent, attr_name, replacement)

    model = model.to(torch.float16)
    reinitialize_rotary_buffers(model)
    model_state = model.state_dict()
    loaded_names = {
        name
        for name in model_state.keys()
        if name.endswith("scale") or name.endswith("scales")
    }

    for name, value in st_dict.items():
        if name in quant_info:
            info = quant_info[name]
            if info.get("storage") == "row_mixed":
                continue
            packed_key = f"{name[:-7]}.packed_weight"
            if packed_key in model_state:
                model_state[packed_key].copy_(value)
                loaded_names.add(packed_key)
            elif name in model_state:
                model_state[name].copy_(
                    dequantize_tensor(
                        value,
                        info["scale"],
                        info["bits"],
                        info["shape"],
                        model_state[name].dtype,
                    )
                )
                loaded_names.add(name)
            continue

        if name in model_state:
            if "inv_freq" in name and not torch.isfinite(value).all():
                print(f"  [WARN] Skipping non-finite rotary buffer: {name}")
                continue
            if value.is_floating_point() and not torch.isfinite(value).all():
                print(f"  [WARN] Sanitizing non-finite tensor from checkpoint: {name}")
                value = torch.nan_to_num(value.to(torch.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
            model_state[name].copy_(value.to(model_state[name].dtype))
            loaded_names.add(name)

    maybe_restore_tied_lm_head(model_state, st_dict, config, loaded_names)
    if has_saved_radial:
        apply_radial_override(model, radial_projection_path)
    if has_saved_radial and radial_projection_path:
        loaded_names.add("model.embed_tokens.projection.weight")
        if "model.embed_tokens.projection.bias" in model_state:
            loaded_names.add("model.embed_tokens.projection.bias")

    initialize_missing_state(model_state, loaded_names, expected_zero_names=set(pruned_info))
    sanitize_live_tensors(model)

    if fp32_rmsnorm:
        patch_norms_for_fp32(model)

    audit_numeric_state(model)
    model.eval()
    return model, tokenizer


def build_generate_kwargs(args, tokenizer):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        kwargs["do_sample"] = False
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--radial-projection", help="Optional override for embed_tokens.projection state_dict")
    parser.add_argument("--prompt", default="The future of AI")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--token-bits", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--logit-clamp", type=float, default=20.0)
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_nano_direct(
        args.model_path,
        radial_projection_path=args.radial_projection,
        token_bits=args.token_bits,
        fp32_rmsnorm=args.fp32_rmsnorm,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")

    print("\nStarting Native Bit-Logic Execution...")
    original_forward = model.forward

    def guarded_forward(*forward_args, **forward_kwargs):
        outputs = original_forward(*forward_args, **forward_kwargs)
        if hasattr(outputs, "logits"):
            logits = torch.nan_to_num(outputs.logits, nan=-args.logit_clamp, posinf=args.logit_clamp, neginf=-args.logit_clamp)
            outputs.logits = logits.clamp(-args.logit_clamp, args.logit_clamp)
        return outputs

    model.forward = guarded_forward

    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **build_generate_kwargs(args, tokenizer))
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nResponse: {response}")
    except Exception as exc:
        print(f"\nCRITICAL ERROR during generation: {exc}")


if __name__ == "__main__":
    main()
