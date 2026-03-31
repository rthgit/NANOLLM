import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TARGET_SUFFIXES = (
    "model.embed_tokens.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def pack_weights(tensor, bits):
    tensor_f32 = tensor.to(torch.float32)
    max_val = tensor_f32.abs().max()

    if bits == 8:
        if max_val == 0:
            return torch.zeros(tensor.shape, dtype=torch.int8), 0.0
        scale = max_val / 127
        packed = torch.clamp(torch.round(tensor_f32 / scale), -128, 127).to(torch.int8)
        return packed, float(scale)

    if bits == 4:
        if max_val == 0:
            return torch.zeros((tensor.numel() + 1) // 2, dtype=torch.uint8), 0.0
        scale = max_val / 7
        u_vals = (torch.clamp(torch.round(tensor_f32 / scale), -8, 7) + 8).to(torch.uint8).flatten()
        if len(u_vals) % 2 != 0:
            u_vals = torch.cat([u_vals, torch.zeros(1, dtype=torch.uint8)])
        packed = (u_vals[::2] << 4) | (u_vals[1::2] & 0x0F)
        return packed, float(scale)

    raise ValueError(f"Unsupported bit-width: {bits}")


def choose_bits(name, bit_rules):
    for suffix, bits in bit_rules:
        if name.endswith(suffix):
            return bits
    return None


def load_index(source_dir):
    index_path = source_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return index["weight_map"]


def copy_metadata(source_dir, out_dir):
    for filename in COPY_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)


def export_safe_bitpacked(source_dir, out_dir, bit_rules):
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_index(source_dir)
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    by_shard = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    new_state = {}
    quant_info = {}
    counts = defaultdict(int)

    for shard_name in sorted(by_shard):
        shard_path = source_dir / shard_name
        print(f"Reading {shard_path} ...", flush=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in sorted(by_shard[shard_name]):
                if tie_word_embeddings and name == "lm_head.weight":
                    continue

                tensor = handle.get_tensor(name)
                bits = choose_bits(name, bit_rules)
                if bits is None:
                    new_state[name] = tensor
                    continue

                packed, scale = pack_weights(tensor, bits)
                new_state[name] = packed
                quant_info[name] = {
                    "scale": scale,
                    "bits": bits,
                    "shape": list(tensor.shape),
                }
                counts[bits] += 1

    copy_metadata(source_dir, out_dir)
    save_file(new_state, str(out_dir / "model.safetensors"))
    (out_dir / "nano_topology.json").write_text(json.dumps(quant_info, indent=2), encoding="utf-8")
    (out_dir / "nano_export_policy.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "bit_rules": {suffix: bits for suffix, bits in bit_rules},
                "tie_word_embeddings": tie_word_embeddings,
                "quantized_tensors": len(quant_info),
                "bit_counts": {str(k): v for k, v in sorted(counts.items())},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved repaired bit-packed artifact to {out_dir}", flush=True)
    print(f"Quantized tensors: {len(quant_info)}", flush=True)
    print(f"Bit counts: {dict(sorted(counts.items()))}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a safer bit-packed artifact from a local HF checkpoint.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attention-bits", type=int, choices=(8, 4), default=8)
    parser.add_argument("--mlp-bits", type=int, choices=(8, 4), default=4)
    parser.add_argument("--embed-bits", type=int, choices=(8, 4))
    parser.add_argument("--q-bits", type=int, choices=(8, 4))
    parser.add_argument("--k-bits", type=int, choices=(8, 4))
    parser.add_argument("--v-bits", type=int, choices=(8, 4))
    parser.add_argument("--o-bits", type=int, choices=(8, 4))
    parser.add_argument("--gate-bits", type=int, choices=(8, 4))
    parser.add_argument("--up-bits", type=int, choices=(8, 4))
    parser.add_argument("--down-bits", type=int, choices=(8, 4))
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    bit_rules = [
        ("model.embed_tokens.weight", args.embed_bits),
        ("self_attn.q_proj.weight", args.q_bits or args.attention_bits),
        ("self_attn.k_proj.weight", args.k_bits or args.attention_bits),
        ("self_attn.v_proj.weight", args.v_bits or args.attention_bits),
        ("self_attn.o_proj.weight", args.o_bits or args.attention_bits),
        ("mlp.gate_proj.weight", args.gate_bits or args.mlp_bits),
        ("mlp.up_proj.weight", args.up_bits or args.mlp_bits),
        ("mlp.down_proj.weight", args.down_bits or args.mlp_bits),
    ]
    export_safe_bitpacked(source_dir, out_dir, bit_rules)


if __name__ == "__main__":
    main()
