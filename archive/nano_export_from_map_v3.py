import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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

    if bits == 2:
        if max_val == 0:
            return torch.zeros((tensor.numel() + 3) // 4, dtype=torch.uint8), 0.0
        scale = max_val / 1
        u_vals = (torch.clamp(torch.round(tensor_f32 / scale), -2, 1) + 2).to(torch.uint8).flatten()
        padding = (4 - (len(u_vals) % 4)) % 4
        if padding > 0:
            u_vals = torch.cat([u_vals, torch.zeros(padding, dtype=torch.uint8)])
        packed = (u_vals[0::4] << 6) | (u_vals[1::4] << 4) | (u_vals[2::4] << 2) | u_vals[3::4]
        return packed, float(scale)

    raise ValueError(f"Unsupported bit-width: {bits}")


def copy_metadata(source_dir, out_dir):
    for filename in COPY_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)


def load_weight_map(source_dir: Path):
    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return index["weight_map"]


def load_precision_map(precision_map_path: Path):
    data = json.loads(precision_map_path.read_text(encoding="utf-8"))
    if "tier_map" not in data:
        raise ValueError("precision-map must contain a top-level tier_map")
    return data


def choose_bits(tensor_name: str, tier_map: dict[str, int], embed_bits: int | None):
    if tensor_name == "model.embed_tokens.weight" and embed_bits is not None:
        return embed_bits
    if tensor_name.endswith(".weight"):
        module_name = tensor_name[:-7]
        if module_name in tier_map:
            return int(tier_map[module_name])
    return None


def export_from_map(source_dir: Path, precision_map_path: Path, out_dir: Path, embed_bits: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    precision_map = load_precision_map(precision_map_path)
    tier_map = precision_map["tier_map"]
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    weight_map = load_weight_map(source_dir)

    by_shard = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    new_state = {}
    quant_info = {}
    pruned_info = {}
    counts = defaultdict(int)

    for shard_name in sorted(by_shard):
        shard_path = source_dir / shard_name
        print(f"Reading {shard_path} ...", flush=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in sorted(by_shard[shard_name]):
                if tie_word_embeddings and name == "lm_head.weight":
                    continue

                bits = choose_bits(name, tier_map, embed_bits)
                if bits is None:
                    module_prefix = name[:-5] if name.endswith(".bias") else None
                    if module_prefix is not None and module_prefix in tier_map and int(tier_map[module_prefix]) == 0:
                        tensor = handle.get_tensor(name)
                        pruned_info[name] = {"shape": list(tensor.shape), "bits": 0}
                        counts[0] += 1
                        continue
                    new_state[name] = handle.get_tensor(name)
                    continue

                tensor = handle.get_tensor(name)
                if bits == 0:
                    pruned_info[name] = {"shape": list(tensor.shape), "bits": 0}
                    counts[0] += 1
                    continue

                packed, scale = pack_weights(tensor, bits)
                new_state[name] = packed
                quant_info[name] = {"scale": scale, "bits": bits, "shape": list(tensor.shape)}
                counts[bits] += 1

    copy_metadata(source_dir, out_dir)
    save_file(new_state, str(out_dir / "model.safetensors"))
    (out_dir / "nano_topology.json").write_text(json.dumps(quant_info, indent=2), encoding="utf-8")
    (out_dir / "nano_pruned.json").write_text(json.dumps(pruned_info, indent=2), encoding="utf-8")
    (out_dir / "nano_export_manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "precision_map": str(precision_map_path),
                "embed_bits": embed_bits,
                "tie_word_embeddings": tie_word_embeddings,
                "quantized_tensors": len(quant_info),
                "pruned_tensors": len(pruned_info),
                "bit_counts": {str(k): v for k, v in sorted(counts.items())},
                "map_summary": precision_map.get("summary", {}),
                "map_settings": precision_map.get("settings", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved canonical NANO artifact to {out_dir}", flush=True)
    print(f"Quantized tensors: {len(quant_info)}", flush=True)
    print(f"Pruned tensors: {len(pruned_info)}", flush=True)
    print(f"Bit counts: {dict(sorted(counts.items()))}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a canonical NANO artifact from a v3 precision map.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--precision-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embed-bits", type=int, choices=(8, 4, 2, 0))
    return parser.parse_args()


def main():
    args = parse_args()
    export_from_map(
        Path(args.source_dir),
        Path(args.precision_map),
        Path(args.out_dir),
        args.embed_bits,
    )


if __name__ == "__main__":
    main()
