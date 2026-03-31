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


def pack_u6(u_vals: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 6-bit values (0..63) into bytes (4 values -> 3 bytes)."""
    flat = u_vals.flatten().to(torch.uint8)
    padding = (4 - (len(flat) % 4)) % 4
    if padding > 0:
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8)])
    groups = flat.reshape(-1, 4)
    a = groups[:, 0]
    b = groups[:, 1]
    c = groups[:, 2]
    d = groups[:, 3]
    byte0 = (a << 2) | (b >> 4)
    byte1 = ((b & 0x0F) << 4) | (c >> 2)
    byte2 = ((c & 0x03) << 6) | d
    return torch.stack([byte0, byte1, byte2], dim=1).flatten()


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

    if bits == 6:
        if max_val == 0:
            return torch.zeros(((tensor.numel() + 3) // 4) * 3, dtype=torch.uint8), 0.0
        scale = max_val / 31
        u_vals = (torch.clamp(torch.round(tensor_f32 / scale), -32, 31) + 32).to(torch.uint8)
        packed = pack_u6(u_vals)
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


def quantize_int8_with_scale(tensor, scale):
    if scale == 0:
        return torch.zeros(tensor.shape, dtype=torch.int8)
    tensor_f32 = tensor.to(torch.float32)
    return torch.clamp(torch.round(tensor_f32 / scale), -128, 127).to(torch.int8)


def pack_rows_with_row_scales(tensor, bits, group_size=None):
    tensor_f32 = tensor.to(torch.float32)
    if tensor_f32.ndim != 2:
        raise ValueError("Row-wise packing expects a 2D tensor.")
    if bits == 8:
        raise ValueError("Row-wise packing is only used for low-bit rows.")

    if group_size is not None and group_size > 0:
        padded_cols = ((tensor_f32.shape[1] + group_size - 1) // group_size) * group_size
        if padded_cols != tensor_f32.shape[1]:
            pad = torch.zeros((tensor_f32.shape[0], padded_cols - tensor_f32.shape[1]), dtype=tensor_f32.dtype)
            work = torch.cat([tensor_f32, pad], dim=1)
        else:
            work = tensor_f32
        work = work.reshape(tensor_f32.shape[0], padded_cols // group_size, group_size)
        row_max = work.abs().amax(dim=2)
    else:
        padded_cols = tensor_f32.shape[1]
        work = tensor_f32
        row_max = tensor_f32.abs().amax(dim=1)

    if bits == 6:
        denom = 31.0
    elif bits == 4:
        denom = 7.0
    else:
        denom = 1.0
    scales = torch.where(row_max > 0, row_max / denom, torch.zeros_like(row_max))
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    if group_size is not None and group_size > 0:
        q = torch.round(work / safe_scales.unsqueeze(2)).reshape(tensor_f32.shape[0], padded_cols)
    else:
        q = torch.round(tensor_f32 / safe_scales.unsqueeze(1))

    if bits == 6:
        u_vals = (torch.clamp(q, -32, 31).to(torch.int16) + 32).to(torch.uint8)
        packed = pack_u6(u_vals)
        if group_size is not None and group_size > 0:
            dequant = (q.reshape(tensor_f32.shape[0], padded_cols // group_size, group_size) * safe_scales.unsqueeze(2)).reshape(
                tensor_f32.shape[0], padded_cols
            )[:, : tensor_f32.shape[1]]
        else:
            dequant = q[:, : tensor_f32.shape[1]] * safe_scales.unsqueeze(1)
        return packed, scales.to(torch.float32), [tensor_f32.shape[0], padded_cols], dequant.to(torch.float32)

    if bits == 4:
        u_vals = (torch.clamp(q, -8, 7).to(torch.int16) + 8).to(torch.uint8).flatten()
        if len(u_vals) % 2 != 0:
            u_vals = torch.cat([u_vals, torch.zeros(1, dtype=torch.uint8)])
        packed = (u_vals[::2] << 4) | (u_vals[1::2] & 0x0F)
        if group_size is not None and group_size > 0:
            dequant = (q.reshape(tensor_f32.shape[0], padded_cols // group_size, group_size) * safe_scales.unsqueeze(2)).reshape(
                tensor_f32.shape[0], padded_cols
            )[:, : tensor_f32.shape[1]]
        else:
            dequant = q[:, : tensor_f32.shape[1]] * safe_scales.unsqueeze(1)
        return packed, scales.to(torch.float32), [tensor_f32.shape[0], padded_cols], dequant.to(torch.float32)

    u_vals = (torch.clamp(q, -2, 1).to(torch.int16) + 2).to(torch.uint8).flatten()
    padding = (4 - (len(u_vals) % 4)) % 4
    if padding > 0:
        u_vals = torch.cat([u_vals, torch.zeros(padding, dtype=torch.uint8)])
    packed = (u_vals[0::4] << 6) | (u_vals[1::4] << 4) | (u_vals[2::4] << 2) | u_vals[3::4]
    if group_size is not None and group_size > 0:
        dequant = (q.reshape(tensor_f32.shape[0], padded_cols // group_size, group_size) * safe_scales.unsqueeze(2)).reshape(
            tensor_f32.shape[0], padded_cols
        )[:, : tensor_f32.shape[1]]
    else:
        dequant = q[:, : tensor_f32.shape[1]] * safe_scales.unsqueeze(1)
    return packed, scales.to(torch.float32), [tensor_f32.shape[0], padded_cols], dequant.to(torch.float32)


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


def export_row_mixed_tensor(weight_name: str, tensor, spec: dict):
    if tensor.ndim != 2:
        raise ValueError(f"Row-mixed export only supports 2D tensors, got {tensor.shape} for {weight_name}")

    base_bits = int(spec.get("base_bits", 8))
    low_bits = int(spec["low_bits"])
    group_size = spec.get("group_size")
    residual_topk = int(spec.get("residual_topk", 0))
    if base_bits != 8:
        raise ValueError("Row-mixed export currently requires base_bits=8.")
    if low_bits not in (6, 4, 2):
        raise ValueError("Row-mixed export currently supports low_bits in {6,4,2}.")

    out_features = tensor.shape[0]
    raw_indices = sorted({int(idx) for idx in spec.get("row_indices", [])})
    if not raw_indices:
        raise ValueError(f"Row-mixed spec for {weight_name} must contain at least one row index.")
    if raw_indices[0] < 0 or raw_indices[-1] >= out_features:
        raise ValueError(f"Row indices out of range for {weight_name}: {raw_indices[:3]} ...")

    low_row_indices = torch.tensor(raw_indices, dtype=torch.int32)
    base_row_indices = torch.tensor(
        [idx for idx in range(out_features) if idx not in set(raw_indices)],
        dtype=torch.int32,
    )

    tensor_f32 = tensor.to(torch.float32)
    base_scale = float((tensor_f32.abs().max() / 127).item()) if tensor_f32.numel() else 0.0
    baseline_packed = quantize_int8_with_scale(tensor_f32, base_scale)
    baseline_tensor = baseline_packed.to(torch.float32) * base_scale
    base_rows = (
        baseline_tensor[base_row_indices.to(torch.long)]
        if len(base_row_indices)
        else baseline_tensor.new_zeros((0, tensor.shape[1]))
    )
    low_rows = baseline_tensor[low_row_indices.to(torch.long)]

    base_packed = quantize_int8_with_scale(base_rows, base_scale)
    low_packed, low_scales, low_packed_shape, low_dequant = pack_rows_with_row_scales(
        low_rows,
        low_bits,
        group_size=group_size,
    )

    prefix = weight_name.replace(".", "__")
    base_key = f"{prefix}__base_packed_weight"
    low_key = f"{prefix}__low_packed_weight"
    low_scale_key = f"{prefix}__low_scales"
    low_index_key = f"{prefix}__low_row_indices"
    residual_index_key = f"{prefix}__residual_indices"
    residual_value_key = f"{prefix}__residual_values"

    state_entries = {
        base_key: base_packed,
        low_key: low_packed,
        low_scale_key: low_scales,
        low_index_key: low_row_indices,
    }
    if residual_topk > 0:
        residual = low_rows - low_dequant
        topk = min(residual_topk, residual.shape[1])
        top_vals, top_idx = torch.topk(residual.abs(), k=topk, dim=1)
        signed_vals = residual.gather(1, top_idx)
        state_entries[residual_index_key] = top_idx.to(torch.int16)
        state_entries[residual_value_key] = signed_vals.to(torch.float16)

    info = {
        "storage": "row_mixed",
        "shape": list(tensor.shape),
        "base_bits": base_bits,
        "base_scale": base_scale,
        "base_shape": list(base_rows.shape),
        "base_rows": int(base_rows.shape[0]),
        "base_key": base_key,
        "low_bits": low_bits,
        "low_shape": list(low_rows.shape),
        "low_packed_shape": low_packed_shape,
        "low_rows": int(low_rows.shape[0]),
        "low_key": low_key,
        "low_scale_key": low_scale_key,
        "low_index_key": low_index_key,
        "low_group_size": group_size,
        "residual_topk": residual_topk,
    }
    if residual_topk > 0:
        info["residual_index_key"] = residual_index_key
        info["residual_value_key"] = residual_value_key
    param_counts = {
        base_bits: int(base_rows.numel()),
        low_bits: int(low_rows.numel()),
    }
    return state_entries, info, param_counts


def export_from_map(source_dir: Path, precision_map_path: Path, out_dir: Path, embed_bits: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    precision_map = load_precision_map(precision_map_path)
    tier_map = precision_map["tier_map"]
    mixed_modules = precision_map.get("mixed_modules", {})
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
    param_counts = defaultdict(int)
    mixed_count = 0

    for shard_name in sorted(by_shard):
        shard_path = source_dir / shard_name
        print(f"Reading {shard_path} ...", flush=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in sorted(by_shard[shard_name]):
                if tie_word_embeddings and name == "lm_head.weight":
                    continue

                module_name = name[:-7] if name.endswith(".weight") else None
                if module_name is not None and module_name in mixed_modules:
                    tensor = handle.get_tensor(name)
                    state_entries, info, mixed_param_counts = export_row_mixed_tensor(name, tensor, mixed_modules[module_name])
                    new_state.update(state_entries)
                    quant_info[name] = info
                    mixed_count += 1
                    for bits, n_params in mixed_param_counts.items():
                        param_counts[bits] += n_params
                    counts["row_mixed"] += 1
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
                param_counts[bits] += int(tensor.numel())

    copy_metadata(source_dir, out_dir)
    save_file(new_state, str(out_dir / "model.safetensors"))

    total_quantized_params = sum(param_counts.values())
    weighted_bits = sum(int(bits) * n_params for bits, n_params in param_counts.items() if isinstance(bits, int))
    effective_bits_actual = (weighted_bits / total_quantized_params) if total_quantized_params else 0.0

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
                "mixed_tensors": mixed_count,
                "pruned_tensors": len(pruned_info),
                "bit_counts": {str(k): v for k, v in sorted(counts.items(), key=lambda item: str(item[0]))},
                "bit_param_counts": {str(k): v for k, v in sorted(param_counts.items())},
                "effective_bits_actual": effective_bits_actual,
                "map_summary": precision_map.get("summary", {}),
                "map_settings": precision_map.get("settings", {}),
                "mixed_modules": mixed_modules,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved canonical NANO artifact to {out_dir}", flush=True)
    print(f"Quantized tensors: {len(quant_info)}", flush=True)
    print(f"Mixed tensors: {mixed_count}", flush=True)
    print(f"Pruned tensors: {len(pruned_info)}", flush=True)
    print(f"Bit counts: {dict(sorted(counts.items(), key=lambda item: str(item[0])))}", flush=True)
    print(f"Bit param counts: {dict(sorted(param_counts.items()))}", flush=True)
    print(f"effective_bits_actual: {effective_bits_actual:.6f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a canonical NANO artifact with optional row-mixed sub-int8 tensors.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--precision-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2, 0))
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
