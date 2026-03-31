import argparse
import gc
import json
import re
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_build_row_scores import build_row_scores_for_module, load_reference_model as load_ranking_model
from nano_export_from_map_v4 import export_from_map, pack_rows_with_row_scales, quantize_int8_with_scale
from nano_inference_direct import MixedRowNanoLinear, NanoLinear
from nano_search_combo_rows import (
    compact_module_name,
    load_fixed_specs_from_log,
    parse_fixed_spec,
    ranking_to_rows,
)
from nano_search_subint8 import (
    cleanup_model,
    evaluate_pair,
    export_candidate_map,
    load_json,
    load_nano_maybe_quiet,
    load_reference_model,
    passes_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-command ping sweep over layers with fixed row ladder and multi-bit ladder.\n"
            "Per-layer policy: test bits/rows in configured order and lock the first exact pass."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--module-suffix", default="mlp.gate_proj")
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--row-counts", nargs="+", type=int, default=[8192, 4096, 2048, 1024, 512])
    parser.add_argument("--bit-ladder", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2), default=8)
    parser.add_argument("--fixed-spec", action="append", default=[])
    parser.add_argument("--fixed-spec-log", action="append", default=[])
    parser.add_argument("--ranking-dir", default="local_runs")
    parser.add_argument("--ranking-model")
    parser.add_argument("--ranking-kind", choices=("nano", "hf"), default="hf")
    parser.add_argument("--ranking-alpha", type=float, default=0.30)
    parser.add_argument("--ranking-beta", type=float, default=0.30)
    parser.add_argument("--ranking-gamma", type=float, default=0.40)
    parser.add_argument("--ranking-skip-sensitivity", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--greedy-steps", type=int, default=4)
    parser.add_argument("--max-loss-delta", type=float, default=0.02)
    parser.add_argument("--min-next-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-greedy-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-full-greedy-match", type=float, default=1.0)
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    parser.add_argument("--quiet-loads", action="store_true")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument(
        "--candidate-eval-mode",
        choices=("ram", "disk"),
        default="ram",
        help="Evaluate candidate quality in RAM (fast, no per-candidate export) or via full disk export.",
    )
    parser.add_argument(
        "--candidate-map-policy",
        choices=("keep", "none"),
        default="keep",
        help="Keep or skip per-candidate map JSON files during sweep.",
    )
    parser.add_argument(
        "--skip-final-export",
        action="store_true",
        help="Do not export final artifact; only emit final_map and ping log.",
    )
    parser.add_argument(
        "--require-size-improve",
        action="store_true",
        help="Apply an accepted layer change only if it strictly reduces global artifact size.",
    )
    parser.add_argument(
        "--keep-layer-artifacts",
        action="store_true",
        help="Keep accepted per-layer artifacts. Default keeps only final artifact.",
    )
    return parser.parse_args()


def module_sort_key(name: str):
    match = re.search(r"model\.layers\.(\d+)\.", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def infer_modules(base_map: dict, module_suffix: str):
    names = [name for name in base_map["tier_map"] if name.endswith(module_suffix)]
    return sorted(names, key=module_sort_key)


def filter_modules_by_layer(modules: list[str], layer_start: int | None, layer_end: int | None):
    if layer_start is None and layer_end is None:
        return modules
    filtered = []
    for name in modules:
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if not match:
            continue
        idx = int(match.group(1))
        if layer_start is not None and idx < layer_start:
            continue
        if layer_end is not None and idx > layer_end:
            continue
        filtered.append(name)
    return filtered


def default_ranking_path(ranking_dir: Path, module_name: str):
    match = re.search(
        r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        module_name,
    )
    if match:
        layer_idx = int(match.group(1))
        suffix = match.group(2).replace("_proj", "")
        legacy = ranking_dir / f"row_scores_layer{layer_idx}_{suffix}_correctsrc.json"
        if legacy.exists():
            return legacy
    return ranking_dir / f"row_scores_{compact_module_name(module_name)}.json"


def artifact_model_size(path: Path):
    model_path = path / "model.safetensors"
    if model_path.exists():
        return model_path.stat().st_size
    return None


def remove_dir(path: Path | None):
    if path is None:
        return
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def parse_fixed_specs(args):
    fixed_specs = []
    fixed_zero_modules = set()
    for raw_spec in args.fixed_spec:
        parsed = parse_fixed_spec(raw_spec)
        ranking_data = load_json(Path(parsed["row_ranking_file"]))
        parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
        if int(parsed["low_bits"]) == 0:
            fixed_zero_modules.add(parsed["module_name"])
        else:
            fixed_specs.append(parsed)
    for log_path in args.fixed_spec_log:
        data = load_json(Path(log_path))
        if "accepted_zero_modules" in data:
            fixed_zero_modules.update(data.get("accepted_zero_modules", []))
        parsed_specs = None
        if "accepted_specs" in data:
            parsed_specs = [
                {
                    "module_name": spec["module_name"],
                    "row_ranking_file": spec["row_ranking_file"],
                    "row_count": int(spec["row_count"]),
                    "low_bits": int(spec["low_bits"]),
                }
                for spec in data.get("accepted_specs", [])
            ]
        if parsed_specs is None:
            parsed_specs = load_fixed_specs_from_log(Path(log_path))
        for parsed in parsed_specs:
            ranking_data = load_json(Path(parsed["row_ranking_file"]))
            parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
            if int(parsed["low_bits"]) == 0:
                fixed_zero_modules.add(parsed["module_name"])
            else:
                fixed_specs.append(parsed)
    return fixed_specs, fixed_zero_modules


def _get_child(module, name: str):
    if name.isdigit():
        return module[int(name)]
    return getattr(module, name)


def _set_child(module, name: str, value):
    if name.isdigit():
        module[int(name)] = value
    else:
        setattr(module, name, value)


def resolve_module_slot(model, module_name: str):
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    child_name = parts[-1]
    current = _get_child(parent, child_name)
    return parent, child_name, current


def build_source_tensor_cache(source_dir: Path):
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = dict(index.get("weight_map", {}))
    else:
        model_file = source_dir / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Missing model.safetensors.index.json and model.safetensors under {source_dir}"
            )
        weight_map = {}
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = model_file.name
    return {"source_dir": source_dir, "weight_map": weight_map, "cache": {}}


def get_source_module_tensors(cache: dict, module_name: str):
    hit = cache["cache"].get(module_name)
    if hit is not None:
        return hit

    weight_key = f"{module_name}.weight"
    bias_key = f"{module_name}.bias"
    weight_map = cache["weight_map"]
    if weight_key not in weight_map:
        raise KeyError(f"Missing source weight tensor: {weight_key}")

    by_shard = {}
    by_shard.setdefault(weight_map[weight_key], []).append(weight_key)
    if bias_key in weight_map:
        by_shard.setdefault(weight_map[bias_key], []).append(bias_key)

    tensors = {}
    for shard_name, keys in by_shard.items():
        shard_path = cache["source_dir"] / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in keys:
                if key.endswith(".weight"):
                    tensors["weight"] = handle.get_tensor(key)
                elif key.endswith(".bias"):
                    tensors["bias"] = handle.get_tensor(key)

    if "weight" not in tensors:
        raise RuntimeError(f"Failed to load source weight for {module_name}")
    cache["cache"][module_name] = tensors
    return tensors


def build_runtime_module(module_name: str, source_tensor_cache: dict, low_bits: int, row_indices: list[int] | None):
    source_tensors = get_source_module_tensors(source_tensor_cache, module_name)
    weight = source_tensors["weight"].to(torch.float32)
    bias = source_tensors.get("bias")

    if int(low_bits) == 8:
        return None

    if int(low_bits) == 0:
        zero_packed = torch.zeros(weight.shape, dtype=torch.int8)
        return NanoLinear(
            module_name,
            zero_packed,
            0.0,
            8,
            list(weight.shape),
            bias=None,
        )

    rows = sorted({int(x) for x in (row_indices or [])})
    if not rows:
        raise ValueError(f"row_indices empty for low_bits={low_bits} on {module_name}")
    out_features = int(weight.shape[0])
    if rows[0] < 0 or rows[-1] >= out_features:
        raise ValueError(f"row_indices out of bounds for {module_name}: min={rows[0]} max={rows[-1]} out={out_features}")

    low_row_indices = torch.tensor(rows, dtype=torch.int32)
    row_set = set(rows)
    base_rows = [idx for idx in range(out_features) if idx not in row_set]
    base_row_indices = torch.tensor(base_rows, dtype=torch.int32)

    if weight.numel():
        base_scale = float((weight.abs().max() / 127.0).item())
    else:
        base_scale = 0.0
    base_quant = quantize_int8_with_scale(weight, base_scale)
    if len(base_rows) > 0:
        base_packed = base_quant[base_row_indices.to(torch.long)]
    else:
        base_packed = torch.zeros((0, weight.shape[1]), dtype=torch.int8)

    low_tensor = weight[low_row_indices.to(torch.long)]
    low_packed, low_scales, low_shape, _ = pack_rows_with_row_scales(low_tensor, int(low_bits), group_size=None)

    return MixedRowNanoLinear(
        module_name,
        base_packed,
        float(base_scale),
        8,
        list(base_packed.shape),
        base_row_indices,
        low_packed,
        low_scales,
        int(low_bits),
        list(low_shape),
        list(low_shape),
        0,
        low_row_indices,
        list(weight.shape),
        residual_indices=None,
        residual_values=None,
        bias=bias,
    )


def apply_runtime_override(model, module_name: str, source_tensor_cache: dict, low_bits: int, row_indices: list[int] | None):
    parent, child_name, previous = resolve_module_slot(model, module_name)
    replacement = build_runtime_module(module_name, source_tensor_cache, int(low_bits), row_indices)
    if replacement is None:
        return previous
    _set_child(parent, child_name, replacement)
    return previous


def evaluate_candidate_map_ram(
    args,
    map_payload: dict,
    label: str,
    out_root: Path,
    reference_model,
    candidate_model,
    tokenizer,
    prompts: list[str],
    module_name: str,
    variable_low_bits: int,
    variable_row_indices: list[int] | None,
    source_tensor_cache: dict,
):
    map_path = None
    if args.candidate_map_policy == "keep":
        maps_dir = out_root / "maps"
        map_path = maps_dir / f"{label}.json"
        maps_dir.mkdir(parents=True, exist_ok=True)
        export_candidate_map(map_payload, map_path)

    parent, child_name, previous = resolve_module_slot(candidate_model, module_name)
    replacement = build_runtime_module(
        module_name,
        source_tensor_cache,
        int(variable_low_bits),
        variable_row_indices,
    )
    if replacement is not None:
        _set_child(parent, child_name, replacement)

    metrics = evaluate_pair(
        reference_model,
        candidate_model,
        tokenizer,
        prompts,
        args.max_length,
        args.greedy_steps,
        stop_on_first_exact_mismatch=(
            args.min_next_token_agreement >= 1.0
            and args.min_greedy_token_agreement >= 1.0
            and args.min_full_greedy_match >= 1.0
        ),
    )
    passed = passes_thresholds(metrics, args)
    return {
        "passed": passed,
        "size_bytes": None,
        "metrics": metrics,
        "map_path": str(map_path) if map_path is not None else None,
        "artifact_dir": None,
        "_parent": parent,
        "_child_name": child_name,
        "_previous": previous,
        "_patched": replacement is not None,
    }


def build_state_map(
    base_map: dict,
    fixed_specs: list[dict],
    fixed_zero_modules: set[str],
    variable_module: str | None = None,
    variable_low_bits: int | None = None,
    variable_row_indices: list[int] | None = None,
):
    candidate = deepcopy(base_map)
    tier_map = dict(candidate.get("tier_map", {}))
    for module_name in fixed_zero_modules:
        tier_map[module_name] = 0

    mixed_modules = {}
    for spec in fixed_specs:
        mixed_modules[spec["module_name"]] = {
            "scheme": "row_mixed",
            "base_bits": 8,
            "low_bits": int(spec["low_bits"]),
            "row_indices": [int(x) for x in spec["row_indices"]],
            "residual_topk": 0,
        }

    if variable_module is not None and variable_low_bits is not None:
        if int(variable_low_bits) == 0:
            tier_map[variable_module] = 0
            mixed_modules.pop(variable_module, None)
        else:
            mixed_modules[variable_module] = {
                "scheme": "row_mixed",
                "base_bits": 8,
                "low_bits": int(variable_low_bits),
                "row_indices": [int(x) for x in (variable_row_indices or [])],
                "residual_topk": 0,
            }

    for module_name in list(mixed_modules.keys()):
        if int(tier_map.get(module_name, 8)) == 0:
            mixed_modules.pop(module_name, None)

    candidate["tier_map"] = tier_map
    candidate["mixed_modules"] = mixed_modules
    candidate["search_policy"] = {
        "embed_bits": candidate.get("search_policy", {}).get("embed_bits", 8),
        "mixed_modules": mixed_modules,
        "forced_zero_modules": sorted(fixed_zero_modules),
    }
    return candidate


def evaluate_candidate_map(
    args,
    source_dir: Path,
    map_payload: dict,
    label: str,
    out_root: Path,
    reference_model,
    tokenizer,
    prompts: list[str],
):
    maps_dir = out_root / "maps"
    scratch_dir = out_root / "scratch"
    artifact_dir = scratch_dir / label
    map_path = None
    if args.candidate_map_policy == "keep":
        map_path = maps_dir / f"{label}.json"
        maps_dir.mkdir(parents=True, exist_ok=True)
        export_candidate_map(map_payload, map_path)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if map_path is None:
        map_path = maps_dir / f"__tmp_{label}.json"
        maps_dir.mkdir(parents=True, exist_ok=True)
        export_candidate_map(map_payload, map_path)
    export_from_map(source_dir, map_path, artifact_dir, args.embed_bits)
    candidate_model = None
    try:
        candidate_model, _ = load_nano_maybe_quiet(
            str(artifact_dir),
            args.quiet_loads,
            args.fp32_rmsnorm,
        )
        metrics = evaluate_pair(
            reference_model,
            candidate_model,
            tokenizer,
            prompts,
            args.max_length,
            args.greedy_steps,
            stop_on_first_exact_mismatch=(
                args.min_next_token_agreement >= 1.0
                and args.min_greedy_token_agreement >= 1.0
                and args.min_full_greedy_match >= 1.0
            ),
        )
        size_bytes = (artifact_dir / "model.safetensors").stat().st_size
        passed = passes_thresholds(metrics, args)
        return {
            "passed": passed,
            "size_bytes": size_bytes,
            "metrics": metrics,
            "map_path": str(map_path) if args.candidate_map_policy == "keep" else None,
            "artifact_dir": str(artifact_dir),
        }
    finally:
        if args.candidate_map_policy == "none" and map_path is not None and map_path.name.startswith("__tmp_"):
            map_path.unlink(missing_ok=True)
        if candidate_model is not None:
            cleanup_model(candidate_model)


def main():
    args = parse_args()
    valid_bits = {0, 2, 4, 6, 8}
    bad_bits = [b for b in args.bit_ladder if b not in valid_bits]
    if bad_bits:
        raise ValueError(f"Unsupported bits in bit_ladder: {bad_bits}. Allowed: {sorted(valid_bits)}")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.candidate_eval_mode == "ram" and args.reference_kind != "nano":
        raise ValueError("--candidate-eval-mode ram currently requires --reference-kind nano.")
    if args.candidate_eval_mode == "ram" and args.require_size_improve:
        print(
            "[WARN] --require-size-improve is ignored in RAM mode; exact size is computed at final export only.",
            flush=True,
        )
        args.require_size_improve = False

    started = time.time()
    source_dir = Path(args.source_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ranking_dir = Path(args.ranking_dir)
    if not ranking_dir.is_absolute():
        ranking_dir = ROOT / ranking_dir
    ranking_dir.mkdir(parents=True, exist_ok=True)

    base_map = load_json(Path(args.base_map))
    prompts = load_json(Path(args.prompts_file))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts-file must contain a non-empty JSON list.")

    if args.modules:
        modules = list(args.modules)
    else:
        modules = infer_modules(base_map, args.module_suffix)
    modules = filter_modules_by_layer(modules, args.layer_start, args.layer_end)

    fixed_specs, fixed_zero_modules = parse_fixed_specs(args)
    fixed_module_names = {spec["module_name"] for spec in fixed_specs}.union(fixed_zero_modules)
    modules = [name for name in modules if name not in fixed_module_names]

    print(f"Loading reference model ({args.reference_kind}): {args.reference_model}", flush=True)
    if args.reference_kind == "nano":
        reference_model, tokenizer = load_nano_maybe_quiet(
            args.reference_model,
            args.quiet_loads,
            args.fp32_rmsnorm,
        )
    else:
        reference_model, tokenizer = load_reference_model(
            args.reference_model,
            args.reference_kind,
            args.fp32_rmsnorm,
        )

    candidate_model = None
    source_tensor_cache = None
    if args.candidate_eval_mode == "ram":
        print(f"Loading candidate working model (nano, RAM mode): {args.reference_model}", flush=True)
        candidate_model, _ = load_nano_maybe_quiet(
            args.reference_model,
            args.quiet_loads,
            args.fp32_rmsnorm,
        )
        source_tensor_cache = build_source_tensor_cache(source_dir)
        if fixed_zero_modules or fixed_specs:
            print(
                f"Applying initial fixed overrides in RAM: zero={len(fixed_zero_modules)} mixed={len(fixed_specs)}",
                flush=True,
            )
        for module_name in sorted(fixed_zero_modules):
            apply_runtime_override(
                candidate_model,
                module_name,
                source_tensor_cache,
                low_bits=0,
                row_indices=[],
            )
        for spec in fixed_specs:
            apply_runtime_override(
                candidate_model,
                spec["module_name"],
                source_tensor_cache,
                low_bits=int(spec["low_bits"]),
                row_indices=spec["row_indices"],
            )

    ranking_model = None
    ranking_tokenizer = None
    ranking_model_path = args.ranking_model or args.source_dir

    current_best_size = artifact_model_size(Path(args.reference_model))
    layer_artifacts = []
    log = {
        "started_utc": int(time.time()),
        "source_dir": str(source_dir),
        "base_map": args.base_map,
        "reference_model": args.reference_model,
        "reference_kind": args.reference_kind,
        "module_suffix": args.module_suffix,
        "bit_ladder": args.bit_ladder,
        "row_counts": args.row_counts,
        "embed_bits": args.embed_bits,
        "candidate_eval_mode": args.candidate_eval_mode,
        "candidate_map_policy": args.candidate_map_policy,
        "skip_final_export": args.skip_final_export,
        "require_size_improve": args.require_size_improve,
        "initial_fixed_specs": [
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": spec["row_count"],
                "low_bits": spec["low_bits"],
            }
            for spec in fixed_specs
        ],
        "initial_zero_modules": sorted(fixed_zero_modules),
        "current_best_size": current_best_size,
        "layers": [],
    }

    try:
        for module_name in modules:
            ranking_path = default_ranking_path(ranking_dir, module_name)
            if not ranking_path.exists():
                if ranking_model is None:
                    print(f"Loading ranking model ({args.ranking_kind}): {ranking_model_path}", flush=True)
                    ranking_model, ranking_tokenizer = load_ranking_model(
                        ranking_model_path,
                        args.ranking_kind,
                        args.fp32_rmsnorm,
                    )
                rows = build_row_scores_for_module(
                    ranking_model,
                    ranking_tokenizer,
                    module_name,
                    prompts,
                    args.max_length,
                    args.ranking_alpha,
                    args.ranking_beta,
                    args.ranking_gamma,
                    args.ranking_skip_sensitivity,
                )
                ranking_payload = {
                    "reference_model": ranking_model_path,
                    "reference_kind": args.ranking_kind,
                    "module_name": module_name,
                    "created_utc": int(time.time()),
                    "weights": {
                        "alpha": args.ranking_alpha,
                        "beta": args.ranking_beta,
                        "gamma": args.ranking_gamma,
                    },
                    "settings": {
                        "max_length": args.max_length,
                        "num_prompts": len(prompts),
                        "skip_sensitivity": args.ranking_skip_sensitivity,
                    },
                    "rows": rows,
                }
                ranking_path.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")
                print(f"[RANK] saved {ranking_path}", flush=True)

            ranking_data = load_json(ranking_path)
            layer_log = {
                "module_name": module_name,
                "ranking_file": str(ranking_path),
                "attempts": [],
                "accepted": None,
            }

            accepted = None
            for bits in args.bit_ladder:
                if bits == 8:
                    accepted = {"bits": 8, "row_count": 0, "reason": "explicit-8-in-ladder"}
                    break

                if bits == 0:
                    label = f"ping_{len(fixed_specs):02d}_{compact_module_name(module_name)}_b0"
                    candidate_map = build_state_map(
                        base_map,
                        fixed_specs,
                        fixed_zero_modules,
                        variable_module=module_name,
                        variable_low_bits=0,
                    )
                    if args.candidate_eval_mode == "ram":
                        result = evaluate_candidate_map_ram(
                            args,
                            candidate_map,
                            label,
                            out_root,
                            reference_model,
                            candidate_model,
                            tokenizer,
                            prompts,
                            module_name,
                            variable_low_bits=0,
                            variable_row_indices=[],
                            source_tensor_cache=source_tensor_cache,
                        )
                    else:
                        result = evaluate_candidate_map(
                            args,
                            source_dir,
                            candidate_map,
                            label,
                            out_root,
                            reference_model,
                            tokenizer,
                            prompts,
                        )
                    result.update({"bits": 0, "row_count": 0})
                    ram_parent = result.pop("_parent", None)
                    ram_child = result.pop("_child_name", None)
                    ram_prev = result.pop("_previous", None)
                    ram_patched = result.pop("_patched", False)
                    layer_log["attempts"].append(result)
                    size_repr = result["size_bytes"] if result["size_bytes"] is not None else "n/a"
                    print(
                        f"[PING] module={module_name} bits=0 row=0 "
                        f"{'PASS' if result['passed'] else 'FAIL'} size={size_repr}",
                        flush=True,
                    )
                    if result["passed"]:
                        accepted = result
                        break
                    if args.candidate_eval_mode == "ram":
                        if ram_patched:
                            _set_child(ram_parent, ram_child, ram_prev)
                    elif result["artifact_dir"]:
                        remove_dir(Path(result["artifact_dir"]))
                    continue

                for row_count in args.row_counts:
                    row_indices = ranking_to_rows(ranking_data, row_count)
                    label = f"ping_{len(fixed_specs):02d}_{compact_module_name(module_name)}_b{bits}_r{row_count}"
                    candidate_map = build_state_map(
                        base_map,
                        fixed_specs,
                        fixed_zero_modules,
                        variable_module=module_name,
                        variable_low_bits=bits,
                        variable_row_indices=row_indices,
                    )
                    if args.candidate_eval_mode == "ram":
                        result = evaluate_candidate_map_ram(
                            args,
                            candidate_map,
                            label,
                            out_root,
                            reference_model,
                            candidate_model,
                            tokenizer,
                            prompts,
                            module_name,
                            variable_low_bits=int(bits),
                            variable_row_indices=row_indices,
                            source_tensor_cache=source_tensor_cache,
                        )
                    else:
                        result = evaluate_candidate_map(
                            args,
                            source_dir,
                            candidate_map,
                            label,
                            out_root,
                            reference_model,
                            tokenizer,
                            prompts,
                        )
                    result.update({"bits": bits, "row_count": row_count})
                    ram_parent = result.pop("_parent", None)
                    ram_child = result.pop("_child_name", None)
                    ram_prev = result.pop("_previous", None)
                    ram_patched = result.pop("_patched", False)
                    layer_log["attempts"].append(result)
                    size_repr = result["size_bytes"] if result["size_bytes"] is not None else "n/a"
                    print(
                        f"[PING] module={module_name} bits={bits} row={row_count} "
                        f"{'PASS' if result['passed'] else 'FAIL'} size={size_repr}",
                        flush=True,
                    )
                    if result["passed"]:
                        accepted = result
                        break
                    if args.candidate_eval_mode == "ram":
                        if ram_patched:
                            _set_child(ram_parent, ram_child, ram_prev)
                    elif result["artifact_dir"]:
                        remove_dir(Path(result["artifact_dir"]))
                if accepted is not None:
                    break

            if accepted is None:
                layer_log["accepted"] = {"bits": 8, "row_count": 0, "reason": "no-pass"}
                print(f"[LOCK] module={module_name} -> keep 8-bit (no passing candidate)", flush=True)
                log["layers"].append(layer_log)
                continue

            if accepted.get("bits") == 8:
                layer_log["accepted"] = accepted
                print(f"[LOCK] module={module_name} -> keep 8-bit", flush=True)
                log["layers"].append(layer_log)
                continue

            if (
                args.require_size_improve
                and accepted.get("size_bytes") is not None
                and current_best_size is not None
                and accepted["size_bytes"] >= current_best_size
            ):
                layer_log["accepted"] = {
                    "bits": 8,
                    "row_count": 0,
                    "reason": "pass-but-no-global-size-improve",
                    "candidate_bits": accepted["bits"],
                    "candidate_row_count": accepted["row_count"],
                    "candidate_size_bytes": accepted["size_bytes"],
                }
                print(
                    f"[LOCK] module={module_name} pass at b{accepted['bits']}/r{accepted['row_count']} "
                    f"but skipped (size not improved)",
                    flush=True,
                )
                if accepted.get("artifact_dir"):
                    remove_dir(Path(accepted["artifact_dir"]))
                log["layers"].append(layer_log)
                continue

            if accepted["bits"] == 0:
                fixed_zero_modules.add(module_name)
                accepted_payload = {
                    "bits": 0,
                    "row_count": 0,
                    "map_path": accepted["map_path"],
                }
                if accepted.get("size_bytes") is not None:
                    accepted_payload["size_bytes"] = accepted["size_bytes"]
                if accepted.get("artifact_dir"):
                    accepted_payload["artifact_dir"] = accepted["artifact_dir"]
                layer_log["accepted"] = accepted_payload
            else:
                fixed_specs.append(
                    {
                        "module_name": module_name,
                        "row_ranking_file": str(ranking_path),
                        "row_count": int(accepted["row_count"]),
                        "low_bits": int(accepted["bits"]),
                        "row_indices": ranking_to_rows(ranking_data, int(accepted["row_count"])),
                    }
                )
                accepted_payload = {
                    "bits": int(accepted["bits"]),
                    "row_count": int(accepted["row_count"]),
                    "map_path": accepted["map_path"],
                }
                if accepted.get("size_bytes") is not None:
                    accepted_payload["size_bytes"] = accepted["size_bytes"]
                if accepted.get("artifact_dir"):
                    accepted_payload["artifact_dir"] = accepted["artifact_dir"]
                layer_log["accepted"] = accepted_payload

            if accepted.get("size_bytes") is not None:
                current_best_size = accepted["size_bytes"]
                log["current_best_size"] = current_best_size
            if accepted.get("artifact_dir"):
                layer_artifacts.append(Path(accepted["artifact_dir"]))
            size_suffix = ""
            if accepted.get("size_bytes") is not None:
                size_suffix = f" size={accepted['size_bytes']}"
            print(
                f"[LOCK] module={module_name} -> b{layer_log['accepted']['bits']} "
                f"r{layer_log['accepted']['row_count']}{size_suffix}",
                flush=True,
            )
            log["layers"].append(layer_log)

    finally:
        cleanup_model(reference_model)
        if candidate_model is not None:
            cleanup_model(candidate_model)
        if ranking_model is not None:
            del ranking_model
            gc.collect()

    final_map = build_state_map(base_map, fixed_specs, fixed_zero_modules)
    final_map_path = out_root / "final_map.json"
    export_candidate_map(final_map, final_map_path)
    final_artifact_dir = out_root / "final_artifact"
    final_size_bytes = None
    if not args.skip_final_export:
        remove_dir(final_artifact_dir)
        export_from_map(source_dir, final_map_path, final_artifact_dir, args.embed_bits)
        final_size_bytes = artifact_model_size(final_artifact_dir)

    if not args.keep_layer_artifacts:
        for artifact in layer_artifacts:
            remove_dir(artifact)
        remove_dir(out_root / "scratch")

    log["accepted_specs"] = [
        {
            "module_name": spec["module_name"],
            "row_ranking_file": spec["row_ranking_file"],
            "row_count": spec["row_count"],
            "low_bits": spec["low_bits"],
        }
        for spec in fixed_specs
    ]
    log["accepted_zero_modules"] = sorted(fixed_zero_modules)
    log["final_map"] = str(final_map_path)
    log["final_artifact_dir"] = str(final_artifact_dir) if not args.skip_final_export else None
    log["final_size_bytes"] = final_size_bytes
    log["finished_utc"] = int(time.time())
    log["elapsed_sec"] = round(time.time() - started, 2)
    log_path = out_root / "ping_results.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nSaved ping log to {log_path}", flush=True)
    if args.skip_final_export:
        print("Final export skipped (--skip-final-export).", flush=True)
    else:
        print(f"Final artifact: {final_artifact_dir}", flush=True)
        print(f"Final size: {log['final_size_bytes']}", flush=True)
    print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
