import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_build_row_scores import build_row_scores_for_module, load_reference_model as load_ranking_model
from nano_search_combo_rows import (
    build_candidate_map,
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
from nano_export_from_map_v4 import export_from_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sonar sweep across a family of modules using a standardized row-count schedule."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--module-suffix", default="mlp.up_proj")
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--row-counts", nargs="+", type=int, required=True)
    parser.add_argument("--low-bits", type=int, choices=(6, 4, 2), default=4)
    parser.add_argument("--fixed-spec", action="append", default=[])
    parser.add_argument("--fixed-spec-log", action="append", default=[])
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2), default=8)
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
    parser.add_argument("--artifact-policy", choices=("all", "best", "none"), default="best")
    parser.add_argument("--selection-mode", choices=("first-pass", "smallest"), default="first-pass")
    parser.add_argument(
        "--require-size-improve",
        action="store_true",
        help="Accept layer changes only when they strictly reduce overall artifact size.",
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
    match = re.search(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", module_name)
    if match:
        layer_idx = int(match.group(1))
        suffix = match.group(2).replace("_proj", "")
        legacy = ranking_dir / f"row_scores_layer{layer_idx}_{suffix}_correctsrc.json"
        if legacy.exists():
            return legacy
    return ranking_dir / f"row_scores_{compact_module_name(module_name)}.json"


def artifact_model_size(path: str):
    model_path = Path(path) / "model.safetensors"
    if model_path.exists():
        return model_path.stat().st_size
    return None


def remove_dir(path: Path | None):
    if path is None:
        return
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def main():
    args = parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

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

    fixed_specs = []
    fixed_module_names = set()
    for raw_spec in args.fixed_spec:
        parsed = parse_fixed_spec(raw_spec)
        ranking_data = load_json(Path(parsed["row_ranking_file"]))
        parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
        fixed_specs.append(parsed)
        fixed_module_names.add(parsed["module_name"])
    for log_path in args.fixed_spec_log:
        for parsed in load_fixed_specs_from_log(Path(log_path)):
            ranking_data = load_json(Path(parsed["row_ranking_file"]))
            parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
            fixed_specs.append(parsed)
            fixed_module_names.add(parsed["module_name"])

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

    ranking_model = None
    ranking_tokenizer = None
    ranking_model_path = args.ranking_model or args.source_dir

    current_best_size = artifact_model_size(args.reference_model)
    log = {
        "started_utc": int(time.time()),
        "source_dir": str(source_dir),
        "base_map": args.base_map,
        "reference_model": args.reference_model,
        "reference_kind": args.reference_kind,
        "module_suffix": args.module_suffix,
        "row_counts": args.row_counts,
        "low_bits": args.low_bits,
        "embed_bits": args.embed_bits,
        "artifact_policy": args.artifact_policy,
        "selection_mode": args.selection_mode,
        "require_size_improve": args.require_size_improve,
        "initial_fixed_specs": [
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": spec["row_count"],
                "low_bits": spec["low_bits"],
                "row_indices": spec["row_indices"],
            }
            for spec in fixed_specs
        ],
        "current_best_size": current_best_size,
        "layers": [],
    }

    try:
        kept_final_artifact_dir = None
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
                "candidates": [],
                "accepted": None,
            }

            best_pass = None

            for row_count in args.row_counts:
                variable_row_indices = ranking_to_rows(ranking_data, row_count)
                candidate_map = build_candidate_map(
                    base_map,
                    fixed_specs,
                    module_name,
                    variable_row_indices,
                    args.low_bits,
                )
                label = (
                    f"chain_{len(fixed_specs):02d}_"
                    f"{compact_module_name(module_name)}_{row_count:04d}r_b{args.low_bits}"
                )
                map_path = out_root / "maps" / f"{label}.json"
                artifact_dir = out_root / "artifacts" / label
                export_candidate_map(candidate_map, map_path)

                print(f"\n[SONAR] module={module_name} rows={row_count}", flush=True)
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
                    row = {
                        "row_count": row_count,
                        "size_bytes": size_bytes,
                        "passed": passed,
                        "metrics": metrics,
                        "artifact_dir": str(artifact_dir),
                        "artifact_retained": args.artifact_policy == "all",
                        "map_path": str(map_path),
                    }
                    previous_best = best_pass
                    if args.selection_mode == "first-pass":
                        is_new_layer_best = passed and (best_pass is None)
                    else:
                        is_new_layer_best = passed and (
                            best_pass is None or size_bytes < best_pass["size_bytes"]
                        )
                    if is_new_layer_best:
                        best_pass = row
                    if args.artifact_policy != "all":
                        row["artifact_retained"] = False
                        if is_new_layer_best:
                            if previous_best is not None and previous_best.get("artifact_retained"):
                                prev_path = previous_best.get("artifact_dir")
                                if prev_path:
                                    remove_dir(Path(prev_path))
                                    previous_best["artifact_dir"] = None
                                    previous_best["artifact_retained"] = False
                            if args.artifact_policy == "best":
                                row["artifact_retained"] = True
                            else:
                                remove_dir(artifact_dir)
                                row["artifact_dir"] = None
                        else:
                            remove_dir(artifact_dir)
                            row["artifact_dir"] = None
                    layer_log["candidates"].append(row)
                    print(
                        f"[{'PASS' if passed else 'FAIL'}] size={size_bytes} "
                        f"loss_delta={metrics['loss_delta_mean']:.6f} "
                        f"next={metrics['next_token_agreement']:.3f} "
                        f"greedy={metrics['greedy_token_agreement']:.3f} "
                        f"full={metrics['full_greedy_match_rate']:.3f}",
                        flush=True,
                    )
                    if passed and args.selection_mode == "first-pass":
                        print(f"[PING] first-pass locked at rows={row_count}; stop scanning this layer.", flush=True)
                        break
                finally:
                    if candidate_model is not None:
                        cleanup_model(candidate_model)

            if best_pass is not None and (
                (not args.require_size_improve) or current_best_size is None or best_pass["size_bytes"] < current_best_size
            ):
                accepted_spec = {
                    "module_name": module_name,
                    "row_ranking_file": str(ranking_path),
                    "row_count": best_pass["row_count"],
                    "low_bits": args.low_bits,
                    "row_indices": ranking_to_rows(ranking_data, best_pass["row_count"]),
                }
                fixed_specs.append(accepted_spec)
                if current_best_size is None or best_pass["size_bytes"] < current_best_size:
                    current_best_size = best_pass["size_bytes"]
                if args.artifact_policy == "best":
                    if kept_final_artifact_dir is not None:
                        prev_path = best_pass.get("artifact_dir")
                        if prev_path is None or Path(prev_path) != kept_final_artifact_dir:
                            remove_dir(kept_final_artifact_dir)
                    if best_pass.get("artifact_dir") is not None:
                        kept_final_artifact_dir = Path(best_pass["artifact_dir"])
                layer_log["accepted"] = {
                    "row_count": best_pass["row_count"],
                    "size_bytes": best_pass["size_bytes"],
                    "artifact_dir": best_pass["artifact_dir"],
                    "map_path": best_pass["map_path"],
                }
                print(
                    f"[ACCEPT] module={module_name} rows={best_pass['row_count']} size={best_pass['size_bytes']}",
                    flush=True,
                )
            else:
                if (
                    args.artifact_policy == "best"
                    and best_pass is not None
                    and best_pass.get("artifact_retained")
                    and best_pass.get("artifact_dir") is not None
                ):
                    remove_dir(Path(best_pass["artifact_dir"]))
                    best_pass["artifact_dir"] = None
                    best_pass["artifact_retained"] = False
                print(f"[SKIP] module={module_name} no improving exact-pass candidate", flush=True)

            log["layers"].append(layer_log)
            log["current_best_size"] = current_best_size
            log["accepted_specs"] = [
                {
                    "module_name": spec["module_name"],
                    "row_ranking_file": spec["row_ranking_file"],
                    "row_count": spec["row_count"],
                    "low_bits": spec["low_bits"],
                }
                for spec in fixed_specs
            ]
    finally:
        cleanup_model(reference_model)
        if ranking_model is not None:
            del ranking_model
            gc.collect()

    if args.artifact_policy == "none" and log.get("accepted_specs"):
        final_map = build_candidate_map(
            base_map,
            [
                {
                    "module_name": spec["module_name"],
                    "row_indices": ranking_to_rows(load_json(Path(spec["row_ranking_file"])), spec["row_count"]),
                    "low_bits": spec["low_bits"],
                }
                for spec in log["accepted_specs"]
            ],
            "__none__",
            [],
            args.low_bits,
        )
        del final_map["mixed_modules"]["__none__"]
        final_map["search_policy"]["mixed_modules"].pop("__none__", None)
        final_map_path = out_root / "final_map.json"
        export_candidate_map(final_map, final_map_path)
        final_artifact_dir = out_root / "final_artifact"
        export_from_map(source_dir, final_map_path, final_artifact_dir, args.embed_bits)
        log["final_artifact_dir"] = str(final_artifact_dir)
    elif args.artifact_policy == "best" and kept_final_artifact_dir is not None:
        log["final_artifact_dir"] = str(kept_final_artifact_dir)

    log["finished_utc"] = int(time.time())
    log["elapsed_sec"] = round(time.time() - started, 2)
    out_path = out_root / "sonar_results.json"
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nSaved sonar log to {out_path}", flush=True)
    print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
