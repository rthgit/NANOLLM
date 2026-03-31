import argparse
import gc
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_export_from_map_v4 import export_from_map
from nano_search_subint8 import (
    cleanup_model,
    evaluate_pair,
    export_candidate_map,
    load_json,
    load_nano_maybe_quiet,
    load_reference_model,
    passes_thresholds,
)


def parse_fixed_spec(spec: str):
    parts = spec.split("::")
    if len(parts) != 4:
        raise ValueError(
            "fixed-spec must be formatted as "
            "'module_name::row_ranking_file::row_count::low_bits'"
        )
    module_name, ranking_file, row_count, low_bits = parts
    return {
        "module_name": module_name,
        "row_ranking_file": ranking_file,
        "row_count": int(row_count),
        "low_bits": int(low_bits),
    }


def load_fixed_specs_from_log(log_path: Path):
    data = load_json(log_path)
    specs = []
    if "fixed_specs" in data and "variable_module" in data:
        specs.extend(
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": int(spec["row_count"]),
                "low_bits": int(spec["low_bits"]),
            }
            for spec in data["fixed_specs"]
        )
        best_pass = data.get("best_pass")
        if best_pass is not None:
            specs.append(
                {
                    "module_name": data["variable_module"],
                    "row_ranking_file": data["variable_ranking_file"],
                    "row_count": int(best_pass["variable_row_count"]),
                    "low_bits": int(data["settings"]["variable_low_bits"]),
                }
            )
        return specs
    if "initial_fixed_specs" in data and "module_template" in data:
        specs.extend(
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": int(spec["row_count"]),
                "low_bits": int(spec["low_bits"]),
            }
            for spec in data["initial_fixed_specs"]
        )
        for layer in data.get("accepted_layers", []):
            specs.append(
                {
                    "module_name": data["module_template"].format(layer=layer),
                    "row_ranking_file": data["donor_ranking_file"],
                    "row_count": int(data["full_row_count"]),
                    "low_bits": int(data["low_bits"]),
                }
            )
        return specs
    raise ValueError(f"Unsupported fixed-spec log format: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search cumulative row-mixed candidates with fixed modules plus one variable module."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--variable-module", required=True)
    parser.add_argument("--variable-ranking-file", required=True)
    parser.add_argument("--variable-row-counts", nargs="+", type=int, required=True)
    parser.add_argument("--variable-low-bits", type=int, choices=(6, 4, 2), default=4)
    parser.add_argument("--fixed-spec", action="append", default=[])
    parser.add_argument("--fixed-spec-log", action="append", default=[])
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2), default=8)
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
    parser.add_argument(
        "--stop-on-first-pass",
        action="store_true",
        help="Stop evaluating variable_row_counts after the first passing candidate.",
    )
    return parser.parse_args()


def ranking_to_rows(ranking_data: dict, row_count: int):
    rows = ranking_data["rows"]
    if row_count <= 0 or row_count > len(rows):
        raise ValueError(f"row_count out of range: {row_count} for ranking with {len(rows)} rows")
    return [int(row["row_index"]) for row in rows[:row_count]]


def build_candidate_map(
    base_map: dict,
    fixed_specs: list[dict],
    variable_module: str,
    variable_row_indices: list[int],
    variable_low_bits: int,
):
    candidate = deepcopy(base_map)
    mixed_modules = deepcopy(candidate.get("mixed_modules", {}))
    for spec in fixed_specs:
        mixed_modules[spec["module_name"]] = {
            "scheme": "row_mixed",
            "base_bits": 8,
            "low_bits": spec["low_bits"],
            "row_indices": spec["row_indices"],
            "residual_topk": 0,
        }
    mixed_modules[variable_module] = {
        "scheme": "row_mixed",
        "base_bits": 8,
        "low_bits": variable_low_bits,
        "row_indices": variable_row_indices,
        "residual_topk": 0,
    }
    candidate["mixed_modules"] = mixed_modules
    candidate["search_policy"] = {
        "embed_bits": candidate.get("search_policy", {}).get("embed_bits", 8),
        "mixed_modules": mixed_modules,
    }
    return candidate


def spec_label(spec: dict):
    return f"{compact_module_name(spec['module_name'])}_{spec['row_count']:04d}r_b{spec['low_bits']}"


def compact_module_name(name: str):
    result = name
    result = result.replace("model.layers.", "l")
    result = result.replace(".self_attn.", "_a_")
    result = result.replace(".mlp.", "_m_")
    result = result.replace(".embed_tokens", "_embed")
    result = result.replace(".", "_")
    return result


def remove_dir(path: Path | None):
    if path is None:
        return
    if not path.exists():
        return
    for _ in range(5):
        try:
            gc.collect()
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.5)
    shutil.rmtree(path, ignore_errors=True)


def main():
    args = parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    started = time.time()
    source_dir = Path(args.source_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    base_map = load_json(Path(args.base_map))
    prompts = load_json(Path(args.prompts_file))
    variable_ranking = load_json(Path(args.variable_ranking_file))
    fixed_specs = []
    for raw_spec in args.fixed_spec:
        fixed_specs.append(parse_fixed_spec(raw_spec))
    for log_path in args.fixed_spec_log:
        fixed_specs.extend(load_fixed_specs_from_log(Path(log_path)))
    for parsed in fixed_specs:
        ranking_data = load_json(Path(parsed["row_ranking_file"]))
        parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])

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

    log = {
        "started_utc": int(time.time()),
        "source_dir": str(source_dir),
        "base_map": args.base_map,
        "reference_model": args.reference_model,
        "reference_kind": args.reference_kind,
        "fixed_specs": [
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": spec["row_count"],
                "low_bits": spec["low_bits"],
                "row_indices": spec["row_indices"],
            }
            for spec in fixed_specs
        ],
        "variable_module": args.variable_module,
        "variable_ranking_file": args.variable_ranking_file,
        "settings": {
            "variable_row_counts": args.variable_row_counts,
            "variable_low_bits": args.variable_low_bits,
            "embed_bits": args.embed_bits,
            "artifact_policy": args.artifact_policy,
            "stop_on_first_pass": args.stop_on_first_pass,
            "prompts_file": args.prompts_file,
            "max_length": args.max_length,
            "greedy_steps": args.greedy_steps,
        },
        "results": [],
    }

    try:
        fixed_label = "__".join(spec_label(spec) for spec in fixed_specs) or "baseline"
        kept_best_artifact_dir = None
        best_pass = None
        for row_count in args.variable_row_counts:
            variable_row_indices = ranking_to_rows(variable_ranking, row_count)
            candidate_map = build_candidate_map(
                base_map,
                fixed_specs,
                args.variable_module,
                variable_row_indices,
                args.variable_low_bits,
            )
            label = (
                f"chain_{len(fixed_specs):02d}_"
                f"{compact_module_name(args.variable_module)}_{row_count:04d}r_b{args.variable_low_bits}"
            )
            map_path = out_root / "maps" / f"{label}.json"
            artifact_dir = out_root / "artifacts" / label
            export_candidate_map(candidate_map, map_path)

            print(
                f"\n[COMBO] fixed={fixed_label} variable={args.variable_module} rows={row_count}",
                flush=True,
            )
            export_from_map(source_dir, map_path, artifact_dir, args.embed_bits)

            candidate_model = None
            remove_candidate_artifact = False
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
                    "variable_row_count": row_count,
                    "variable_row_indices": variable_row_indices,
                    "size_bytes": size_bytes,
                    "metrics": metrics,
                    "passed": passed,
                    "artifact_dir": str(artifact_dir),
                    "artifact_retained": args.artifact_policy == "all",
                    "map_path": str(map_path),
                }
                if args.artifact_policy != "all":
                    row["artifact_retained"] = False
                    is_new_best = passed and (
                        best_pass is None or size_bytes < best_pass["size_bytes"]
                    )
                    if is_new_best:
                        if args.artifact_policy == "best":
                            if kept_best_artifact_dir is not None and kept_best_artifact_dir != artifact_dir:
                                remove_dir(kept_best_artifact_dir)
                            kept_best_artifact_dir = artifact_dir
                            row["artifact_retained"] = True
                        else:
                            remove_candidate_artifact = True
                            row["artifact_dir"] = None
                        best_pass = row
                    else:
                        remove_candidate_artifact = True
                        row["artifact_dir"] = None
                log["results"].append(row)
                print(
                    f"[{'PASS' if passed else 'FAIL'}] size={size_bytes} "
                    f"loss_delta={metrics['loss_delta_mean']:.6f} "
                    f"next={metrics['next_token_agreement']:.3f} "
                    f"greedy={metrics['greedy_token_agreement']:.3f} "
                    f"full={metrics['full_greedy_match_rate']:.3f}",
                    flush=True,
                )
                if passed and args.stop_on_first_pass:
                    print("[STOP] first passing candidate reached; stopping search for this module.", flush=True)
                    break
            finally:
                if candidate_model is not None:
                    cleanup_model(candidate_model)
                    candidate_model = None
                gc.collect()
            if remove_candidate_artifact:
                remove_dir(artifact_dir)
    finally:
        cleanup_model(reference_model)

    passed_rows = [row for row in log["results"] if row["passed"]]
    log["passed_count"] = len(passed_rows)
    log["best_pass"] = min(passed_rows, key=lambda row: row["size_bytes"]) if passed_rows else None
    if log["best_pass"] is not None and args.artifact_policy == "none":
        final_artifact_dir = out_root / "final_artifact"
        export_from_map(source_dir, Path(log["best_pass"]["map_path"]), final_artifact_dir, args.embed_bits)
        log["best_pass"]["artifact_dir"] = str(final_artifact_dir)
        log["best_pass"]["artifact_retained"] = True
        log["final_artifact_dir"] = str(final_artifact_dir)
    elif log["best_pass"] is not None and args.artifact_policy == "best":
        log["final_artifact_dir"] = log["best_pass"]["artifact_dir"]
    log["finished_utc"] = int(time.time())
    log["elapsed_sec"] = round(time.time() - started, 2)

    out_path = out_root / "combo_results.json"
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nSaved combo log to {out_path}", flush=True)
    print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
