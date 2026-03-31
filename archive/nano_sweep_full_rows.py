import argparse
import gc
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_export_from_map_v4 import export_from_map
from nano_search_combo_rows import (
    build_candidate_map,
    compact_module_name,
    load_fixed_specs_from_log,
    parse_fixed_spec,
    ranking_to_rows,
    remove_dir,
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
            "Sweep full-row row-mixed candidates across sequential layers, "
            "reusing a donor ranking and stopping at the first failure."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--module-template", default="model.layers.{layer}.mlp.up_proj")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--donor-ranking-file", required=True)
    parser.add_argument("--full-row-count", type=int, default=8192)
    parser.add_argument("--low-bits", type=int, choices=(6, 4, 2), default=4)
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
    parser.add_argument("--stop-on-first-fail", action="store_true", default=True)
    return parser.parse_args()


def spec_label(spec: dict) -> str:
    return f"{compact_module_name(spec['module_name'])}_{spec['row_count']:04d}r_b{spec['low_bits']}"


def materialize_fixed_specs(raw_specs: list[str]):
    specs = []
    for raw_spec in raw_specs:
        parsed = parse_fixed_spec(raw_spec)
        ranking_data = load_json(Path(parsed["row_ranking_file"]))
        parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
        specs.append(parsed)
    return specs


def materialize_specs_from_logs(log_paths: list[str]):
    specs = []
    for log_path in log_paths:
        for parsed in load_fixed_specs_from_log(Path(log_path)):
            ranking_data = load_json(Path(parsed["row_ranking_file"]))
            parsed["row_indices"] = ranking_to_rows(ranking_data, parsed["row_count"])
            specs.append(parsed)
    return specs


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

    donor_ranking = load_json(Path(args.donor_ranking_file))
    full_row_indices = ranking_to_rows(donor_ranking, args.full_row_count)
    fixed_specs = materialize_fixed_specs(args.fixed_spec)
    fixed_specs.extend(materialize_specs_from_logs(args.fixed_spec_log))

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
        "module_template": args.module_template,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "donor_ranking_file": args.donor_ranking_file,
        "full_row_count": args.full_row_count,
        "low_bits": args.low_bits,
        "initial_fixed_specs": [
            {
                "module_name": spec["module_name"],
                "row_ranking_file": spec["row_ranking_file"],
                "row_count": spec["row_count"],
                "low_bits": spec["low_bits"],
            }
            for spec in fixed_specs
        ],
        "results": [],
        "accepted_layers": [],
        "first_fail": None,
    }

    best_artifact_dir = None
    current_best_size = None
    stop_on_exact_mismatch = (
        args.min_next_token_agreement == 1.0
        and args.min_greedy_token_agreement == 1.0
        and args.min_full_greedy_match == 1.0
    )

    try:
        for layer in range(args.layer_start, args.layer_end + 1):
            module_name = args.module_template.format(layer=layer)
            candidate_map = build_candidate_map(
                base_map,
                fixed_specs,
                module_name,
                full_row_indices,
                args.low_bits,
            )
            label = (
                f"chain_{len(fixed_specs):02d}_"
                f"{compact_module_name(module_name)}_{args.full_row_count:04d}r_b{args.low_bits}"
            )
            map_dir = out_root / "maps"
            artifact_dir = out_root / "artifacts" / label
            map_dir.mkdir(parents=True, exist_ok=True)
            map_path = map_dir / f"{label}.json"

            export_candidate_map(candidate_map, map_path)
            export_from_map(source_dir, map_path, artifact_dir, embed_bits=args.embed_bits)

            size_bytes = (artifact_dir / "model.safetensors").stat().st_size
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
                stop_on_first_exact_mismatch=stop_on_exact_mismatch,
            )
            cleanup_model(candidate_model)
            candidate_model = None
            gc.collect()

            result = {
                "layer": layer,
                "module_name": module_name,
                "row_count": args.full_row_count,
                "size_bytes": size_bytes,
                "metrics": metrics,
                "passed": False,
                "artifact_dir": str(artifact_dir),
                "map_path": str(map_path),
            }

            if passes_thresholds(metrics, args):
                result["passed"] = True
                log["accepted_layers"].append(layer)
                fixed_specs.append(
                    {
                        "module_name": module_name,
                        "row_ranking_file": args.donor_ranking_file,
                        "row_count": args.full_row_count,
                        "low_bits": args.low_bits,
                        "row_indices": full_row_indices,
                    }
                )
                if args.artifact_policy == "best":
                    if current_best_size is None or size_bytes < current_best_size:
                        remove_dir(best_artifact_dir)
                        best_artifact_dir = artifact_dir
                        current_best_size = size_bytes
                    else:
                        remove_dir(artifact_dir)
                elif args.artifact_policy == "none":
                    remove_dir(artifact_dir)
                print(
                    f"[PASS] layer={layer} size={size_bytes} "
                    f"loss_delta={metrics['loss_delta_mean']:.6f} "
                    f"next={metrics['next_token_agreement']:.3f} "
                    f"greedy={metrics['greedy_token_agreement']:.3f} "
                    f"full={metrics['full_greedy_match_rate']:.3f}",
                    flush=True,
                )
            else:
                log["first_fail"] = result
                print(
                    f"[FAIL] layer={layer} size={size_bytes} "
                    f"loss_delta={metrics['loss_delta_mean']:.6f} "
                    f"next={metrics['next_token_agreement']:.3f} "
                    f"greedy={metrics['greedy_token_agreement']:.3f} "
                    f"full={metrics['full_greedy_match_rate']:.3f}",
                    flush=True,
                )
                if args.artifact_policy != "all":
                    remove_dir(artifact_dir)
                if args.stop_on_first_fail:
                    log["results"].append(result)
                    break

            log["results"].append(result)
            (out_root / "full_sweep_results.json").write_text(
                json.dumps(log, indent=2),
                encoding="utf-8",
            )

        if args.artifact_policy == "none" and log["accepted_layers"]:
            final_module = args.module_template.format(layer=log["accepted_layers"][-1])
            final_candidate = build_candidate_map(
                base_map,
                fixed_specs[:-1],
                final_module,
                full_row_indices,
                args.low_bits,
            )
            final_map_path = out_root / "final_artifact_map.json"
            final_dir = out_root / "final_artifact"
            export_candidate_map(final_candidate, final_map_path)
            export_from_map(source_dir, final_map_path, final_dir, embed_bits=args.embed_bits)

    finally:
        cleanup_model(reference_model)
        log["finished_utc"] = int(time.time())
        log["elapsed_sec"] = round(time.time() - started, 2)
        (out_root / "full_sweep_results.json").write_text(
            json.dumps(log, indent=2),
            encoding="utf-8",
        )
        print(f"Saved full sweep log to {out_root / 'full_sweep_results.json'}", flush=True)
        print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
