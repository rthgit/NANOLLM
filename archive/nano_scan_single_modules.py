import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_export_from_map_v3 import export_from_map
from nano_search_subint8 import (
    canonical_suffix,
    cleanup_model,
    evaluate_pair,
    export_candidate_map,
    load_json,
    load_nano_maybe_quiet,
    load_reference_model,
    passes_thresholds,
    recalc_summary,
    state_to_jsonable,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan single-module sub-int8 candidates against a strict reference gate."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--greedy-steps", type=int, default=4)
    parser.add_argument("--suffixes", nargs="+", default=["up"])
    parser.add_argument("--lower-bits", type=int, choices=(6, 4, 2), default=4)
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2), default=8)
    parser.add_argument("--start-rank", type=int, default=0)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--ascending", action="store_true", default=True)
    parser.add_argument("--max-loss-delta", type=float, default=0.02)
    parser.add_argument("--min-next-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-greedy-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-full-greedy-match", type=float, default=1.0)
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    parser.add_argument("--quiet-loads", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    parser.add_argument("--num-threads", type=int)
    return parser.parse_args()


def build_single_module_map(base_map: dict, module_name: str, lower_bits: int, embed_bits: int):
    result = json.loads(json.dumps(base_map))
    tier_map = result["tier_map"]
    if module_name not in tier_map:
        raise KeyError(f"Module not found in map: {module_name}")
    tier_map[module_name] = lower_bits
    for block in result["blocks"]:
        if block["name"] == module_name:
            block["tier_bits"] = lower_bits
    result["summary"] = recalc_summary(result["blocks"], tier_map, embed_bits)
    result["search_policy"] = {
        "embed_bits": embed_bits,
        "lower_bits": lower_bits,
        "modules": [module_name],
    }
    return result


def main():
    args = parse_args()
    if args.num_threads:
        import torch

        torch.set_num_threads(args.num_threads)

    started = time.time()
    source_dir = Path(args.source_dir)
    base_map = load_json(Path(args.base_map))
    prompts = load_json(Path(args.prompts_file))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    suffixes = [canonical_suffix(name) for name in args.suffixes]
    candidates = [block for block in base_map["blocks"] if block["suffix"] in suffixes]
    candidates.sort(key=lambda row: row["score"], reverse=not args.ascending)

    if args.start_rank:
        candidates = candidates[args.start_rank :]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No candidate modules selected.")

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
        "settings": {
            "suffixes": suffixes,
            "lower_bits": args.lower_bits,
            "embed_bits": args.embed_bits,
            "start_rank": args.start_rank,
            "max_candidates": args.max_candidates,
            "ascending": args.ascending,
            "prompts_file": args.prompts_file,
            "max_length": args.max_length,
            "greedy_steps": args.greedy_steps,
        },
        "results": [],
    }

    for idx, block in enumerate(candidates, start=1):
        module_name = block["name"]
        label = f"{module_name.replace('.', '_')}_b{args.lower_bits}"
        map_path = out_root / "maps" / f"{label}.json"
        artifact_dir = out_root / "artifacts" / label
        candidate_map = build_single_module_map(base_map, module_name, args.lower_bits, args.embed_bits)
        export_candidate_map(candidate_map, map_path)

        print(f"\n[SCAN {idx}/{len(candidates)}] {module_name}", flush=True)
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
                "module_name": module_name,
                "score": block["score"],
                "layer_idx": block["layer_idx"],
                "suffix": block["suffix"],
                "state": state_to_jsonable({"embed_bits": args.embed_bits, "fractions": {}}),
                "size_bytes": size_bytes,
                "map_summary": candidate_map["summary"],
                "metrics": metrics,
                "passed": passed,
                "artifact_dir": str(artifact_dir),
                "map_path": str(map_path),
            }
            log["results"].append(row)
            print(
                f"[{'PASS' if passed else 'FAIL'}] size={size_bytes} "
                f"loss_delta={metrics['loss_delta_mean']:.6f} "
                f"next={metrics['next_token_agreement']:.3f} "
                f"greedy={metrics['greedy_token_agreement']:.3f} "
                f"full={metrics['full_greedy_match_rate']:.3f}",
                flush=True,
            )
            if passed and args.stop_on_pass:
                break
        finally:
            if candidate_model is not None:
                cleanup_model(candidate_model)

    cleanup_model(reference_model)
    passed_rows = [row for row in log["results"] if row["passed"]]
    log["passed_count"] = len(passed_rows)
    log["finished_utc"] = int(time.time())
    log["elapsed_sec"] = round(time.time() - started, 2)
    if passed_rows:
        passed_rows.sort(key=lambda row: (row["size_bytes"], row["metrics"]["loss_delta_mean"]))
        log["best_pass"] = passed_rows[0]
    else:
        log["best_pass"] = None

    out_path = out_root / "scan_results.json"
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nSaved scan log to {out_path}", flush=True)
    print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
