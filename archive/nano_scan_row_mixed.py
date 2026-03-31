import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_export_from_map_v4 import export_from_map, load_weight_map
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
        description="Scan row-mixed low-bit candidates inside a single module."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--module-name", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--row-counts", nargs="+", type=int, required=True)
    parser.add_argument("--selection-mode", choices=("prefix", "single"), default="prefix")
    parser.add_argument("--row-ranking-file")
    parser.add_argument("--lower-bits", type=int, choices=(6, 4, 2), default=4)
    parser.add_argument("--embed-bits", type=int, choices=(8, 6, 4, 2), default=8)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--residual-topk", type=int, default=0)
    parser.add_argument("--rank-kind", choices=("rms", "mean_abs"), default="rms")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--greedy-steps", type=int, default=4)
    parser.add_argument("--max-loss-delta", type=float, default=0.02)
    parser.add_argument("--min-next-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-greedy-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-full-greedy-match", type=float, default=1.0)
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    parser.add_argument("--quiet-loads", action="store_true")
    parser.add_argument("--num-threads", type=int)
    return parser.parse_args()


def load_tensor_from_source(source_dir: Path, tensor_name: str):
    weight_map = load_weight_map(source_dir)
    shard_name = weight_map[tensor_name]
    shard_path = source_dir / shard_name
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def rank_rows(tensor, rank_kind: str, descending: bool):
    weight_f = tensor.to(torch.float32)
    if rank_kind == "rms":
        scores = torch.sqrt(torch.mean(weight_f.square(), dim=1))
    else:
        scores = weight_f.abs().mean(dim=1)
    order = torch.argsort(scores, descending=descending)
    return order.tolist(), scores.tolist()


def build_candidate_map(
    base_map: dict,
    module_name: str,
    row_indices: list[int],
    lower_bits: int,
    embed_bits: int,
    group_size: int | None,
    residual_topk: int,
):
    result = deepcopy(base_map)
    mixed_modules = deepcopy(result.get("mixed_modules", {}))
    mixed_modules[module_name] = {
        "scheme": "row_mixed",
        "base_bits": 8,
        "low_bits": lower_bits,
        "row_indices": row_indices,
        "group_size": group_size,
        "residual_topk": residual_topk,
    }
    result["mixed_modules"] = mixed_modules
    result.setdefault("search_policy", {})
    result["search_policy"] = {
        "embed_bits": embed_bits,
        "mixed_modules": mixed_modules,
    }
    return result


def main():
    args = parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    started = time.time()
    source_dir = Path(args.source_dir)
    base_map = load_json(Path(args.base_map))
    prompts = load_json(Path(args.prompts_file))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tensor_name = f"{args.module_name}.weight"
    weight = load_tensor_from_source(source_dir, tensor_name)
    if args.row_ranking_file:
        ranking = load_json(Path(args.row_ranking_file))
        ranked_rows = [int(row["row_index"]) for row in ranking["rows"]]
        row_scores = [float(row["score"]) for row in ranking["rows"]]
    else:
        ranked_rows, row_scores = rank_rows(weight, args.rank_kind, args.descending)
    max_rows = weight.shape[0]

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
        "module_name": args.module_name,
        "tensor_name": tensor_name,
        "settings": {
            "row_counts": args.row_counts,
            "lower_bits": args.lower_bits,
            "embed_bits": args.embed_bits,
            "group_size": args.group_size,
            "residual_topk": args.residual_topk,
            "rank_kind": args.rank_kind,
            "descending": args.descending,
            "row_ranking_file": args.row_ranking_file,
            "prompts_file": args.prompts_file,
            "max_length": args.max_length,
            "greedy_steps": args.greedy_steps,
        },
        "results": [],
    }

    for row_count in args.row_counts:
        if args.selection_mode == "prefix":
            if row_count <= 0 or row_count > max_rows:
                raise ValueError(f"row_count out of range: {row_count} for tensor with {max_rows} rows")
            selected = sorted(ranked_rows[:row_count])
            label = f"{args.module_name.replace('.', '_')}_rows{row_count:04d}_b{args.lower_bits}"
            display = f"rows={row_count}"
        else:
            if row_count < 0 or row_count >= max_rows:
                raise ValueError(f"row rank out of range: {row_count} for tensor with {max_rows} rows")
            selected = [ranked_rows[row_count]]
            label = f"{args.module_name.replace('.', '_')}_rank{row_count:04d}_b{args.lower_bits}"
            display = f"rank={row_count} row={selected[0]}"

        map_path = out_root / "maps" / f"{label}.json"
        artifact_dir = out_root / "artifacts" / label
        candidate_map = build_candidate_map(
            base_map,
            args.module_name,
            selected,
            args.lower_bits,
            args.embed_bits,
            args.group_size,
            args.residual_topk,
        )
        export_candidate_map(candidate_map, map_path)

        print(f"\n[ROWS] module={args.module_name} {display}", flush=True)
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
                "row_indices": selected,
                "row_scores": [row_scores[idx] for idx in selected],
                "size_bytes": size_bytes,
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
        finally:
            if candidate_model is not None:
                cleanup_model(candidate_model)

    cleanup_model(reference_model)
    passed_rows = [row for row in log["results"] if row["passed"]]
    log["passed_count"] = len(passed_rows)
    log["finished_utc"] = int(time.time())
    log["elapsed_sec"] = round(time.time() - started, 2)
    log["best_pass"] = passed_rows[0] if passed_rows else None

    out_path = out_root / "scan_results.json"
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nSaved scan log to {out_path}", flush=True)
    print(f"Elapsed: {log['elapsed_sec']} sec", flush=True)


if __name__ == "__main__":
    main()
