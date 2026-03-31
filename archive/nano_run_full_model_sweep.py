import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SUFFIXES = [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-model NANO sweep in chained stages (one suffix at a time)."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--suffixes", nargs="+", default=DEFAULT_SUFFIXES)
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--row-counts", nargs="+", type=int, default=[8192, 4096, 2048, 1024, 512])
    parser.add_argument("--bit-ladder", nargs="+", type=int, default=[0, 2, 4, 6])
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
    parser.add_argument("--candidate-eval-mode", choices=("ram", "disk"), default="ram")
    parser.add_argument("--candidate-map-policy", choices=("keep", "none"), default="none")
    parser.add_argument("--keep-layer-artifacts", action="store_true")
    return parser.parse_args()


def short_suffix(suffix: str):
    return suffix.replace("self_attn.", "attn_").replace("mlp.", "").replace(".", "_")


def build_step_cmd(args, suffix: str, step_out: Path, prev_log: Path | None, is_last: bool):
    cmd = [
        sys.executable,
        "nano_ping_multibit_sweep.py",
        "--source-dir",
        args.source_dir,
        "--base-map",
        args.base_map,
        "--reference-model",
        args.reference_model,
        "--reference-kind",
        args.reference_kind,
        "--out-root",
        str(step_out),
        "--prompts-file",
        args.prompts_file,
        "--module-suffix",
        suffix,
        "--row-counts",
        *[str(x) for x in args.row_counts],
        "--bit-ladder",
        *[str(x) for x in args.bit_ladder],
        "--embed-bits",
        str(args.embed_bits),
        "--ranking-dir",
        args.ranking_dir,
        "--ranking-kind",
        args.ranking_kind,
        "--ranking-alpha",
        str(args.ranking_alpha),
        "--ranking-beta",
        str(args.ranking_beta),
        "--ranking-gamma",
        str(args.ranking_gamma),
        "--max-length",
        str(args.max_length),
        "--greedy-steps",
        str(args.greedy_steps),
        "--max-loss-delta",
        str(args.max_loss_delta),
        "--min-next-token-agreement",
        str(args.min_next_token_agreement),
        "--min-greedy-token-agreement",
        str(args.min_greedy_token_agreement),
        "--min-full-greedy-match",
        str(args.min_full_greedy_match),
        "--candidate-eval-mode",
        args.candidate_eval_mode,
        "--candidate-map-policy",
        args.candidate_map_policy,
    ]
    if args.layer_start is not None:
        cmd += ["--layer-start", str(args.layer_start)]
    if args.layer_end is not None:
        cmd += ["--layer-end", str(args.layer_end)]
    if args.ranking_model:
        cmd += ["--ranking-model", args.ranking_model]
    if args.ranking_skip_sensitivity:
        cmd += ["--ranking-skip-sensitivity"]
    if args.fp32_rmsnorm:
        cmd += ["--fp32-rmsnorm"]
    if args.quiet_loads:
        cmd += ["--quiet-loads"]
    if args.num_threads:
        cmd += ["--num-threads", str(args.num_threads)]
    if args.keep_layer_artifacts:
        cmd += ["--keep-layer-artifacts"]
    if not is_last:
        cmd += ["--skip-final-export"]
    if prev_log is not None:
        cmd += ["--fixed-spec-log", str(prev_log)]
    return cmd


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "started_utc": int(time.time()),
        "source_dir": args.source_dir,
        "base_map": args.base_map,
        "reference_model": args.reference_model,
        "reference_kind": args.reference_kind,
        "candidate_eval_mode": args.candidate_eval_mode,
        "suffixes": args.suffixes,
        "steps": [],
    }

    prev_log = None
    started = time.time()
    for idx, suffix in enumerate(args.suffixes, start=1):
        step_name = f"step_{idx:02d}_{short_suffix(suffix)}"
        step_out = out_root / step_name
        step_out.mkdir(parents=True, exist_ok=True)

        is_last = idx == len(args.suffixes)
        cmd = build_step_cmd(args, suffix, step_out, prev_log, is_last=is_last)
        print(f"\n[STEP {idx}/{len(args.suffixes)}] {suffix}", flush=True)
        print(" ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
        elapsed = round(time.time() - t0, 2)
        if proc.returncode != 0:
            raise RuntimeError(f"Step failed ({suffix}) with exit code {proc.returncode}")

        log_path = step_out / "ping_results.json"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing step log: {log_path}")
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
        summary["steps"].append(
            {
                "suffix": suffix,
                "out_root": str(step_out),
                "log_path": str(log_path),
                "elapsed_sec": elapsed,
                "accepted_specs": len(log_data.get("accepted_specs", [])),
                "accepted_zero_modules": len(log_data.get("accepted_zero_modules", [])),
                "final_size_bytes": log_data.get("final_size_bytes"),
            }
        )
        prev_log = log_path

        summary_path = out_root / "full_sweep_summary.json"
        summary["updated_utc"] = int(time.time())
        summary["elapsed_sec"] = round(time.time() - started, 2)
        summary["last_log"] = str(prev_log)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["finished_utc"] = int(time.time())
    summary["elapsed_sec"] = round(time.time() - started, 2)
    summary_path = out_root / "full_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
