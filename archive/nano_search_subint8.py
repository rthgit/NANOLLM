import argparse
import contextlib
import gc
import io
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_export_from_map_v4 import export_from_map
from nano_inference_direct import load_nano_direct


SUFFIX_ALIASES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
SHORT_NAMES = {value: key for key, value in SUFFIX_ALIASES.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy sub-int8 search on top of a canonical NANO int8 control."
    )
    parser.add_argument("--source-dir", required=True, help="HF source directory used for export")
    parser.add_argument("--base-map", required=True, help="Canonical v3 map JSON, usually all-8")
    parser.add_argument("--reference-model", required=True, help="Reference model or artifact path")
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--out-root", required=True, help="Directory to store candidate maps/artifacts/logs")
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--greedy-steps", type=int, default=4)
    parser.add_argument("--lower-suffixes", nargs="+", default=["gate", "up"])
    parser.add_argument("--lower-bits", type=int, choices=(6, 4, 2), default=4)
    parser.add_argument("--fraction-step", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--embed-lower-bits", type=int, choices=(6, 4, 2))
    parser.add_argument("--max-loss-delta", type=float, default=0.02)
    parser.add_argument("--min-next-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-greedy-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-full-greedy-match", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    parser.add_argument("--quiet-loads", action="store_true")
    return parser.parse_args()


def canonical_suffix(name: str) -> str:
    if name in SUFFIX_ALIASES:
        return SUFFIX_ALIASES[name]
    if name in SHORT_NAMES:
        return name
    raise ValueError(f"Unsupported suffix alias: {name}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp_fraction(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def recalc_summary(blocks: list[dict], tier_map: dict[str, int], embed_bits: int | None):
    tier_counts = {"8": 0, "4": 0, "2": 0, "0": 0}
    tier_params = {"8": 0, "4": 0, "2": 0, "0": 0}
    total_params = 0
    bits_weighted = 0
    for block in blocks:
        bits = int(tier_map[block["name"]])
        params = int(block["params"])
        total_params += params
        bits_weighted += params * bits
        tier_counts[str(bits)] += 1
        tier_params[str(bits)] += params
    effective_bits = (bits_weighted / total_params) if total_params else 0.0
    return {
        "num_blocks": len(blocks),
        "total_target_params": total_params,
        "effective_bits": effective_bits,
        "tier_counts": tier_counts,
        "tier_params": tier_params,
        "embed_bits": embed_bits,
    }


def derive_candidate_map(base_map: dict, state: dict, lower_bits: int):
    blocks = [dict(block) for block in base_map["blocks"]]
    tier_map = dict(base_map["tier_map"])

    by_suffix = {}
    for block in blocks:
        by_suffix.setdefault(block["suffix"], []).append(block)

    for suffix, fraction in state["fractions"].items():
        if fraction <= 0:
            continue
        candidates = sorted(by_suffix.get(suffix, []), key=lambda row: row["score"])
        if not candidates:
            continue
        count = int(math.ceil(len(candidates) * fraction))
        for block in candidates[:count]:
            tier_map[block["name"]] = min(int(tier_map[block["name"]]), lower_bits)

    for block in blocks:
        block["tier_bits"] = int(tier_map[block["name"]])

    result = deepcopy(base_map)
    result["tier_map"] = tier_map
    result["blocks"] = sorted(blocks, key=lambda row: row["score"], reverse=True)
    result["summary"] = recalc_summary(result["blocks"], tier_map, state["embed_bits"])
    result.setdefault("search_policy", {})
    result["search_policy"] = {
        "embed_bits": state["embed_bits"],
        "lower_bits": lower_bits,
        "fractions": dict(state["fractions"]),
    }
    return result


def signature_for_map(candidate_map: dict, embed_bits: int | None):
    changed = sorted(
        (name, int(bits))
        for name, bits in candidate_map["tier_map"].items()
        if int(bits) != 8
    )
    return {"embed_bits": embed_bits, "changed": changed}


def label_for_state(state: dict, ordered_suffixes: list[str], lower_bits: int):
    parts = [f"embed{state['embed_bits']}"]
    for suffix in ordered_suffixes:
        fraction = state["fractions"].get(suffix, 0.0)
        if fraction <= 0:
            continue
        short = SHORT_NAMES.get(suffix, suffix.replace(".", "_"))
        pct = int(round(fraction * 100))
        parts.append(f"{short}{pct:02d}b{lower_bits}")
    return "-".join(parts)


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_reference_model(path: str, kind: str, fp32_rmsnorm: bool):
    if kind == "nano":
        model, tokenizer = load_nano_direct(path, fp32_rmsnorm=fp32_rmsnorm)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=("auto" if torch.cuda.is_available() else {"": "cpu"}),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_nano_maybe_quiet(path: str, quiet: bool, fp32_rmsnorm: bool):
    if quiet:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            return load_nano_direct(path, fp32_rmsnorm=fp32_rmsnorm)
    return load_nano_direct(path, fp32_rmsnorm=fp32_rmsnorm)


def greedy_token_ids(model, input_ids, attention_mask, steps: int):
    cur_ids = input_ids.clone()
    cur_mask = attention_mask.clone()
    generated = []
    eos_token_id = getattr(model.config, "eos_token_id", None)
    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(input_ids=cur_ids, attention_mask=cur_mask)
            next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], dtype=cur_ids.dtype, device=cur_ids.device)
            cur_ids = torch.cat([cur_ids, next_tensor], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)],
                dim=1,
            )
            if eos_token_id is not None:
                if isinstance(eos_token_id, list) and next_id in eos_token_id:
                    break
                if isinstance(eos_token_id, int) and next_id == eos_token_id:
                    break
    return generated


def evaluate_pair(
    reference_model,
    candidate_model,
    tokenizer,
    prompts: list[str],
    max_length: int,
    greedy_steps: int,
    stop_on_first_exact_mismatch: bool = False,
):
    rows = []
    next_matches = 0
    greedy_matches = 0
    greedy_total = 0
    full_matches = 0
    ref_losses = []
    cand_losses = []
    stopped_early = False

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        first_device = next(candidate_model.parameters()).device
        enc = {key: value.to(first_device) for key, value in enc.items()}

        with torch.inference_mode():
            ref_out = reference_model(**enc, labels=enc["input_ids"])
            cand_out = candidate_model(**enc, labels=enc["input_ids"])

        ref_loss = float(ref_out.loss.item())
        cand_loss = float(cand_out.loss.item())
        ref_losses.append(ref_loss)
        cand_losses.append(cand_loss)

        ref_next = int(ref_out.logits[:, -1, :].argmax(dim=-1).item())
        cand_next = int(cand_out.logits[:, -1, :].argmax(dim=-1).item())
        next_match = ref_next == cand_next
        next_matches += int(next_match)

        ref_greedy = greedy_token_ids(reference_model, enc["input_ids"], enc["attention_mask"], greedy_steps)
        cand_greedy = greedy_token_ids(candidate_model, enc["input_ids"], enc["attention_mask"], greedy_steps)
        max_len = max(len(ref_greedy), len(cand_greedy))
        prompt_matches = 0
        for idx in range(max_len):
            ref_token = ref_greedy[idx] if idx < len(ref_greedy) else None
            cand_token = cand_greedy[idx] if idx < len(cand_greedy) else None
            prompt_matches += int(ref_token == cand_token)
        greedy_matches += prompt_matches
        greedy_total += max_len
        full_match = ref_greedy == cand_greedy
        full_matches += int(full_match)

        rows.append(
            {
                "prompt": prompt,
                "reference_loss": ref_loss,
                "candidate_loss": cand_loss,
                "loss_delta": cand_loss - ref_loss,
                "reference_next_token": ref_next,
                "candidate_next_token": cand_next,
                "next_token_match": next_match,
                "reference_greedy": ref_greedy,
                "candidate_greedy": cand_greedy,
                "greedy_token_match_rate": (prompt_matches / max_len) if max_len else 1.0,
                "full_greedy_match": full_match,
            }
        )

        if stop_on_first_exact_mismatch and (not next_match or not full_match or prompt_matches != max_len):
            stopped_early = True
            break

    summary = {
        "reference_loss_mean": sum(ref_losses) / len(ref_losses),
        "candidate_loss_mean": sum(cand_losses) / len(cand_losses),
        "loss_delta_mean": (sum(cand_losses) - sum(ref_losses)) / len(ref_losses),
        "next_token_agreement": next_matches / len(rows),
        "greedy_token_agreement": (greedy_matches / greedy_total) if greedy_total else 1.0,
        "full_greedy_match_rate": full_matches / len(rows),
        "evaluated_prompts": len(rows),
        "total_prompts": len(prompts),
        "stopped_early": stopped_early,
        "rows": rows,
    }
    return summary


def passes_thresholds(metrics: dict, args) -> bool:
    return (
        metrics["loss_delta_mean"] <= args.max_loss_delta
        and metrics["next_token_agreement"] >= args.min_next_token_agreement
        and metrics["greedy_token_agreement"] >= args.min_greedy_token_agreement
        and metrics["full_greedy_match_rate"] >= args.min_full_greedy_match
    )


def state_to_jsonable(state: dict):
    return {
        "embed_bits": state["embed_bits"],
        "fractions": dict(state["fractions"]),
    }


def make_moves(state: dict, suffixes: list[str], args):
    moves = []
    if args.embed_lower_bits is not None and state["embed_bits"] > args.embed_lower_bits:
        embed_state = deepcopy(state)
        embed_state["embed_bits"] = args.embed_lower_bits
        moves.append(("embed", embed_state))

    for suffix in suffixes:
        current = state["fractions"].get(suffix, 0.0)
        if current >= args.max_fraction:
            continue
        next_value = clamp_fraction(min(args.max_fraction, current + args.fraction_step))
        moved_state = deepcopy(state)
        moved_state["fractions"][suffix] = next_value
        moves.append((suffix, moved_state))
    return moves


def export_candidate_map(candidate_map: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate_map, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    started = time.time()
    source_dir = Path(args.source_dir)
    base_map_path = Path(args.base_map)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = load_json(Path(args.prompts_file))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts-file must contain a non-empty JSON list.")

    suffixes = [canonical_suffix(name) for name in args.lower_suffixes]
    base_map = load_json(base_map_path)
    missing_suffixes = [suffix for suffix in suffixes if not any(block["suffix"] == suffix for block in base_map["blocks"])]
    if missing_suffixes:
        raise ValueError(f"Requested suffixes not present in base map: {missing_suffixes}")

    print(f"Loading reference model ({args.reference_kind}): {args.reference_model}", flush=True)
    if args.reference_kind == "nano":
        reference_model, tokenizer = load_nano_maybe_quiet(
            args.reference_model,
            args.quiet_loads,
            args.fp32_rmsnorm,
        )
    else:
        reference_model, tokenizer = load_reference_model(args.reference_model, args.reference_kind, args.fp32_rmsnorm)

    baseline_path = Path(args.reference_model) / "model.safetensors"
    baseline_size = baseline_path.stat().st_size if baseline_path.exists() else None

    state = {
        "embed_bits": 8,
        "fractions": {suffix: 0.0 for suffix in suffixes},
    }

    search_log = {
        "started_utc": int(time.time()),
        "source_dir": str(source_dir),
        "base_map": str(base_map_path),
        "reference_model": args.reference_model,
        "reference_kind": args.reference_kind,
        "baseline_size_bytes": baseline_size,
        "settings": {
            "prompts_file": args.prompts_file,
            "max_length": args.max_length,
            "greedy_steps": args.greedy_steps,
            "lower_suffixes": suffixes,
            "lower_bits": args.lower_bits,
            "fraction_step": args.fraction_step,
            "max_fraction": args.max_fraction,
            "embed_lower_bits": args.embed_lower_bits,
            "thresholds": {
                "max_loss_delta": args.max_loss_delta,
                "min_next_token_agreement": args.min_next_token_agreement,
                "min_greedy_token_agreement": args.min_greedy_token_agreement,
                "min_full_greedy_match": args.min_full_greedy_match,
            },
        },
        "accepted": [],
        "rejected": [],
    }

    best_size = baseline_size if baseline_size is not None else float("inf")
    seen_signatures = set()

    while True:
        moves = make_moves(state, suffixes, args)
        if not moves:
            break

        round_results = []
        for move_name, candidate_state in moves:
            candidate_map = derive_candidate_map(base_map, candidate_state, args.lower_bits)
            signature = json.dumps(signature_for_map(candidate_map, candidate_state["embed_bits"]), sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            label = label_for_state(candidate_state, suffixes, args.lower_bits)
            map_path = out_root / "maps" / f"{label}.json"
            artifact_dir = out_root / "artifacts" / label
            export_candidate_map(candidate_map, map_path)

            print(f"\n[TRY] {label} via move={move_name}", flush=True)
            export_from_map(source_dir, map_path, artifact_dir, candidate_state["embed_bits"])

            candidate_model = None
            try:
                candidate_model, _candidate_tokenizer = load_nano_maybe_quiet(
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
                record = {
                    "label": label,
                    "move": move_name,
                    "state": state_to_jsonable(candidate_state),
                    "size_bytes": size_bytes,
                    "map_summary": candidate_map["summary"],
                    "metrics": metrics,
                    "passed": passed,
                    "artifact_dir": str(artifact_dir),
                    "map_path": str(map_path),
                }
                round_results.append(record)
                bucket = "accepted" if passed else "rejected"
                search_log[bucket].append(record)
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

        passing = [row for row in round_results if row["passed"] and row["size_bytes"] < best_size]
        if not passing:
            break

        passing.sort(
            key=lambda row: (
                row["size_bytes"],
                row["metrics"]["loss_delta_mean"],
                -row["metrics"]["greedy_token_agreement"],
            )
        )
        winner = passing[0]
        state = {
            "embed_bits": int(winner["state"]["embed_bits"]),
            "fractions": {suffix: float(value) for suffix, value in winner["state"]["fractions"].items()},
        }
        best_size = winner["size_bytes"]
        print(f"[KEEP] {winner['label']} -> {best_size} bytes", flush=True)

    search_log["finished_utc"] = int(time.time())
    search_log["elapsed_sec"] = round(time.time() - started, 2)
    search_log["final_state"] = state_to_jsonable(state)
    search_log["best_size_bytes"] = None if math.isinf(best_size) else best_size

    out_path = out_root / "search_results.json"
    out_path.write_text(json.dumps(search_log, indent=2), encoding="utf-8")
    print(f"\nSaved search log to {out_path}", flush=True)
    print(f"Elapsed: {search_log['elapsed_sec']} sec", flush=True)

    cleanup_model(reference_model)


if __name__ == "__main__":
    main()
