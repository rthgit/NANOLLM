import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_PROMPTS = [
    "Write a Python function to sort a list using bubble sort.",
    "If I have 3 apples and eat 2, how many apples do I have left?",
    "A red shirt is wet. I put it in the bright sun for 2 hours. What happens to the shirt?",
    "Translate to French: I love programming and building artificial intelligence.",
    "Explain what a neural network is in exactly 3 simple sentences.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke/fidelity tests for Nano artifact directories.")
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Artifact directory path (contains spec.json, quantized_modules.pt). "
        "If omitted, tries NANO_ARTIFACT_DIR and then auto-detect under /kaggle/working.",
    )
    parser.add_argument("--prompts-file", default="", help="Optional JSON file containing a list of prompts")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--baseline-mode",
        choices=["auto", "none", "8bit", "4bit"],
        default="auto",
        help="Comparison baseline. 'auto' uses artifact_spec.build_reference_mode when available.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", default="", help="Optional report path (default: <artifact-dir>/test_report.json)")
    # parse_known_args avoids failures from notebook-injected argv flags.
    args, _unknown = parser.parse_known_args()
    return args


def resolve_artifact_dir(arg_value: str) -> Path:
    if arg_value:
        p = Path(arg_value).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"Artifact directory does not exist: {p}")
        return p

    env_path = os.getenv("NANO_ARTIFACT_DIR", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"NANO_ARTIFACT_DIR does not exist: {p}")
        return p

    kaggle_root = Path("/kaggle/working")
    preferred = [
        kaggle_root / "final_artifact_3B",
        kaggle_root / "final_artifact_7B",
        kaggle_root / "final_artifact_Qwen2.5-14B-Instruct",
    ]
    existing = [p for p in preferred if p.exists() and p.is_dir()]
    if len(existing) == 1:
        return existing[0]

    # Generic fallback: any directory containing the required files.
    generic = []
    if kaggle_root.exists():
        for p in kaggle_root.iterdir():
            if not p.is_dir():
                continue
            if (p / "spec.json").exists() and (p / "quantized_modules.pt").exists():
                generic.append(p.resolve())
    if len(generic) == 1:
        return generic[0]

    raise SystemExit(
        "Missing artifact dir. Provide --artifact-dir or set NANO_ARTIFACT_DIR.\n"
        "Example: !python nano_artifact_test.py --artifact-dir /kaggle/working/final_artifact_3B"
    )


def clear_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_model_input_device(model) -> torch.device:
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict):
        for target in dm.values():
            if isinstance(target, torch.device):
                return target
            if isinstance(target, int):
                return torch.device(f"cuda:{target}")
            if isinstance(target, str) and target.startswith("cuda"):
                return torch.device(target)
        if "cpu" in dm.values():
            return torch.device("cpu")
    return next(model.parameters()).device


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        p = Path(args.prompts_file)
        return json.loads(p.read_text(encoding="utf-8"))
    return DEFAULT_PROMPTS


@torch.inference_mode()
def next_token_logits(model, tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    dev = get_model_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    return model(**enc).logits[0, -1, :].detach().cpu()


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, max_length: int) -> str:
    dev = get_model_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def load_loader_module(loader_path: Path):
    spec = importlib.util.spec_from_file_location("nano_loader", str(loader_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_baseline_quant_cfg(mode: str) -> BitsAndBytesConfig:
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def main() -> None:
    args = parse_args()
    artifact_dir = resolve_artifact_dir(args.artifact_dir)
    loader_path = artifact_dir / "load_artifact.py"
    spec_path = artifact_dir / "spec.json"
    state_path = artifact_dir / "quantized_modules.pt"
    output_path = Path(args.output).resolve() if args.output else (artifact_dir / "test_report.json")
    prompts = load_prompts(args)

    started = time.time()
    report: Dict[str, object] = {
        "artifact_dir": str(artifact_dir),
        "started_at_unix": int(started),
        "checks": {},
    }

    for p in [loader_path, spec_path, state_path]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    loader = load_loader_module(loader_path)
    model, tokenizer, artifact_spec = loader.load_artifact(str(artifact_dir))
    report["artifact_spec"] = artifact_spec
    report["checks"]["artifact_loaded"] = True

    baseline_mode = args.baseline_mode
    if baseline_mode == "auto":
        baseline_mode = str(artifact_spec.get("build_reference_mode", "8bit")).lower()
        if baseline_mode not in {"none", "8bit", "4bit"}:
            baseline_mode = "8bit"
    report["baseline_mode_used"] = baseline_mode

    generation_rows = []
    artifact_logits = {}
    for prompt in prompts:
        gen = greedy_generate(model, tokenizer, prompt, args.max_new_tokens, args.max_length)
        artifact_logits[prompt] = next_token_logits(model, tokenizer, prompt, args.max_length)
        generation_rows.append(
            {
                "prompt": prompt,
                "completion": gen,
                "completion_len": len(gen),
                "non_empty": bool(gen.strip()),
            }
        )

    report["generation"] = generation_rows
    report["checks"]["generation_non_empty"] = all(r["non_empty"] for r in generation_rows)

    if baseline_mode != "none":
        base_model_id = artifact_spec["base_model_id"]
        qcfg = get_baseline_quant_cfg(baseline_mode)
        del model
        clear_runtime_memory()

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=qcfg,
            device_map=args.device_map,
        ).eval()
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token

        cosine_rows = []
        for prompt in prompts:
            b = next_token_logits(base_model, base_tokenizer, prompt, args.max_length)
            a = artifact_logits[prompt]
            cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            cosine_rows.append({"prompt": prompt, "cosine": cos})

        avg_cos = sum(r["cosine"] for r in cosine_rows) / len(cosine_rows)
        min_cos = min(r["cosine"] for r in cosine_rows)
        report["cosine"] = cosine_rows
        report["cosine_summary"] = {"avg": avg_cos, "min": min_cos, "threshold": args.min_cosine}
        report["checks"]["cosine_threshold_pass"] = avg_cos >= args.min_cosine

        del base_model
        clear_runtime_memory()
    else:
        report["checks"]["cosine_threshold_pass"] = None

    report["elapsed_sec"] = int(time.time() - started)
    core_checks = [
        report["checks"]["artifact_loaded"],
        report["checks"]["generation_non_empty"],
    ]
    if report["checks"]["cosine_threshold_pass"] is not None:
        core_checks.append(bool(report["checks"]["cosine_threshold_pass"]))
    report["pass"] = all(core_checks)

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    if "cosine_summary" in report:
        print(json.dumps(report["cosine_summary"], indent=2))
    print(f"PASS={report['pass']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
