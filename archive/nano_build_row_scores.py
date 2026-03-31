import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_inference_direct import load_nano_direct


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if math.isclose(mn, mx):
        return [0.5 for _ in values]
    den = mx - mn
    return [(v - mn) / den for v in values]


def parse_args():
    parser = argparse.ArgumentParser(description="Build row-level NANO scores for a specific linear module.")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-kind", choices=("nano", "hf"), default="nano")
    parser.add_argument("--module-name", action="append", default=[])
    parser.add_argument("--module-template")
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--prompts-file", default="prompts_eval_core.json")
    parser.add_argument("--out-json")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.40)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--fp32-rmsnorm", action="store_true")
    parser.add_argument("--num-threads", type=int)
    return parser.parse_args()


def load_reference_model(path: str, kind: str, fp32_rmsnorm: bool):
    if kind == "nano":
        return load_nano_direct(path, fp32_rmsnorm=fp32_rmsnorm)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map=("auto" if torch.cuda.is_available() else {"": "cpu"}),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def compact_module_name(name: str):
    result = name
    result = result.replace("model.layers.", "l")
    result = result.replace(".self_attn.", "_a_")
    result = result.replace(".mlp.", "_m_")
    result = result.replace(".embed_tokens", "_embed")
    result = result.replace(".", "_")
    return result


def resolve_module_names(args):
    module_names = list(args.module_name)
    if args.module_template:
        if args.layer_start is None or args.layer_end is None:
            raise ValueError("--layer-start and --layer-end are required with --module-template")
        module_names.extend(
            args.module_template.format(layer=layer)
            for layer in range(args.layer_start, args.layer_end + 1)
        )
    if not module_names:
        raise ValueError("Provide at least one --module-name or a --module-template range.")
    return module_names


def build_row_scores_for_modules(
    model,
    tokenizer,
    module_names: list[str],
    prompts: list[str],
    max_length: int,
    alpha: float,
    beta: float,
    gamma: float,
    skip_sensitivity: bool,
):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules = {}
    weights = {}
    row_norms = {}
    usage_sums = {}
    usage_counts = {}
    named_modules = dict(model.named_modules())
    for module_name in module_names:
        module = named_modules.get(module_name)
        if module is None:
            raise ValueError(f"Module not found: {module_name}")
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"Module is not nn.Linear: {module_name}")
        modules[module_name] = module
        weights[module_name] = module.weight.detach().float().cpu()
        row_norms[module_name] = torch.sqrt(torch.mean(weights[module_name].square(), dim=1))
        usage_sums[module_name] = torch.zeros(weights[module_name].shape[0], dtype=torch.float64)
        usage_counts[module_name] = 0

    hooks = []

    def make_forward_hook(module_name: str):
        def forward_hook(_module, _inputs, outputs):
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if not isinstance(tensor, torch.Tensor):
                return
            usage_sums[module_name].add_(
                tensor.detach().float().abs().mean(dim=(0, 1)).to(dtype=torch.float64, device="cpu")
            )
            usage_counts[module_name] += 1

        return forward_hook

    for module_name, module in modules.items():
        hooks.append(module.register_forward_hook(make_forward_hook(module_name)))

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    first_device = next(model.parameters()).device
    enc = {key: value.to(first_device) for key, value in enc.items()}

    with torch.inference_mode():
        model(**enc)
    for hook in hooks:
        hook.remove()

    usages = {
        module_name: usage_sums[module_name] / max(1, usage_counts[module_name])
        for module_name in module_names
    }

    if skip_sensitivity:
        row_saliencies = {
            module_name: torch.zeros_like(row_norms[module_name])
            for module_name in module_names
        }
    else:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        row_saliencies = {}
        for module_name, module in modules.items():
            grad = module.weight.grad
            if grad is None:
                row_saliencies[module_name] = torch.zeros_like(row_norms[module_name])
            else:
                row_saliencies[module_name] = (
                    grad.detach().float().cpu() * weights[module_name]
                ).abs().mean(dim=1)
        model.zero_grad(set_to_none=True)
        model.eval()

    rows_by_module = {}
    for module_name in module_names:
        row_norm = row_norms[module_name]
        usage = usages[module_name]
        row_saliency = row_saliencies[module_name]
        norm_n = normalize(row_norm.tolist())
        usage_n = normalize(usage.tolist())
        sal_n = normalize(row_saliency.tolist())

        rows = []
        for idx in range(weights[module_name].shape[0]):
            score = alpha * norm_n[idx] + beta * usage_n[idx] + gamma * sal_n[idx]
            rows.append(
                {
                    "row_index": idx,
                    "norm": float(row_norm[idx].item()),
                    "usage_out": float(usage[idx].item()),
                    "saliency": float(row_saliency[idx].item()),
                    "norm_n": float(norm_n[idx]),
                    "usage_n": float(usage_n[idx]),
                    "saliency_n": float(sal_n[idx]),
                    "score": float(score),
                }
            )
        rows.sort(key=lambda row: row["score"])
        rows_by_module[module_name] = rows
    return rows_by_module


def build_row_scores_for_module(
    model,
    tokenizer,
    module_name: str,
    prompts: list[str],
    max_length: int,
    alpha: float,
    beta: float,
    gamma: float,
    skip_sensitivity: bool,
):
    return build_row_scores_for_modules(
        model,
        tokenizer,
        [module_name],
        prompts,
        max_length,
        alpha,
        beta,
        gamma,
        skip_sensitivity,
    )[module_name]


def main():
    args = parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    prompts = json.loads(Path(args.prompts_file).read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts-file must contain a non-empty JSON list.")
    module_names = resolve_module_names(args)
    if len(module_names) == 1:
        if not args.out_json:
            raise ValueError("--out-json is required for a single module run.")
    elif not args.out_dir:
        raise ValueError("--out-dir is required when generating scores for multiple modules.")

    model, tokenizer = load_reference_model(args.reference_model, args.reference_kind, args.fp32_rmsnorm)
    rows_by_module = build_row_scores_for_modules(
        model,
        tokenizer,
        module_names,
        prompts,
        args.max_length,
        args.alpha,
        args.beta,
        args.gamma,
        args.skip_sensitivity,
    )

    created_utc = int(time.time())
    outputs = {}
    for module_name in module_names:
        outputs[module_name] = {
            "reference_model": args.reference_model,
            "reference_kind": args.reference_kind,
            "module_name": module_name,
            "created_utc": created_utc,
            "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
            "settings": {
                "max_length": args.max_length,
                "num_prompts": len(prompts),
                "skip_sensitivity": args.skip_sensitivity,
            },
            "rows": rows_by_module[module_name],
        }

    if len(module_names) == 1:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(outputs[module_names[0]], indent=2), encoding="utf-8")
        print(f"Saved row scores to {out_path}", flush=True)
        print(
            f"Lowest rows: {[row['row_index'] for row in outputs[module_names[0]]['rows'][:8]]}",
            flush=True,
        )
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for module_name in module_names:
            out_path = out_dir / f"row_scores_{compact_module_name(module_name)}.json"
            out_path.write_text(json.dumps(outputs[module_name], indent=2), encoding="utf-8")
        print(f"Saved {len(module_names)} row-score files to {out_dir}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
