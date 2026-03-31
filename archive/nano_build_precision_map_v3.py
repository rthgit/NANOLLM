import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
BITS_ORDER = (8, 4, 2, 0)


def pick_dtype(name: str):
    name = (name or "fp16").lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return torch.float16


def normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mn = min(values.values())
    mx = max(values.values())
    if math.isclose(mn, mx):
        return {k: 0.5 for k in values}
    den = mx - mn
    return {k: float((v - mn) / den) for k, v in values.items()}


def renormalize_weights(alpha: float, beta: float, gamma: float, keep_sensitivity: bool):
    active = [("alpha", alpha), ("beta", beta)]
    if keep_sensitivity:
        active.append(("gamma", gamma))
    total = sum(weight for _, weight in active)
    if total <= 0:
        raise ValueError("At least one active scoring weight must be > 0.")
    scale = 1.0 / total
    out = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
    for name, weight in active:
        out[name] = weight * scale
    return out


def parse_layer_index(name: str):
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def get_suffix(name: str):
    for suffix in TARGET_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def get_family(name: str):
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def get_group_key(module_info: dict, group_by: str):
    if group_by == "all":
        return "all"
    if group_by == "family":
        return module_info["family"]
    if group_by == "suffix":
        return module_info["suffix"]
    if group_by == "layer":
        layer_idx = module_info["layer_idx"]
        return f"layer.{layer_idx}" if layer_idx is not None else "layer.none"
    if group_by == "layer_suffix":
        layer_idx = module_info["layer_idx"]
        layer_key = f"layer.{layer_idx}" if layer_idx is not None else "layer.none"
        return f"{layer_key}:{module_info['suffix']}"
    raise ValueError(f"Unsupported group-by: {group_by}")


def get_min_bits(module_info: dict, args):
    suffix = module_info["suffix"]
    suffix_overrides = {
        "self_attn.q_proj": args.q_min_bits,
        "self_attn.k_proj": args.k_min_bits,
        "self_attn.v_proj": args.v_min_bits,
        "self_attn.o_proj": args.o_min_bits,
        "mlp.gate_proj": args.gate_min_bits,
        "mlp.up_proj": args.up_min_bits,
        "mlp.down_proj": args.down_min_bits,
    }
    if suffix_overrides[suffix] is not None:
        return suffix_overrides[suffix]
    if module_info["family"] == "attention" and args.attention_min_bits is not None:
        return args.attention_min_bits
    if module_info["family"] == "mlp" and args.mlp_min_bits is not None:
        return args.mlp_min_bits
    return args.min_bits


def assign_tiers_by_mass(modules_info: list[dict], ratios: list[float], min_bits: int):
    if len(ratios) != 4:
        raise ValueError("Need 4 ratios for [8,4,2,0]-bit tiers.")
    if not modules_info:
        return {}

    allowed = [bits for bits in BITS_ORDER if bits >= min_bits]
    if not allowed:
        raise ValueError(f"No allowed tiers remain for min_bits={min_bits}.")

    ratio_by_bits = dict(zip(BITS_ORDER, ratios))
    active_ratios = [ratio_by_bits[bits] for bits in allowed]
    total_ratio = sum(active_ratios)
    if total_ratio <= 0:
        raise ValueError(f"Active tier ratios sum to zero for min_bits={min_bits}.")
    normalized_ratios = [ratio / total_ratio for ratio in active_ratios]

    total_params = sum(module["params"] for module in modules_info)
    target_mass = [ratio * total_params for ratio in normalized_ratios]
    sorted_modules = sorted(modules_info, key=lambda item: item["score"], reverse=True)

    out = {}
    tier_idx = 0
    tier_mass = 0
    for module in sorted_modules:
        bits = allowed[tier_idx]
        out[module["name"]] = bits
        tier_mass += module["params"]
        if tier_idx < len(allowed) - 1 and tier_mass >= target_mass[tier_idx]:
            tier_idx += 1
            tier_mass = 0
    return out


def collect_target_modules(model, suffixes: tuple[str, ...]):
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(suffix) for suffix in suffixes):
            modules[name] = module
    return modules


def run_usage_pass(model, tokenizer, prompts: list[str], modules: dict[str, torch.nn.Module], max_length: int):
    sums = {name: 0.0 for name in modules}
    counts = {name: 0 for name in modules}
    hooks = []

    def make_hook(name):
        def _hook(_module, _inputs, outputs):
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if isinstance(tensor, torch.Tensor):
                sums[name] += float(tensor.detach().float().abs().mean().item())
                counts[name] += 1
        return _hook

    for name, module in modules.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            first_device = next(model.parameters()).device
            enc = {key: value.to(first_device) for key, value in enc.items()}
            model(**enc)

    for hook in hooks:
        hook.remove()
    return {name: (sums[name] / counts[name]) if counts[name] > 0 else 0.0 for name in modules}


def run_grad_sensitivity(model, tokenizer, prompts: list[str], modules: dict[str, torch.nn.Module], max_length: int):
    if not prompts:
        return {name: 0.0 for name in modules}

    model.train()
    model.zero_grad(set_to_none=True)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    first_device = next(model.parameters()).device
    enc = {key: value.to(first_device) for key, value in enc.items()}
    out = model(**enc, labels=enc["input_ids"])
    out.loss.backward()

    sens = {}
    for name, module in modules.items():
        weight = module.weight
        grad = weight.grad
        if grad is None:
            sens[name] = 0.0
            continue
        if hasattr(weight, "dequantize"):
            weight_f = weight.dequantize().detach().float()
        else:
            weight_f = weight.detach().float()
        sens[name] = float((grad.detach().float() * weight_f).abs().mean().item())

    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a canonical NANO precision map using normalized norm, output usage, and saliency."
    )
    parser.add_argument("--model-ref", required=True, help="HF model id or local model path")
    parser.add_argument("--prompts-file", required=True, help="JSON file containing a list of prompts")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--usage-samples", type=int, default=20)
    parser.add_argument("--sensitivity-samples", type=int, default=4)
    parser.add_argument("--skip-sensitivity", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.40)
    parser.add_argument("--tier-ratios", default="0.20,0.40,0.20,0.20", help="Ratios for [8,4,2,0]")
    parser.add_argument("--group-by", choices=["all", "family", "suffix", "layer", "layer_suffix"], default="suffix")
    parser.add_argument("--norm-kind", choices=["rms", "mean_abs"], default="rms")
    parser.add_argument("--target-suffixes", nargs="*", default=list(TARGET_SUFFIXES))
    parser.add_argument("--min-bits", type=int, choices=(0, 2, 4, 8), default=0)
    parser.add_argument("--attention-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--mlp-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--q-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--k-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--v-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--o-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--gate-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--up-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--down-min-bits", type=int, choices=(0, 2, 4, 8))
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

    prompts = json.loads(Path(args.prompts_file).read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts-file must contain a non-empty JSON list.")

    keep_sensitivity = not (args.skip_sensitivity or args.load_in_8bit or args.load_in_4bit)
    weights = renormalize_weights(args.alpha, args.beta, args.gamma, keep_sensitivity)
    if not keep_sensitivity:
        print(
            f"Renormalized weights with sensitivity disabled: "
            f"alpha={weights['alpha']:.3f} beta={weights['beta']:.3f} gamma={weights['gamma']:.3f}",
            flush=True,
        )

    dtype = pick_dtype(args.dtype)
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = dtype
        if args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif args.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
    else:
        load_kwargs["device_map"] = {"": "cpu"}
        load_kwargs["torch_dtype"] = dtype

    print(f"Loading model: {args.model_ref}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_ref, **load_kwargs)
    model.eval()

    target_suffixes = tuple(args.target_suffixes)
    modules = collect_target_modules(model, target_suffixes)
    if not modules:
        raise RuntimeError("No target modules found. Check model architecture or target suffixes.")
    print(f"Target modules: {len(modules)}", flush=True)

    norms = {}
    params = {}
    module_meta = {}
    for name, module in modules.items():
        weight = module.weight.detach()
        if hasattr(weight, "dequantize"):
            weight_f = weight.dequantize().float()
        else:
            weight_f = weight.float()
        if args.norm_kind == "rms":
            norms[name] = float(torch.sqrt(torch.mean(weight_f.square())).item())
        else:
            norms[name] = float(weight_f.abs().mean().item())
        params[name] = int(weight_f.numel())
        suffix = get_suffix(name)
        info = {
            "name": name,
            "suffix": suffix,
            "family": get_family(name),
            "layer_idx": parse_layer_index(name),
            "params": params[name],
        }
        info["group_key"] = get_group_key(info, args.group_by)
        info["min_bits"] = get_min_bits(info, args)
        module_meta[name] = info

    usage_prompts = prompts[: max(1, min(args.usage_samples, len(prompts)))]
    usage = run_usage_pass(model, tokenizer, usage_prompts, modules, args.max_length)

    if keep_sensitivity:
        sens_prompts = prompts[: max(1, min(args.sensitivity_samples, len(prompts)))]
        sensitivity = run_grad_sensitivity(model, tokenizer, sens_prompts, modules, args.max_length)
    else:
        sens_prompts = []
        sensitivity = {name: 0.0 for name in modules}

    norm_n = normalize_scores(norms)
    usage_n = normalize_scores(usage)
    sens_n = normalize_scores(sensitivity)

    modules_info = []
    for name in modules:
        info = dict(module_meta[name])
        info.update(
            {
                "norm": norms[name],
                "usage_out": usage[name],
                "saliency": sensitivity[name],
                "norm_n": norm_n[name],
                "usage_n": usage_n[name],
                "saliency_n": sens_n[name],
            }
        )
        info["score"] = float(
            weights["alpha"] * info["norm_n"]
            + weights["beta"] * info["usage_n"]
            + weights["gamma"] * info["saliency_n"]
        )
        modules_info.append(info)

    ratios = [float(value.strip()) for value in args.tier_ratios.split(",")]
    tier_map = {}
    grouped = {}
    for module_info in modules_info:
        group_key = module_info["group_key"]
        floor_key = f"min_bits={module_info['min_bits']}"
        grouped.setdefault((group_key, floor_key), []).append(module_info)

    for (_group_key, _floor_key), bucket in grouped.items():
        min_bits = bucket[0]["min_bits"]
        tier_map.update(assign_tiers_by_mass(bucket, ratios, min_bits))

    blocks = []
    tier_counts = {"8": 0, "4": 0, "2": 0, "0": 0}
    tier_params = {"8": 0, "4": 0, "2": 0, "0": 0}
    for info in sorted(modules_info, key=lambda item: item["score"], reverse=True):
        assigned_bits = tier_map[info["name"]]
        row = dict(info)
        row["tier_bits"] = assigned_bits
        blocks.append(row)
        key = str(assigned_bits)
        tier_counts[key] += 1
        tier_params[key] += info["params"]

    total_params = sum(params.values())
    bits_weighted = sum(params[block["name"]] * block["tier_bits"] for block in blocks)
    effective_bits = (bits_weighted / total_params) if total_params > 0 else 0.0

    output = {
        "model_ref": args.model_ref,
        "created_utc": int(time.time()),
        "weights": {
            "requested": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
            "effective": weights,
        },
        "settings": {
            "dtype": args.dtype,
            "max_length": args.max_length,
            "usage_samples": len(usage_prompts),
            "sensitivity_samples": len(sens_prompts),
            "skip_sensitivity": not keep_sensitivity,
            "tier_ratios": ratios,
            "group_by": args.group_by,
            "norm_kind": args.norm_kind,
            "target_suffixes": list(target_suffixes),
            "min_bits": {
                "default": args.min_bits,
                "attention": args.attention_min_bits,
                "mlp": args.mlp_min_bits,
                "q_proj": args.q_min_bits,
                "k_proj": args.k_min_bits,
                "v_proj": args.v_min_bits,
                "o_proj": args.o_min_bits,
                "gate_proj": args.gate_min_bits,
                "up_proj": args.up_min_bits,
                "down_proj": args.down_min_bits,
            },
        },
        "summary": {
            "num_blocks": len(blocks),
            "total_target_params": total_params,
            "effective_bits": effective_bits,
            "tier_counts": tier_counts,
            "tier_params": tier_params,
            "elapsed_sec": round(time.time() - started, 2),
        },
        "tier_map": {block["name"]: block["tier_bits"] for block in blocks},
        "blocks": blocks,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved v3 precision map to {out_path}", flush=True)
    print(f"effective_bits: {effective_bits:.4f}", flush=True)
    print(f"tier_counts: {tier_counts}", flush=True)


if __name__ == "__main__":
    main()
