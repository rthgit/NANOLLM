import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EMBED_CANDIDATES = [
    "model.embed_tokens.weight",
    "embed_tokens.weight",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_support_files(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in base_dir.iterdir():
        if item.is_dir():
            continue
        if item.name == "config.json":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.suffix in {".safetensors", ".bin"}:
            continue
        shutil.copy2(item, out_dir / item.name)

    repo_dir = Path(__file__).resolve().parent.parent
    for extra_name in ("LICENSE", "NOTICE"):
        extra = repo_dir / extra_name
        if extra.exists():
            shutil.copy2(extra, out_dir / extra_name)


def load_base_tensor(base_dir: Path, key: str):
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = read_json(index_path)
        fname = data["weight_map"].get(key)
        if fname is None:
            return None
        with safe_open(str(base_dir / fname), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    single_path = base_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)

    return None


def iter_base_tensors(base_dir: Path):
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = read_json(index_path)
        by_file = {}
        for key, fname in data["weight_map"].items():
            by_file.setdefault(fname, []).append(key)
        for fname, keys in by_file.items():
            with safe_open(str(base_dir / fname), framework="pt", device="cpu") as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        return

    single_path = base_dir / "model.safetensors"
    if not single_path.exists():
        raise RuntimeError(f"missing base weights under {base_dir}")

    with safe_open(str(single_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


def quantize_rowwise_int8(weight: torch.Tensor):
    w = weight.to(torch.float32).contiguous()
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float16).contiguous()


def should_promote_fp16(module_name: str, variant: str) -> bool:
    suffix = module_name.split(".")[-1]
    if ".self_attn." not in module_name:
        return False
    if variant == "tiedq":
        return False
    if variant == "attentionfp16":
        return suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}
    if variant == "qkvfp16":
        return suffix in {"q_proj", "k_proj", "v_proj"}
    raise RuntimeError(f"unsupported variant: {variant}")


def infer_in_features(state: dict) -> int:
    if state["prot_q"].shape[0] > 0:
        return int(state["prot_q"].shape[1])
    bits = int(state["bits"])
    if "deg_q_packed" in state and state["deg_q_packed"].shape[0] > 0:
        packed_cols = int(state["deg_q_packed"].shape[1])
        return packed_cols * (8 // bits) - int(state.get("pad", 0))
    if "deg_q" in state and state["deg_q"].shape[0] > 0:
        return int(state["deg_q"].shape[1])
    raise RuntimeError("cannot infer in_features from empty state")


def build_truequant_spec(state: dict) -> dict:
    return {
        "kind": "truequant_linear",
        "in_features": infer_in_features(state),
        "out_features": int(state["out_features"]),
        "prot_rows": int(state["prot_q"].shape[0]),
        "deg_rows": int(state["deg_idx"].numel()),
        "bits": int(state["bits"]),
        "has_bias": "bias" in state,
    }


def export_variant(artifact_dir: Path, base_dir: Path, out_dir: Path, public_base_id: str, variant: str) -> None:
    spec_path = artifact_dir / "spec.json"
    state_path = artifact_dir / "quantized_modules.pt"
    if not spec_path.exists() or not state_path.exists():
        raise RuntimeError(f"artifact dir must contain spec.json and quantized_modules.pt: {artifact_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_spec = read_json(spec_path)
    artifact_state = torch.load(state_path, map_location="cpu")

    copy_support_files(base_dir, out_dir)

    promoted_modules = {name for name in artifact_state if should_promote_fp16(name, variant)}
    quantized_modules = {name for name in artifact_state if name not in promoted_modules}

    tensors = {}
    for key, tensor in iter_base_tensors(base_dir):
        if key == "lm_head.weight":
            continue
        if key in EMBED_CANDIDATES:
            continue
        owner = key.rsplit(".", 1)[0]
        if owner in quantized_modules:
            continue
        tensors[key] = tensor.contiguous()

    embed_weight = None
    for key in EMBED_CANDIDATES:
        embed_weight = load_base_tensor(base_dir, key)
        if embed_weight is not None:
            break
    if embed_weight is None:
        raise RuntimeError("missing embed_tokens.weight in base model")

    embed_q, embed_scale = quantize_rowwise_int8(embed_weight)
    tensors["model.embed_tokens.q"] = embed_q
    tensors["model.embed_tokens.scale"] = embed_scale

    nanollm_modules = {
        "model.embed_tokens": {
            "kind": "embedding",
            "num_embeddings": int(embed_q.shape[0]),
            "embedding_dim": int(embed_q.shape[1]),
        }
    }

    for name in sorted(quantized_modules):
        state = artifact_state[name]
        nanollm_modules[name] = build_truequant_spec(state)

        tensors[f"{name}.prot_q"] = state["prot_q"].contiguous()
        tensors[f"{name}.prot_scale"] = state["prot_scale"].contiguous()
        tensors[f"{name}.prot_idx"] = state["prot_idx"].contiguous()
        if "deg_q_packed" in state:
            tensors[f"{name}.deg_q_packed"] = state["deg_q_packed"].contiguous()
        else:
            tensors[f"{name}.deg_q"] = state["deg_q"].contiguous()
        tensors[f"{name}.deg_scale"] = state["deg_scale"].contiguous()
        tensors[f"{name}.deg_idx"] = state["deg_idx"].contiguous()
        if "bias" in state:
            tensors[f"{name}.bias"] = state["bias"].contiguous()

    config = read_json(base_dir / "config.json")
    config["architectures"] = ["NanoQwenForCausalLM"]
    auto_map = config.get("auto_map", {})
    auto_map["AutoModelForCausalLM"] = "modeling_nanollm.NanoQwenForCausalLM"
    config["auto_map"] = auto_map
    config["tie_word_embeddings"] = False
    config["nanollm_modules"] = nanollm_modules
    write_json(out_dir / "config.json", config)

    compact_spec = {
        "format": "nano-compact-canonical-v1",
        "base_model_id": public_base_id,
        "source_artifact_format": artifact_spec.get("format", "unknown"),
        "source_reference_mode": artifact_spec.get("build_reference_mode", "unknown"),
        "variant": variant,
        "promoted_fp16_modules": sorted(promoted_modules),
        "quantized_modules": sorted(quantized_modules),
        "nanollm_modules": nanollm_modules,
    }
    write_json(out_dir / "nano_compact_spec.json", compact_spec)

    shutil.copy2(Path(__file__).with_name("modeling_nanollm.py"), out_dir / "modeling_nanollm.py")
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--public-base-id", default="")
    parser.add_argument(
        "--variant",
        choices=["tiedq", "attentionfp16", "qkvfp16"],
        default="qkvfp16",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    public_base_id = args.public_base_id or read_json(artifact_dir / "spec.json").get("base_model_id", "")
    if not public_base_id:
        raise RuntimeError("missing public base model id")

    export_variant(
        artifact_dir=artifact_dir,
        base_dir=base_dir,
        out_dir=out_dir,
        public_base_id=public_base_id,
        variant=args.variant,
    )
    print(f"exported {args.variant} compact to {out_dir}")


if __name__ == "__main__":
    main()
