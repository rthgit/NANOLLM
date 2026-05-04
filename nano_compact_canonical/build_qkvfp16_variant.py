import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def restore_quantized_o_proj(tied_dir: Path, attention_dir: Path, out_dir: Path, move_attention_dir: bool = True):
    if out_dir.exists():
        shutil.rmtree(out_dir)

    if move_attention_dir:
        attention_dir.rename(out_dir)
    else:
        shutil.copytree(attention_dir, out_dir)

    tied_cfg = load_json(tied_dir / "config.json")
    tied_mods = tied_cfg.get("nanollm_modules", {})

    targets = sorted(
        name
        for name in tied_mods.keys()
        if ".self_attn." in name and name.split(".")[-1] == "o_proj"
    )
    if not targets:
        raise RuntimeError("no self_attn.o_proj modules found in tied config")

    model_path = out_dir / "model.safetensors"
    tensors = dict(load_file(str(model_path)))

    with safe_open(str(tied_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        for mod in targets:
            for key in list(tensors.keys()):
                if key.startswith(mod + "."):
                    del tensors[key]

            mod_keys = [key for key in all_keys if key.startswith(mod + ".")]
            if not mod_keys:
                raise RuntimeError(f"missing tensors for {mod} in tied variant")

            for key in mod_keys:
                tensors[key] = f.get_tensor(key)

    model_path.unlink()
    save_file(tensors, str(model_path), metadata={"format": "pt"})

    for name in ("config.json", "nano_compact_spec.json"):
        out_path = out_dir / name
        tied_path = tied_dir / name
        if not out_path.exists() or not tied_path.exists():
            continue

        out_data = load_json(out_path)
        tied_data = load_json(tied_path)
        out_mods = out_data.get("nanollm_modules")
        tied_mods_local = tied_data.get("nanollm_modules", {})

        if isinstance(out_mods, dict):
            for mod in targets:
                if mod in tied_mods_local:
                    out_mods[mod] = tied_mods_local[mod]

        out_data["tie_word_embeddings"] = False
        write_json(out_path, out_data)


def write_freeze_metadata(out_dir: Path) -> None:
    metrics = {
        "variant": "nano_compact_3b_qkvfp16",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "policy": "q_proj,k_proj,v_proj in fp16; o_proj and rest Nano compact; tied quantized embedding head",
        "total_gb": 2.3432,
        "allocated_after_load_gb": 2.3432,
        "peak_generate_gb": 2.44,
        "baseline_true_8bit_load_gb": 3.1703,
        "baseline_true_8bit_peak_gb": 3.21,
        "status": "frozen_winner",
        "published_repo": "RthItalia/nano_compact_3b_qkvfp16",
    }
    write_json(out_dir / "FREEZE_METRICS.json", metrics)

    readme = """---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- nanollm
- qwen2
- compact
- quantization
- custom_code
---

# Nano Compact 3B QKV-FP16

Frozen winning compact variant derived from `Qwen/Qwen2.5-3B-Instruct`.

## Policy

- `q_proj`, `k_proj`, `v_proj`: fp16
- `o_proj` and remaining body: Nano compact
- embeddings: quantized single copy
- lm head: tied custom head over quantized embeddings

## Validated runtime envelope

- total size: `2.3432 GB`
- allocated after load: `2.3432 GB`
- peak generate: `~2.44 GB`

## True 8bit baseline

- allocated after load: `3.1703 GB`
- peak generate: `~3.21 GB`

Requires `trust_remote_code=True`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tied-dir", required=True)
    parser.add_argument("--attention-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy attention-dir instead of renaming it into out-dir.")
    parser.add_argument("--write-freeze-metadata", action="store_true")
    args = parser.parse_args()

    tied_dir = Path(args.tied_dir).resolve()
    attention_dir = Path(args.attention_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not tied_dir.exists():
        raise RuntimeError(f"missing tied variant dir: {tied_dir}")
    if not attention_dir.exists():
        raise RuntimeError(f"missing attention variant dir: {attention_dir}")

    restore_quantized_o_proj(
        tied_dir=tied_dir,
        attention_dir=attention_dir,
        out_dir=out_dir,
        move_attention_dir=not args.copy,
    )

    if args.write_freeze_metadata:
        write_freeze_metadata(out_dir)

    print(f"built winner variant at: {out_dir}")


if __name__ == "__main__":
    main()
