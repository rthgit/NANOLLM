# NanoLLM v3.1

NanoLLM is a proprietary mixed-precision quantization and artifact pipeline for Qwen/Qwen2.5 models. This repository is the code and release tooling repo; generated model artifacts are distributed through Hugging Face and are intentionally not committed to GitHub.

## Current Release Artifacts

| Model | Local artifact | Zip size | Test gate | Avg cosine | Min cosine | Locked / 8-bit pending |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | `final_artifact_3B.zip` | 799,189,680 bytes | PASS | 0.990625 | 0.984375 | 143 / 109 |
| Qwen2.5-7B-Instruct | `final_artifact_7B.zip` | 891,419,698 bytes | PASS | 0.990625 | 0.98046875 | 66 / 130 |
| Qwen2.5-14B-Instruct | `final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip` | 1,482,019,132 bytes | PASS | 0.990625 | 0.98046875 | 76 / 260 |

The current gate in `nano_artifact_test.py` requires `avg cosine >= 0.99` plus non-empty greedy generation. It records `min cosine` for diagnostics but does not gate on it.

## Repository Layout

- `kaggle_nano_3B_gpu.py`: 3B RunPod/Kaggle build runner.
- `kaggle_nano_cell_gpu.py`: 7B RunPod/Kaggle build runner.
- `kaggle_nano_universal_v3.py`: universal runner for larger Qwen models; supports `NANO_HF_MAX_WORKERS` for low-quota downloads.
- `nano_artifact_test.py`: artifact smoke/fidelity test.
- `run_nano_tests_matrix.py`: helper for test matrices.
- `release_hf_v31.py`: uploads validated zip artifacts to Hugging Face model repos.
- `NANOHF/load_artifact.py`: inference-only loader used inside artifact zips.
- `NANOHF/README.md`: Hugging Face model card template.

## Artifact Policy

Do not commit model payloads to GitHub. The repository ignores `*.zip`, `*.pt`, `*.safetensors`, `local_runs/`, `results*/`, and token files. Release artifacts should be uploaded with:

```powershell
$env:HF_TOKEN = '<your-token>'
python release_hf_v31.py
```

Default Hugging Face targets:

- `RthItalia/NanoLLM-Qwen2.5-3B-v3.1`
- `RthItalia/NanoLLM-Qwen2.5-7B-v3.1`
- `RthItalia/NanoLLM-Qwen2.5-14B-v3.1`

Set `HF_REPO_ALL` to publish all three zips into one model repo under `3B/`, `7B/`, and `14B/` subfolders.

## Build Notes

The v3.1 runners use an original-baseline reference policy:

- `reference_scope=original_baseline`
- `pending_policy=leave_in_base_8bit`
- no `b8 k0` fallback replacement

This avoids compounding drift and leaves modules that do not pass the cascade in the base bitsandbytes 8-bit model.

## License

The quantization pipeline source is proprietary/internal. Generated artifacts are distributed separately on Hugging Face under the license terms declared in the model card.
