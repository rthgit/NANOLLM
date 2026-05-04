---
language:
  - en
license: other
tags:
  - quantization
  - qwen
  - qwen2.5
  - mixed-precision
  - inference
library_name: transformers
pipeline_tag: text-generation
---

# NanoLLM Releases

This directory documents two different release classes:

1. legacy overlay artifacts
2. the newer self-contained compact winner for `3B`

They are not the same thing and should not be presented as equivalent.

## Recommended Current Release

Recommended public model for `3B`:

- `RthItalia/nano_compact_3b_qkvfp16`
- canonical card draft in this repo: `NANOHF/HF_MODEL_CARD_qkvfp16.md`

Validated policy:

- `q_proj`, `k_proj`, `v_proj` in `fp16`
- `o_proj` and most of the body in Nano compact format
- quantized single-copy embeddings
- tied custom output head over quantized embeddings

Validated envelope:

- model size: `2.3432 GB`
- allocated after load: `2.3432 GB`
- peak generate VRAM: about `2.44 GB`

True `8bit` baseline used for comparison:

- allocated after load: `3.1703 GB`
- peak generate VRAM: about `3.21 GB`

## Legacy Overlay Artifacts

The files below are legacy intermediate research artifacts:

- `final_artifact_3B.zip`
- `final_artifact_7B.zip`
- `final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip`

Those overlay artifacts require a loader that starts from the base model and then replaces selected modules with `TrueQuantLinear`.

That path is still useful for research and debugging, but it is not the final recommended user-facing release path for the validated `3B` winner.

## Legacy Overlay Validation Snapshot

| Model | Artifact | Zip size | Gate | Avg cosine | Min cosine | Locked / 8-bit pending |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | `final_artifact_3B.zip` | 799,189,680 bytes | PASS | 0.990625 | 0.984375 | 143 / 109 |
| Qwen2.5-7B-Instruct | `final_artifact_7B.zip` | 891,419,698 bytes | PASS | 0.990625 | 0.98046875 | 66 / 130 |
| Qwen2.5-14B-Instruct | `final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip` | 1,482,019,132 bytes | PASS | 0.990625 | 0.98046875 | 76 / 260 |

The cosine gate above belongs to the overlay stage, not to the final self-contained compact winner.

## Overlay Quick Start

```python
from load_artifact import load_artifact

model, tokenizer, spec = load_artifact("final_artifact_3B")
prompt = "Write a Python function to sort a list using bubble sort."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Runtime Notes

- `build_reference_mode`: `8bit`
- `reference_scope`: `original_baseline`
- `pending_policy`: `leave_in_base_8bit`
- `NANO_LOAD_4BIT=1` remains experimental for the overlay path

## License

Treat the current public release story as a composite or dual-license setup:

- the upstream Qwen base model keeps its own license terms
- the Nano quantization pipeline and release-specific runtime code keep the Nano repository license terms

Do not collapse those two layers into a single license claim unless the repository legal text is rewritten accordingly.

See also:

- `../LICENSE`
- `../LICENSING.md`
