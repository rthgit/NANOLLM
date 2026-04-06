---
language:
  - en
  - zh
  - it
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

# NanoLLM Qwen v3.1

NanoLLM v3.1 artifacts are compact overlay artifacts for Qwen2.5 models. The loader starts from the base model in bitsandbytes 8-bit mode, then replaces the modules that passed the NanoLLM cascade with `TrueQuantLinear` modules.

## Validated Artifacts

| Model | Artifact | Zip size | Gate | Avg cosine | Min cosine | Locked / 8-bit pending |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | `final_artifact_3B.zip` | 799,189,680 bytes | PASS | 0.990625 | 0.984375 | 143 / 109 |
| Qwen2.5-7B-Instruct | `final_artifact_7B.zip` | 891,419,698 bytes | PASS | 0.990625 | 0.98046875 | 66 / 130 |
| Qwen2.5-14B-Instruct | `final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip` | 1,482,019,132 bytes | PASS | 0.990625 | 0.98046875 | 76 / 260 |

The current release gate checks average next-token-logit cosine similarity against the 8-bit reference: `avg >= 0.99`. Minimum cosine is reported as a diagnostic.

## Quick Start

```python
from load_artifact import load_artifact

model, tokenizer, spec = load_artifact("final_artifact_Qwen2.5-14B-Instruct")
prompt = "Write a Python function to sort a list using bubble sort."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Requirements:

```bash
pip install torch transformers accelerate bitsandbytes safetensors
```

## Runtime Notes

- `build_reference_mode`: `8bit`
- `reference_scope`: `original_baseline`
- `pending_policy`: `leave_in_base_8bit`
- `NANO_LOAD_4BIT=1` can be used experimentally to load the base model in 4-bit, but the release tests use 8-bit.

## License

The NanoLLM quantization pipeline is proprietary/internal. Generated artifacts are published for research and evaluation subject to the repository license terms.
