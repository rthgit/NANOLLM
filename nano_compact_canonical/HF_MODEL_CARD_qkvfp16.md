---
language:
  - en
license: other
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - qwen2.5
  - quantization
  - mixed-precision
  - custom-code
  - text-generation
  - nanollm
model-index:
  - name: nano_compact_3b_qkvfp16
    results:
      - task:
          type: text-generation
        dataset:
          name: Internal 4-prompt smoke suite
          type: internal
        metrics:
          - type: model_size_gb
            value: 2.3432
          - type: vram_load_gb
            value: 2.3432
          - type: vram_peak_generate_gb
            value: 2.44
          - type: baseline_true_8bit_load_gb
            value: 3.1703
          - type: baseline_true_8bit_peak_gb
            value: 3.21
---

# Nano Compact 3B QKV-FP16

`RthItalia/nano_compact_3b_qkvfp16` is the validated compact self-contained variant derived from `Qwen/Qwen2.5-3B-Instruct`.

This release is not the original overlay artifact. It is the final exported self-contained folder that loads directly with `transformers` plus `trust_remote_code=True`.

## What This Variant Is

This model uses a mixed runtime policy:

- `q_proj`, `k_proj`, `v_proj`: stored and loaded in `fp16`
- `o_proj` and most of the remaining transformer body: stored in Nano compact format
- `model.embed_tokens`: stored as a single quantized copy
- `lm_head`: tied custom head over the quantized embeddings

The objective of this policy is not maximum compression at any cost. It is the best validated tradeoff found between:

- disk size
- VRAM usage
- quality relative to the true `8bit` baseline

## Validated Runtime Envelope

Measured on the validated `3B` run:

- model size: `2.3432 GB`
- allocated after load: `2.3432 GB`
- peak generation VRAM: `~2.44 GB`

True `8bit` baseline used for comparison:

- allocated after load: `3.1703 GB`
- peak generation VRAM: `~3.21 GB`

So this winner variant preserved a meaningful VRAM advantage over the `8bit` baseline while recovering enough quality to pass the smoke comparison used during validation.

## Quality Claim

The quality claim for this release is intentionally narrow:

- it was compared against the true `8bit` baseline on a small internal prompt suite
- it is not claimed to match the full original model in all tasks
- it is not claimed to outperform the base model

During development, more aggressive variants such as:

- fully tied quantized head (`tiedq`)
- fully quantized attention

reached better size and VRAM numbers but failed the quality gate against the true `8bit` reference.  
`qkvfp16` was the first variant that restored acceptable behavior on the reference prompt set while keeping a substantial memory advantage.

## How To Load

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "RthItalia/nano_compact_3b_qkvfp16"

tok = AutoTokenizer.from_pretrained(
    repo_id,
    use_fast=True,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    device_map="cuda",
    dtype=torch.float16,
).eval()
```

## Example Generation

```python
messages = [
    {"role": "user", "content": "Explain what a neural network is in exactly 3 simple sentences."}
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out = model.generate(
        **inp,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.08,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

print(tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Requirements

```bash
pip install torch transformers accelerate safetensors
```

`bitsandbytes` is not required for this exported winner variant at runtime.

## Important Notes

- `trust_remote_code=True` is required.
- The custom runtime uses a `NanoTiedHead` implementation that ties output logits to the quantized embedding table without registering the embedding module twice.
- The custom linear layers use chunked forward paths to keep peak VRAM under control.

## Limitations

- Validation was narrow and engineering-driven, not a full benchmark suite.
- This release is specifically tuned around `Qwen/Qwen2.5-3B-Instruct`.
- It should be treated as a compact experimental runtime artifact, not as a drop-in scientific proof of broader architectural claims.

## License Note

This release should be described as a dual-license or dual-layer research distribution.

Built with Qwen.

The intended reading is:

- Qwen-derived materials retain the relevant Qwen Research License obligations
- Nano-specific runtime, packaging, and documentation changes are additional repository-authored research components

For redistributed copies, keep:

- `LICENSE`
- `NOTICE`
- clear indication of modified files where applicable

The Hugging Face metadata still stays at `license: other` because this is not accurately described by a single simple SPDX identifier.

## Provenance

- base model: `Qwen/Qwen2.5-3B-Instruct`
- winner policy name: `qkvfp16`
- published repo: `RthItalia/nano_compact_3b_qkvfp16`
