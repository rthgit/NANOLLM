# NANO / NanoLLM

Repository status: transitional.

This repository currently mixes:

- historical research directions
- legacy overlay-artifact release scripts
- the newer validated compact self-contained `3B` winner

The important point is this:

- the validated released winner is **not** the old overlay zip path
- the validated released winner is **`RthItalia/nano_compact_3b_qkvfp16`**

## What Was Actually Validated

Validated compact winner:

- base model: `Qwen/Qwen2.5-3B-Instruct`
- policy:
  - `q_proj`, `k_proj`, `v_proj` in `fp16`
  - `o_proj` and most of the body in Nano compact format
  - quantized single-copy embeddings
  - tied custom output head over the quantized embeddings

Measured envelope:

- model size: about `2.3432 GB`
- allocated after load: about `2.3432 GB`
- peak generate VRAM: about `2.44 GB`

True `8bit` baseline used for comparison:

- allocated after load: `3.1703 GB`
- peak generate VRAM: about `3.21 GB`

Published model:

- `https://huggingface.co/RthItalia/nano_compact_3b_qkvfp16`

## What This Repo Contains Today

- root docs and historical notes
- `NANOHF/`
  - release docs for both legacy overlay artifacts and the current self-contained winner
- `nano_compact_canonical/`
  - canonical minimal exporter/runtime tree for the validated `qkvfp16` winner
- `nano_native_inference.py`
  - research/native-shell direction
- `release_hf_v31.py`
  - legacy artifact release script

## What Is Legacy vs Current

### Current validated path

Use the self-contained compact model:

- `RthItalia/nano_compact_3b_qkvfp16`

This is the path that was actually tested end-to-end for:

- size
- VRAM
- smoke quality against the true `8bit` baseline

### Legacy path

The old `final_artifact_*.zip` flow is still useful as an intermediate research artifact, but it is not the best final user-facing release path for the validated `3B` winner.

## Canonical Winner Source

The validated `qkvfp16` winner now has a dedicated minimal source tree in:

- `nano_compact_canonical/`

That directory is the cleanest current source-of-truth for:

- final exporter
- final `modeling_nanollm.py`
- final smoke test
- final HF model card

## Recommended Runtime Example

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

## License

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [LICENSING.md](LICENSING.md).

## Notes

- `trust_remote_code=True` is required for the published compact winner.
- The repository is currently documented as a dual-license or dual-layer research distribution.
- Built with Qwen.
- Redistributed release folders should carry both `LICENSE` and `NOTICE`.
- The claims around Radial-Former / native-bit shell remain research-direction material unless separately revalidated as a shipped runtime path.
