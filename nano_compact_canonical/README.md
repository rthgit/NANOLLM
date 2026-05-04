# Nano Compact Canonical

Canonical minimal source tree for the validated `3B` compact winner.

## Scope

This directory is intentionally narrow:

- target model family: `Qwen/Qwen2.5-3B-Instruct`
- validated winner policy: `qkvfp16`
- no cascade rebuild logic
- no notebook-only hotfixes

## Winner Policy

- `model.layers.*.self_attn.q_proj`: `fp16`
- `model.layers.*.self_attn.k_proj`: `fp16`
- `model.layers.*.self_attn.v_proj`: `fp16`
- `model.layers.*.self_attn.o_proj`: Nano compact
- `mlp.*`: Nano compact
- `model.embed_tokens`: quantized single copy
- `lm_head`: tied custom head over quantized embeddings

## Validated Envelope

- size: about `2.3432 GB`
- load VRAM: about `2.3432 GB`
- peak VRAM: about `2.44 GB`

True `8bit` baseline used for comparison:

- load VRAM: `3.1703 GB`
- peak VRAM: about `3.21 GB`

Published winner:

- `RthItalia/nano_compact_3b_qkvfp16`

## Files

- [`export_nano_compact.py`](./export_nano_compact.py)
- [`modeling_nanollm.py`](./modeling_nanollm.py)
- [`smoke_test.py`](./smoke_test.py)
- [`build_qkvfp16_variant.py`](./build_qkvfp16_variant.py)
- [`HF_MODEL_CARD_qkvfp16.md`](./HF_MODEL_CARD_qkvfp16.md)

## Typical Flow

1. Build the original overlay artifact with the existing cascade runner.
2. Run `export_nano_compact.py` with `--variant qkvfp16`.
3. Run `smoke_test.py` on the exported folder.
4. Publish the exported folder if the metrics match the validated envelope.

## License

This directory should be read together with [`../LICENSING.md`](../LICENSING.md).
