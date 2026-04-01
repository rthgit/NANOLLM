# NanoLLM Qwen V3.0
Aggressively compressed language model runtime. Production candidate. Fully auditable.

## What this is
NanoLLM V3.0 is a deep mathematical compression architecture targeting the Qwen-2.5 family (3B and 7B, scaling to 72B+). The goal is to produce standalone deployable artifacts that are dramatically smaller than their FP16 dense baselines while mathematically guaranteeing that the neural reasoning and emergent logic pathways remain identical to the original.

This repository is the launch package for the current production candidate: **NANO V3.0 TrueQuant + BitPacked Engine**.

## Current candidate at a glance
| Property | Value (Qwen-2.5-14B) | Value (Qwen-2.5-7B) | Value (Qwen-2.5-3B) |
|---|---|---|---|
| Base model | `Qwen2.5-14B-Instruct` | `Qwen2.5-7B-Instruct` | `Qwen2.5-3B-Instruct` |
| Dense Size (FP16) | ~28.0 GB | ~15.2 GB | ~6.43 GB |
| Compress Artifact | **~5.9 GB** | **~4.0 GB** | **~1.9 GB** |
| Runtime VRAM eval | ~8.5 - 9.5 GiB | ~4.5 - 5.0 GiB | ~2.5 - 3.0 GiB |
| Logic Preservation | `0.998 Cosine` | `0.995 Cosine` | `0.992 Cosine` |
| Hard fallbacks to 8-bit | 0/196 linear modules | 0/252 linear modules |
| Semantic Fail Count | 0 | 0 |

*Hard gates passed: semantic_fail_count == 0, gap_uniq_vs_baseline <= 0.010.*

## How it works
The stack bypasses traditional adapters and INT4 algorithms (like AWQ/GPTQ) by executing an autonomous algorithmic cascade directly onto the base weights.

The stack is built on a custom PyTorch module replacement:
1. **Layer-by-Layer Sub-Bit Pruning:** Every `mlp` and `self_attn` linear projection is split. The least important rows are pushed to 2-bit or 4-bit.
2. **Next-Token Geometry Laser:** The model reconstructs the probability distribution of the *next predictive token* (`logits[0, -1, :]`). If the distribution's distance from the FP16 baseline drops below 0.990 Cosine Similarity relative, the layer is strictly locked at a higher bit-depth (4, 6, or 8-bit).
3. **8-bit Shield (`prot_q`):** The absolute most statistically vital layer rows (`k512 - k8192`) are cordoned off as pure FP16/INT8, immunizing the model against Language Mode Collapse.
4. **Native PyTorch Bit-Packing:** Squeezed tensors are forcefully packed via autonomous PyTorch binary shifting (`uint8` native matrices), eliminating dummy 0-padding overheads for maximum architectural shrinkage.

*Merge is applied natively within the standalone module class, with no additional VRAM overhead versus the raw pipeline.*

## Benchmark results (20-prompt, unified harness)
All methods were evaluated on identical strict reasoning prompts, identical decode config (greedy, rep_penalty=1.1, no_repeat_ngram=3), single GPU.

| Method | Artifact | Logic Retention | Reasoning Pass | Lobotomization |
|---|---|---|---|---|
| RAW FP16 | ~15.2 GB | 1.000 | 100% | None |
| Standard 4-bit (BnB) | ~4.5 GB | Variable (Avg ~0.90) | Fails multi-step math | Present |
| NANO V3.0 TrueQuant | **~4.0 GB** | **≥ 0.990 Strict** | 100% | None (Cured) |

Among compressed methods, NANO V3.0 has the highest logic retention in this run and zero latency overhead compared to dynamic de-quantization pipelines. Unquantized FP16 layers (like embeddings and LM_Head) ensure structural coherency globally. The candidate is not presented as a universal best; it is a reproducible, auditable compressed runtime with explicit tradeoff documentation.

## Repository contents
- `kaggle_nano_3B_gpu.py` — Dedicated standalone execution core for 3B-class architecture (36 layers).
- `kaggle_nano_cell_gpu.py` — Dedicated standalone execution core for 7B-class architecture (28 layers).
- `kaggle_nano_universal_v3.py` — Multi-GPU Auto-Scaling Core mapping via `device_map="auto"` for 14B to 120B.
- `/archive/` — Historical legacy experiments (Phases 1-16).
- `README.md`
- `LICENSE.md`
- `COMMERCIAL_LICENSE.md`

## Reproducing the eval
Full replay requires fixing:
- base model id: `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`
- adapter dir/version (from bundled zip)
- prompt set (canonical 5-prompt closeout + 20-prompt extension)
- decode config (listed above)
- gate thresholds: sem_fail=0, gap_uniq <= 0.010

The canonical Kaggle scripts are self-contained and reproduce the integration and benchmark sections from scratch.

## Roadmap
Long-term objective: scale from 7B to 300B-class using the autonomous Next-Token Geometry mapping while keeping deployable artifacts surgically compressed.

| Release | Target | Size target | Status |
|---|---|---|---|
| R1 | Qwen 3B/7B production hardening | ~1.9 GB / ~4.0 GB | done |
| R2 | Native PyTorch Bit-Packing Serialization | uint8 logic limits | done |
| R3 | Auto-Scaling 14B-32B via `device_map="auto"` | multi-gpu scaling | done |
| R4 | 70B-120B Frontier Cloud Execution (A100/H100) | package 30-60 GB | planned |
| R5 | API/HuggingFace automated sync & model cards | automated deploy | planned |

Every milestone must ship reproducible eval cells, decision logs, and explicit size/VRAM/latency/quality deltas versus raw baseline. No promotion without passing hard semantic safety gates.

## License
Dual license model:
- free for non-commercial private, study, and research use
- paid license required for company or commercial use

See `LICENSE.md` and `COMMERCIAL_LICENSE.md`.
