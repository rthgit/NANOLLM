# NanoLLM 3B - Technical Paper (Revision 2026-03-12)

**Revision date:** 2026-03-12
**Program status:** standalone 3B tail cycle closed; adapter integration campaign completed
**Current program decision:** production candidate promoted from `V13C + QNanoLoRA merged` under fixed hard gates, with explicit quality-cost tradeoff documented

## Abstract

This paper documents the current technical state of NanoLLM 3B after three phases:

1. Phase 1 composed baseline promotion,
2. Phase 2 standalone-tail closure,
3. deterministic QNanoLoRA integration over V13C.

The standalone tail campaign (`tail_v1 -> tail_v16`) remains closed with no standalone-only promotion over the official baseline. The integration path `V13C + QNanoLoRA merged` passed formal hard gates in closeout conditions, was packaged as a production handoff candidate, and was then stress-checked against raw 3B with an extended benchmark.

Main finding of this revision:
- extreme compression (adapter-scale artifacts) is achievable with hard safety stability preserved,
- but a measurable quality/loss gap appears on broader prompt coverage and likelihood metrics.

## 1. Scope and Claims

This document does not claim universal SOTA. It claims a reproducible engineering result in a constrained environment:

- model family: `unsloth/Llama-3.2-3B`
- runtime: single GPU Kaggle T4-like envelope (`~14.56 GiB`)
- evaluation style: deterministic prompt sets and fixed decode policy
- promotion rule: hard gates, not subjective preference

Primary contribution in this revision:
- a deterministic, size-locked QNanoLoRA adapter protocol integrated into V13C with auditable in-run fixes and replay contract.

## 2. Canonical Baseline (Frozen)

Official baseline (Phase 1 winner, non-standalone composed stack):

- bundle: `nanollm_best_composed_bundle_v2.zip`
- SHA256: `af900e14df24ab021d5b72df06a42c113124a3f168bb720917007b7233223cc1`

Canonical blend coefficients:
- `internal = 0.17208333333333334`
- `head = 0.072`
- `embed = 0.0175`

Canonical baseline metrics:
- `han = 0`
- `loop = 0`
- `short = 0`
- `generic = 0`
- `uniq = 0.8902088377723971`

This baseline remains the official quality reference.

## 3. Phase 2 Standalone Tail Closure (Historical Context)

Standalone campaign status:
- closed cycle: `tail_v1 -> tail_v16`
- no standalone-only promotion

Best standalone research checkpoint:
- `phase2_standalone_tail_v13_hard_guard_best.pt`
- `best_step = 60`
- `semantic_fail_count = 0`
- `uniq = 0.8631519989336286`

Rejected late branches:
- `v14_ce_anchor`: regression trend
- `v15_selector_guard`: below V13 and below baseline
- `v16_final_gate`: selector mismatch and quality drop at selected step

Operational policy from closure:
- block `v17+` micro-iterations in same standalone-tail recipe
- only resume standalone with objective/data/selector regime change

## 4. QNanoLoRA Deterministic Adapter Specification

### 4.1 Goal

Build a size-locked adapter close to target budget while preserving deterministic replay.

### 4.2 Locked configuration

- `TARGET_BYTES = 141_792_565`
- `TARGET_MODULES = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
- `RANK = 23`
- `LORA_ALPHA = 46`
- `LORA_DROPOUT = 0.05`
- `BIAS = none`
- calibration constants:
  - `params_per_rank = 1_519_616`
  - `bytes_per_param_real = 4.071028828`
  - estimated size `~142_287_212 B`

Observed output artifact size:
- standard adapter: `157_073_597 B`
- fp16 export: `87_165_747 B`

## 5. Training Run (V2 KL-TopK, Runtime-Anchored)

Method label:
- `phase1_dream_seed_v2_kl_topk`

Dream generation/filtering:
- total dreams: `640`
- accepted: `629`
- rejected: `11` (`low_uniq=3`, `word_count=8`)

Teacher soft-target prep:
- windows prepared: `442`
- train windows: `397`
- validation windows: `45`

Optimization:
- max steps: `160`
- objective: `CE + KL(top-k)`
- `kl_topk = 32`
- `kl_temp = 1.3`
- `ce_w = 0.35`
- `kl_w = 0.65`
- sequence length: `224`

Best checkpoint from training:
- `best_val_step = 160`
- `best_val_loss = 27.72745966911316`

Output dirs:
- `/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_std`
- `/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_fp16`

## 6. Runtime Integration: V13C + QNanoLoRA Merge

Integration order:
1. load dense base model
2. merge LoRA deltas into dense linears
3. apply V13C blend stack (internal/group + lexicon/head)
4. evaluate under fixed protocol

Merge equation:

`W_merged = W_base + (lora_alpha / rank) * (B @ A)`

Observed merge stats:
- `merged = 196`
- `skipped = 0`
- scaling `= 46 / 23 = 2.0`

## 7. In-Run Deterministic Fixes (Canonicalized)

The successful run required explicit fixes that are now part of replay contract:

1. execute cell inline when markdown source was unavailable in Kaggle mount
2. fallback artifact resolution:
   - missing `grouped_mlp_hidden_anchor_v13c_best.pt`
   - fallback to `phase2_standalone_tail_v13_hard_guard_best.pt`
3. V13 checkpoint normalization:
   - fallback `state -> tail_state`
   - fallback metadata: `best_group_alpha`, `best_step`, `best_cached_kl`
   - deterministic `wrap_* -> layer_proj` mapping when needed
4. hard pre-eval assertion:
   - `len(v13['state']) > 0`
5. robust LoRA key parser with and without `.default` suffix

Full trace:
- `REPRO_CHANGELOG_20260312_V13C_QNANOLORA.md`

## 8. Formal Evaluation Protocol

Prompt protocol:
- fixed core prompt sets (4-prompt closeout and 20-prompt extension)

Decode protocol:
- `max_new_tokens = 64`
- `do_sample = False`
- `repetition_penalty = 1.10`
- `no_repeat_ngram_size = 3`

Promotion hard gates:
- `semantic_fail_count == 0`
- `short == 0`
- `gap_uniq_vs_baseline <= 0.03`

## 9. Result A (Closeout): Baseline V13C vs V13C + QNanoLoRA

Baseline V13C:
- `uniq = 0.8902088377723971`
- `semantic_fail_count = 0`
- `short = 0`

V13C + QNanoLoRA:
- `uniq = 0.8638771186440678`
- `semantic_fail_count = 0`
- `short = 0`

Delta:
- `gap_uniq_vs_baseline = 0.0263317191283293`

Gate verdict:
- `pass_hard_gates = True`
- `close_enough_vs_baseline = True`
- `better_or_equal_semantic = True`

This is the formal reason the candidate was packaged.

## 10. Result B (Short Compare): Raw 3B vs V13C + QNanoLoRA (4 prompts + NLL)

Raw Llama 3.2 3B:
- `uniq = 0.8599371260805874`
- `semantic_fail_count = 0`
- `avg_nll = 4.859243106842041`
- `ppl = 128.92658152042966`

V13C + QNanoLoRA:
- `uniq = 0.8638771186440678`
- `semantic_fail_count = 0`
- `avg_nll = 5.025225257873535`
- `ppl = 152.20453752191045`

Delta candidate - raw:
- `uniq_delta = +0.0039399925634804`
- `nll_delta = +0.165982151031494`
- `ppl_ratio = 1.1806`

Interpretation:
- slight uniq gain on this very short prompt set,
- but likelihood degradation.

## 11. Result C (Extended Benchmark): Raw 3B vs V13C + QNanoLoRA (20 prompts)

### 11.1 Extended generation results

Raw 3B:
- `uniq = 0.8322959487659384`
- `semantic_fail_count = 0`
- `short = 0`
- `loop = 0`
- `generic = 1`
- `latency_avg_s = 2.8001569390296934`

V13C + QNanoLoRA:
- `uniq = 0.8014548242398506`
- `semantic_fail_count = 0`
- `short = 0`
- `loop = 0`
- `generic = 1`
- `latency_avg_s = 5.441246366500854`

Prompt-level uniq wins:
- candidate better: `7`
- raw better: `13`
- ties: `0`

### 11.2 Extended likelihood results

- `RAW avg_nll = 5.6187`
- `CAND avg_nll = 5.7043`
- `NLL delta = +0.0855`
- `PPL ratio = 1.0893` (candidate about `+8.93%` worse)

### 11.3 Net conclusion from extended benchmark

Against raw 3B, the candidate shows:
- preserved hard stability (`semantic_fail_count` unchanged at 0),
- measurable quality loss on broader prompt coverage,
- measurable likelihood degradation,
- higher latency.

So, for a "raw replacement" criterion, the candidate is not superior.

## 12. Size vs Quality Tradeoff (Quantified)

Approximate dense model footprint observed in run context:
- raw download: `~6.43 GB`

Adapter footprint:
- standard: `157 MB`
- fp16 export: `87 MB`

Compression ratio versus dense:
- `~41x` smaller (157 MB)
- `~74x` smaller (87 MB)

Observed tradeoff (20-prompt benchmark):
- `uniq` relative drop versus raw: about `-3.7%`
- `PPL` relative increase versus raw: about `+8.9%`
- hard safety stability: unchanged

Practical meaning:
- very large memory footprint reduction with moderate quality degradation.

## 13. Promotion and Packaging

Promotion status in program workflow:
- promoted as production handoff candidate under current hard-gate policy
- not promoted as strict raw-3B replacement after extended benchmark

Canonical eval artifacts:
- `/kaggle/working/v13c_qnanolora_merge_clean_eval.json`
- `/kaggle/working/llama3b_vs_v13c_qnanolora_compare_20prompts.json`

Bundle produced and downloaded:
- `production_candidate_v13c_qnanolora_<timestamp>.zip`

Bundle minimum content:
- adapter directory (`*_std`)
- eval report JSON(s)
- training summary JSON
- manifest

## 14. Compression Methods Benchmark Plan (Unified Protocol)

To compare NanoLLM with other compression families under one protocol, methods should be benchmarked side-by-side on identical prompts, identical decode config, and identical hardware envelope.

Candidate method set for the unified benchmark:
- RAW FP16 (reference)
- BitsAndBytes INT8
- BitsAndBytes NF4 4-bit
- V13C + QNanoLoRA merged
- (optional, if artifacts exist) AWQ 4-bit
- (optional, if artifacts exist) GPTQ 4-bit
- (optional) standalone V13 tail variants for historical reference

Core metrics:
- hard stability: `semantic_fail_count`, `short`, `loop`, `han`
- generation diversity: `uniq`
- likelihood: `avg_nll`, `ppl`
- runtime: `latency_avg_s`
- footprint: on-disk model/adapters and VRAM at eval time

Output requirement:
- single JSON report with all methods and deltas vs RAW FP16

## 15. Reproducibility Contract

Replay is valid only if all are fixed:
- base model id,
- adapter dir/version,
- V13 artifact fallback logic,
- checkpoint normalization rules,
- prompt set,
- decode config,
- gate thresholds.

Operational files for audit:
- `RUN_DECISIONS.md`
- `TASKS.md`
- `REPRO_CHANGELOG_20260312_V13C_QNANOLORA.md`

## 16. Limitations and Risk Statement

Current limitations:
- benchmark still single-hardware and single-context-length,
- no multi-seed confidence intervals yet,
- optional methods (AWQ/GPTQ) require compatible model artifacts.

Risk statement:
- current promotion is valid for project hard-gate workflow,
- not evidence of universal superiority over raw dense inference.

## 17. Final Conclusion

NanoLLM 3B now has:
- a frozen canonical baseline,
- a closed standalone-tail branch,
- a deterministic adapter-integration candidate with full in-run fix trace,
- explicit quantified compression/quality tradeoff.

The engineering objective was met: produce a reproducible, aggressively compressed candidate that preserves hard stability and is fully auditable.

## 18. Unified Compression Benchmark Results (2026-03-12)

This section records the unified, same-harness benchmark over 20 prompts and shared NLL set.

Method set executed:
- `RAW_FP16`
- `RAW_INT8_BNB`
- `RAW_NF4_4BIT_BNB`
- `V13C_QNANOLORA_MERGED`
- `RAW_AWQ_LOCAL` (not available)
- `RAW_GPTQ_LOCAL` (not available)

Canonical output file:
- `/kaggle/working/compression_methods_benchmark_3b.json`

### 18.1 Raw metrics

- `RAW_FP16`: `uniq=0.8323`, `avg_nll=5.6187`, `ppl=275.54`, `sem_fail=0`, `lat=2.45s`
- `RAW_INT8_BNB`: `uniq=0.8030`, `avg_nll=5.6133`, `ppl=274.03`, `sem_fail=0`, `lat=7.94s`
- `RAW_NF4_4BIT_BNB`: `uniq=0.7945`, `avg_nll=5.6660`, `ppl=288.88`, `sem_fail=0`, `lat=4.80s`
- `V13C_QNANOLORA_MERGED`: `uniq=0.8180`, `avg_nll=5.6753`, `ppl=291.57`, `sem_fail=0`, `lat=5.31s`

### 18.2 Delta vs RAW_FP16

- `RAW_INT8_BNB`: `d_uniq=-0.0293`, `d_nll=-0.0055`, `ppl_ratio=0.9945`, `sem_fail_delta=0`, `latency_ratio~3.24x`
- `RAW_NF4_4BIT_BNB`: `d_uniq=-0.0378`, `d_nll=+0.0473`, `ppl_ratio=1.0484`, `sem_fail_delta=0`, `latency_ratio~1.96x`
- `V13C_QNANOLORA_MERGED`: `d_uniq=-0.0143`, `d_nll=+0.0566`, `ppl_ratio=1.0582`, `sem_fail_delta=0`, `latency_ratio~2.17x`

### 18.3 Ranking by metric (among compressed methods)

- Uniq retention (higher is better):
  1. `V13C_QNANOLORA_MERGED` (`0.8180`)
  2. `RAW_INT8_BNB` (`0.8030`)
  3. `RAW_NF4_4BIT_BNB` (`0.7945`)

- Likelihood retention by NLL (lower delta is better):
  1. `RAW_INT8_BNB` (`-0.0055`)
  2. `RAW_NF4_4BIT_BNB` (`+0.0473`)
  3. `V13C_QNANOLORA_MERGED` (`+0.0566`)

- Latency (lower is better):
  1. `RAW_NF4_4BIT_BNB` (`4.80s`)
  2. `V13C_QNANOLORA_MERGED` (`5.31s`)
  3. `RAW_INT8_BNB` (`7.94s`)

All tested methods preserved hard stability (`semantic_fail_count=0`).

### 18.4 Decision impact

Under unified multi-method benchmark, `V13C_QNANOLORA_MERGED` remains a valid compressed handoff candidate with strong uniqueness retention among compressed variants, but it is not the best method on likelihood or latency.

Therefore:
- keep it as a reproducible compressed candidate,
- do not frame it as globally best compression method without additional optimization passes.

### 18.5 Paper-ready benchmark table (size + GPU + quality)

Hardware envelope for all methods in this report:
- single GPU run
- total VRAM observed by runtime: `~14.56 GiB`

| Method | Artifact size used in this campaign | Relative size vs RAW FP16 (~6.43 GB) | GPU VRAM used during eval (GiB) | Avg latency (s) | Uniq | Avg NLL | PPL | Semantic fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_FP16 | ~6.43 GB dense weights | 1.00x | 6.16 (observed) | 2.45 | 0.8323 | 5.6187 | 275.54 | 0 |
| RAW_INT8_BNB | runtime-quantized (no packaged artifact in this run); online estimate ~3.22 GB equivalent weights | ~2.00x smaller vs RAW FP16 (online estimate) | not emitted in summary JSON excerpt | 7.94 | 0.8030 | 5.6133 | 274.03 | 0 |
| RAW_NF4_4BIT_BNB | runtime-quantized (no packaged artifact in this run); online estimate ~1.65-1.81 GB equivalent weights | ~3.55x to ~3.90x smaller vs RAW FP16 (online estimate) | not emitted in summary JSON excerpt | 4.80 | 0.7945 | 5.6660 | 288.88 | 0 |
| V13C_QNANOLORA_MERGED | adapter: 157 MB std (87 MB fp16 export) + dense base at runtime | ~41x smaller adapter vs dense (std), ~74x (fp16 export) | 6.81 (observed) | 5.31 | 0.8180 | 5.6753 | 291.57 | 0 |

Interpretation of size row:
- for BnB INT8/NF4 in this run, quantization was applied at load-time from the same dense base; no dedicated packaged quantized artifact was produced in this benchmark output.
- for `V13C_QNANOLORA_MERGED`, the compressed distributable component is the adapter, but runtime still loads dense base weights before applying merge.
- INT8/NF4 size values above are online-derived estimates mapped to the observed RAW footprint (~6.43 GB).


### 18.6 Online-backed estimate model for INT8 and NF4

Primary sources:
- Hugging Face Transformers bitsandbytes docs: quantizing in 8-bit approximately halves memory usage, and nested quantization can save about 0.4 bits/parameter in 4-bit mode.
  - https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
- QLoRA paper (NF4 + double quantization concept):
  - https://arxiv.org/abs/2305.14314

Applied estimates using observed RAW footprint `~6.43 GB`:

- INT8 estimate:
  - rule: `8-bit ~= 50% of fp16 weight memory`
  - size: `6.43 * 0.5 ~= 3.22 GB`

- NF4 estimate (range):
  - baseline 4-bit with overhead often approximated around `~4.5 bits/param`
  - nested/double quantization reduces overhead by about `~0.4 bits/param` (HF docs), bringing effective total around `~4.1 bits/param`
  - mapped to RAW 16-bit equivalent:
    - upper range (`4.5/16`): `6.43 * 0.28125 ~= 1.81 GB`
    - lower range (`4.1/16`): `6.43 * 0.25625 ~= 1.65 GB`

Caveat:
- these are theoretical-equivalent weight footprint estimates from online documented rules; real runtime VRAM depends on kernels, activation buffers, KV cache, and framework implementation.
