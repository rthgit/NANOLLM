# Run Decisions

## Phase 1 Closed

Phase 1 is closed.

Primary result:
- winner: `best current non-standalone blend stack`
- promoted bundle: `nanollm_best_composed_bundle_v2.zip`
- SHA256: `af900e14df24ab021d5b72df06a42c113124a3f168bb720917007b7233223cc1`

Canonical alpha state:
- `internal = 0.17208333333333334`
- `head = 0.072`
- `embed = 0.0175`

Canonical best eval:
- `han = 0`
- `loop = 0`
- `short = 0`
- `generic = 0`
- `uniq = 0.8902088377723971`

Decision:
- `Promote current winner`
- `Do not promote standalone or grouped-handoff branches`

## Phase 2 Standalone 3B Tail Campaign (v1-v16) - CLOSED

Decision date: `2026-03-11`

Final decision:
- `No standalone promotion from v1-v16`
- `Freeze v13 as best standalone research checkpoint`
- `Reject v14, v15, and v16 as promotion candidates`

Why:
- Best closed-cycle standalone probe (V13) stays below official baseline quality (`uniq 0.8631519989336286 < 0.8902088377723971`).
- V15 best probe (`step=20`) remained below V13 and baseline (`uniq 0.8552016364699006`).
- V16 selector gate favored `step=40` although qualitative output at `step=60` was better, causing low selected probe quality (`uniq 0.7938834154351395`).
- Stability improvements were not sufficient to beat baseline on overall answer quality.

Frozen research artifacts:
- `/kaggle/working/phase2_standalone_tail_v13_hard_guard_best.pt`
- `/kaggle/working/phase2_standalone_tail_v13_hard_guard_summary.json`

Operational rule from now:
- Stop standalone tail micro-steps in this exact recipe (`v17+` blocked).
- Restart only with material regime change (objective, cache/data, or selector design).




## V13C + QNanoLoRA merge validation (2026-03-12)

Decision:
- `Promote candidate for production handoff` with hard gates passed.

Canonical run facts:
- adapter: `/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_std`
- comparison:
  - `baseline uniq = 0.8902088377723971`
  - `merged uniq = 0.8638771186440678`
  - `gap uniq = 0.0263317191283293` (within `<= 0.03`)
- safety gates:
  - `semantic_fail_count = 0`
  - `short = 0`

Artifacts:
- `/kaggle/working/v13c_qnanolora_merge_clean_eval.json`
- `production_candidate_v13c_qnanolora_<timestamp>.zip` (downloaded)

Repro notes:
- See `REPRO_CHANGELOG_20260312_V13C_QNANOLORA.md` for exact in-run fixes.

## Extended benchmark update (2026-03-12, 20 prompts)

Scope:
- direct compare: `RAW_LLAMA_3B_20P` vs `V13C_PLUS_QNANOLORA_20P`

Observed metrics:
- `RAW uniq = 0.8322959487659384`
- `CAND uniq = 0.8014548242398506`
- `uniq delta (cand-raw) = -0.0308411245260878`
- `RAW avg_nll = 5.6187`
- `CAND avg_nll = 5.7043`
- `nll delta (cand-raw) = +0.0855`
- `ppl ratio = 1.0893`
- semantic hard failures: `0 -> 0`
- prompt-level uniq wins: `cand 7`, `raw 13`

Decision refinement:
- Candidate remains valid for project hard-gate handoff workflow.
- Candidate is **not** a strict raw-3B replacement under extended quality criterion.

## Unified compression benchmark (2026-03-12)

Canonical report:
- `/kaggle/working/compression_methods_benchmark_3b.json`

Methods executed:
- RAW FP16, RAW INT8 (BnB), RAW NF4 4-bit (BnB), V13C+QNanoLoRA merged
- AWQ/GPTQ local artifacts not available in this run

Headline metrics:
- `RAW_FP16`: uniq `0.8323`, avg_nll `5.6187`, latency `2.45s`
- `RAW_INT8_BNB`: uniq `0.8030`, avg_nll `5.6133`, latency `7.94s`
- `RAW_NF4_4BIT_BNB`: uniq `0.7945`, avg_nll `5.6660`, latency `4.80s`
- `V13C_QNANOLORA_MERGED`: uniq `0.8180`, avg_nll `5.6753`, latency `5.31s`

Decision note:
- `V13C_QNANOLORA_MERGED` is best uniq-retention among compressed methods tested.
- It is not best on NLL/latency in this benchmark.
- Keep as compressed handoff candidate, not as universal best compression choice.
