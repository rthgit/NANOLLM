# NANO Research Diary

## 2026-03-16

### Goal
- Push the 3B canonical int8-control artifact below the int8 baseline without losing exact agreement on the current prompt gate.

### Canonical Baselines
- 3B baseline artifact: `e:\NANO\local_runs\Llama-3.2-3B-NANO-Canonical-v3-Safe88-Embed8`
- 3B baseline size: `3,213,303,584` bytes
- Baseline source: `e:\testmob\out\NB8_NF4_3B_COMPARE\merged_hf`
- 7B canonical baseline also exists, but this phase is focused on 3B.

### Critical Method Fix
- Early row-mixed scans were exported from a different HF source than the baseline artifact.
- That mismatch created false drift and invalidated several earlier negative results.
- The correct source for 3B comparisons is `e:\testmob\out\NB8_NF4_3B_COMPARE\merged_hf`.

### Runtime / Exporter State
- `nano_inference_direct.py` supports `row_mixed` tensors.
- `nano_export_from_map_v4.py` exports row-mixed tensors with int8 base rows plus low-bit selected rows.
- `nano_scan_row_mixed.py` scans a single module against the reference artifact.
- `nano_build_row_scores.py` builds row-level rankings using norm, output usage, and saliency.
- `nano_search_combo_rows.py` now performs cumulative multi-module searches and writes persistent combo logs.
- `nano_search_combo_rows.py` was updated to use compact labels because long Windows paths broke multi-module exports.

### Row-Mixed Single-Module Results
- Module: `model.layers.0.mlp.up_proj`
- Exact-pass frontier:
  - `511` rows at 4-bit: pass
  - `512` rows at 4-bit: fail
- Best verified single-module artifact:
  - `e:\NANO\local_runs\scan_3b_rowmixed_layer0_up_prefix509_510_511_correctsrc\artifacts\model_layers_0_mlp_up_proj_rows0511_b4`

- Module: `model.layers.1.mlp.up_proj`
- Exact-pass result:
  - `512` rows at 4-bit: pass

### Combined Multi-Module Results
- First naive combo:
  - `layer0=511` and `layer1=512`
  - failed exact greedy agreement
- Refined combo search:
  - `layer0=511`, `layer1=448`: pass
  - `layer0=511`, `layer1=479`: pass
  - `layer0=511`, `layer1=480`: fail

### Layer 2 Expansion
- Module: `model.layers.2.mlp.up_proj`
- Fixed combo base:
  - `layer0=511`
  - `layer1=479`
- Results:
  - `64`: pass
  - `128`: pass
  - `256`: pass
  - `512`: pass
  - `768`: pass
  - `1024`: fail
  - `1536`: pass
  - `1664`: pass
  - `1792`: pass
  - `1920`: fail
  - `1984`: fail
- Important note:
  - The search is not monotonic. Some larger prefixes can recover exact agreement after a smaller failing prefix.
- Best verified layer-2 setting in the current chain:
  - `layer0=511`, `layer1=479`, `layer2=1792`

### Layer 3 Expansion
- Module: `model.layers.3.mlp.up_proj`
- Fixed combo base:
  - `layer0=511`
  - `layer1=479`
  - `layer2=1792`
- Results:
  - `64`: pass
  - `128`: pass
  - `256`: pass
  - `512`: pass
  - `1024`: pass
  - `1536`: pass
  - `2048`: pass
  - `3072`: pass
  - `4096`: pass
  - `6144`: pass
  - `8192`: pass
- This means `model.layers.3.mlp.up_proj` can be moved fully to row-mixed 4-bit inside the current combo while still preserving exact agreement on the gate.

### Best Verified Current Artifact
- Artifact:
  - `e:\NANO\local_runs\l3x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4`
- Log:
  - `e:\NANO\local_runs\l3x\combo_results.json`
- Size:
  - `3,196,536,872` bytes
- Delta vs baseline:
  - `-16,766,712` bytes
- Gate metrics:
  - `next_token_agreement = 1.0`
  - `greedy_token_agreement = 1.0`
  - `full_greedy_match_rate = 1.0`

### Important Boundary Conditions
- For cumulative search, interference between modules is real.
- A module that passes alone at a certain row count may fail when combined with another module.
- Therefore the canonical search rule is:
  - always evaluate cumulative artifacts against the baseline reference
  - never assume solo-module safe counts remain safe in combination
- Windows path length is also a practical constraint when labeling multi-module artifacts.

### Immediate Next Step
- Use the best current combo as the fixed base:
  - `layer0=511`
  - `layer1=479`
  - `layer2=1792`
  - `layer3=8192`
- Attack `model.layers.4.mlp.up_proj` next.

## 2026-03-17

### Candidate Retention Policy
- Search artifacts should not be persisted by default.
- `nano_search_combo_rows.py` and `nano_sonar_sweep.py` now support:
  - `--artifact-policy all`
  - `--artifact-policy best`
  - `--artifact-policy none`
- The intended operating mode is:
  - `best` during active search if a live checkpoint is useful
  - `none` when only the final accepted artifact matters

### local_runs Cleanup
- Added cleanup script:
  - `e:\NANO\cleanup_local_runs.py`
- Cleanup policy:
  - keep the 3B canonical baseline
  - keep the 7B canonical baseline
  - keep the current best 3B search artifact from `l3x`
  - archive small JSON logs and maps into `e:\NANO\local_runs\research_logs`
  - delete superseded heavy candidate directories

### Cleanup Result
- Mode applied:
  - `--apply`
- Space freed:
  - `316,065,981,603` bytes
- Preserved artifacts:
  - `e:\NANO\local_runs\Llama-3.2-3B-NANO-Canonical-v3-Safe88-Embed8`
  - `e:\NANO\local_runs\Qwen2.5-7B-NANO-Canonical-v3-Safe88-Embed8`
  - `e:\NANO\local_runs\l3x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4`
- Preserved search metadata:
  - top-level canonical maps and row-score files
  - archived experiment logs under `e:\NANO\local_runs\research_logs`

### Layer 4 Expansion
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l4_m_up_proj.json`
- Verified exact-pass row counts:
  - `64`: pass
  - `128`: pass
  - `256`: pass
  - `512`: pass
  - `1024`: pass
  - `8192`: pass
- Important conclusion:
  - `model.layers.4.mlp.up_proj` can be moved fully to row-mixed `4-bit` while preserving exact agreement on the current gate.
- Best layer-4 artifact:
  - `e:\NANO\local_runs\l4x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4__l4_m_up_proj_8192r_b4`
- Size:
  - `3,184,019,880` bytes
- Delta vs baseline:
  - `-29,283,704` bytes

### Layer 5 Expansion
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l5_m_up_proj.json`
- Verified exact-pass row count:
  - `8192`: pass
- Important conclusion:
  - `model.layers.5.mlp.up_proj` can also be moved fully to row-mixed `4-bit` on top of the layer-4 chain.
- Best layer-5 artifact:
  - `e:\NANO\local_runs\l5x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4__l4_m_up_proj_8192r_b4__l5_m_up_proj_8192r_b4`
- Size:
  - `3,171,502,896` bytes
- Delta vs baseline:
  - `-41,800,688` bytes

### Layer 6 Expansion
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l6_m_up_proj.json`
- Verified exact-pass row count:
  - `8192`: pass
- Important conclusion:
  - `model.layers.6.mlp.up_proj` also passes as a full row-mixed `4-bit` module on top of the existing chain.
- Current best verified artifact:
  - `e:\NANO\local_runs\l6x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4__l4_m_up_proj_8192r_b4__l5_m_up_proj_8192r_b4__l6_m_up_proj_8192r_b4`
- Size:
  - `3,158,985,912` bytes
- Delta vs baseline:
  - `-54,317,672` bytes
- Gate metrics:
  - `loss_delta_mean = 0.000835`
  - `next_token_agreement = 1.0`
  - `greedy_token_agreement = 1.0`
  - `full_greedy_match_rate = 1.0`

### Layer 7 Expansion
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l7_m_up_proj.json`
- Verified exact-pass row count:
  - `8192`: pass
- Important conclusion:
  - `model.layers.7.mlp.up_proj` also passes as a full row-mixed `4-bit` module on top of the existing chain.
- Current best verified artifact:
  - `e:\NANO\local_runs\l7x\artifacts\l0_m_up_proj_0511r_b4__l1_m_up_proj_0479r_b4__l2_m_up_proj_1792r_b4__l3_m_up_proj_8192r_b4__l4_m_up_proj_8192r_b4__l5_m_up_proj_8192r_b4__l6_m_up_proj_8192r_b4__l7_m_up_proj_8192r_b4`
- Size:
  - `3,146,468,928` bytes
- Delta vs baseline:
  - `-66,834,656` bytes
- Gate metrics:
  - `loss_delta_mean = -0.003469`
  - `next_token_agreement = 1.0`
  - `greedy_token_agreement = 1.0`
  - `full_greedy_match_rate = 1.0`

### Full-8192 Sweep Rule
- New working rule:
  - do not build per-layer rankings for middle layers before testing the full module
  - for `8192/8192` row-mixed checks, row order is irrelevant, so a donor ranking file is sufficient
  - continue full-module tests until the first failure
  - only at the first failing layer open a real per-layer ranking and fine search
- Supporting script:
  - `e:\NANO\nano_sweep_full_rows.py`

### First Full-8192 Failure
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_8_27\full_sweep_results.json`
- Tested chain:
  - layers `0..7` fixed as:
    - `layer0 = 511`
    - `layer1 = 479`
    - `layer2 = 1792`
    - `layer3 = 8192`
    - `layer4 = 8192`
    - `layer5 = 8192`
    - `layer6 = 8192`
    - `layer7 = 8192`
- First failing full-module candidate:
  - `model.layers.8.mlp.up_proj = 8192`
- Failure metrics:
  - size = `3,133,951,944` bytes
  - `loss_delta_mean = -0.003307`
  - `next_token_agreement = 1.0`
  - `greedy_token_agreement = 0.950`
  - `full_greedy_match_rate = 0.800`
- Practical conclusion:
  - `layer8` is the first layer that needs real row ranking and fine search.

### Layer 8 Fine Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l8_m_up_proj.json`
- Coarse search logs:
  - `e:\NANO\local_runs\l8c\combo_results.json`
  - `e:\NANO\local_runs\l8low\combo_results.json`
  - `e:\NANO\local_runs\l8mid\combo_results.json`
- Verified results:
  - `1024`: pass
  - `2048`: pass
  - `2304`: pass
  - `2560`: fail
  - `2816`: fail
  - `3072`: fail
  - `4096`: fail
  - `6144`: fail
  - `7168`: fail
  - `8192`: fail
- Current exact frontier for `layer8`:
  - best known exact-pass = `2304`
  - first known fail above it = `2560`
- Best layer-8 artifact:
  - `e:\NANO\local_runs\l8mid\artifacts\chain_08_l8_m_up_proj_2304r_b4`
- Size:
  - `3,142,948,808` bytes
- Delta vs baseline:
  - `-70,354,776` bytes

### Full-8192 Sweep After Layer 8 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_9_27\full_sweep_results.json`
- Fixed chain used:
  - `layer0 = 511`
  - `layer1 = 479`
  - `layer2 = 1792`
  - `layer3 = 8192`
  - `layer4 = 8192`
  - `layer5 = 8192`
  - `layer6 = 8192`
  - `layer7 = 8192`
  - `layer8 = 2304`
- Full-module results:
  - `layer9 = 8192`: pass
  - `layer10 = 8192`: pass
  - `layer11 = 8192`: pass
  - `layer12 = 8192`: fail
- Current best verified artifact:
  - `e:\NANO\local_runs\full8192_sweep_9_27\artifacts\chain_11_l11_m_up_proj_8192r_b4`
- Size:
  - `3,105,397,872` bytes
- Delta vs baseline:
  - `-107,905,712` bytes
- First new failing full-module candidate:
  - `model.layers.12.mlp.up_proj = 8192`
- Failure metrics:
  - `loss_delta_mean = -0.031958`
  - `next_token_agreement = 0.000`
  - `greedy_token_agreement = 0.750`
  - `full_greedy_match_rate = 0.000`

### Layer 12 Fine Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l12_m_up_proj.json`
- Coarse and refinement logs:
  - `e:\NANO\local_runs\l12c\combo_results.json`
  - `e:\NANO\local_runs\l12mid\combo_results.json`
  - `e:\NANO\local_runs\l12high\combo_results.json`
  - `e:\NANO\local_runs\l12edge\combo_results.json`
- Verified results:
  - `1024`: pass
  - `2048`: pass
  - `2304`: pass
  - `2560`: fail
  - `2816`: pass
  - `2880`: pass
  - `2944`: pass
  - `3008`: pass
  - `3040`: fail
  - `3056`: fail
  - `3064`: fail
  - `3072`: fail
  - `4096`: fail
  - `8192`: fail
- Current exact frontier for `layer12`:
  - best known exact-pass = `3008`
  - first known fail above it = `3040`
- Best layer-12 artifact:
  - `e:\NANO\local_runs\l12high\artifacts\chain_12_l12_m_up_proj_3008r_b4`
- Size:
  - `3,100,802,040` bytes
- Delta vs baseline:
  - `-112,501,544` bytes

### Full-8192 Sweep After Layer 12 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_13_27\full_sweep_results.json`
- Fixed chain used:
  - `layer0 = 511`
  - `layer1 = 479`
  - `layer2 = 1792`
  - `layer3 = 8192`
  - `layer4 = 8192`
  - `layer5 = 8192`
  - `layer6 = 8192`
  - `layer7 = 8192`
  - `layer8 = 2304`
  - `layer9 = 8192`
  - `layer10 = 8192`
  - `layer11 = 8192`
  - `layer12 = 3008`
- Full-module results:
  - `layer13 = 8192`: pass
  - `layer14 = 8192`: pass
  - `layer15 = 8192`: fail
- Current best verified artifact after this sweep:
  - `e:\NANO\local_runs\full8192_sweep_13_27\artifacts\chain_14_l14_m_up_proj_8192r_b4`
- Size:
  - `3,075,768,088` bytes
- Delta vs baseline:
  - `-137,535,496` bytes

### Layer 15 Coarse Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l15_m_up_proj.json`
- Coarse log:
  - `e:\NANO\local_runs\l15c\combo_results.json`
- Verified results:
  - `2048`: pass
  - `4096`: fail
  - `6144`: pass
- Important note:
  - `layer15` is also non-monotonic under the current gate.
- Best known exact-pass for `layer15` so far:
  - `6144`
- Best layer-15 artifact:
  - `e:\NANO\local_runs\l15c\artifacts\chain_15_l15_m_up_proj_6144r_b4`
- Size:
  - `3,066,380,448` bytes
- Delta vs baseline:
  - `-146,923,136` bytes

### Full-8192 Sweep After Layer 15 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_16_27\full_sweep_results.json`
- Fixed chain used:
  - `layer0 = 511`
  - `layer1 = 479`
  - `layer2 = 1792`
  - `layer3 = 8192`
  - `layer4 = 8192`
  - `layer5 = 8192`
  - `layer6 = 8192`
  - `layer7 = 8192`
  - `layer8 = 2304`
  - `layer9 = 8192`
  - `layer10 = 8192`
  - `layer11 = 8192`
  - `layer12 = 3008`
  - `layer13 = 8192`
  - `layer14 = 8192`
  - `layer15 = 6144`
- Full-module results:
  - `layer16 = 8192`: pass
  - `layer17 = 8192`: fail
- Current best verified artifact:
  - `e:\NANO\local_runs\full8192_sweep_16_27\artifacts\chain_16_l16_m_up_proj_8192r_b4`
- Size:
  - `3,053,863,472` bytes
- Delta vs baseline:
  - `-159,440,112` bytes
- First new failing full-module candidate:
  - `model.layers.17.mlp.up_proj = 8192`
- Failure metrics:
  - `loss_delta_mean = -0.001897`
  - `next_token_agreement = 1.000`
  - `greedy_token_agreement = 0.950`
  - `full_greedy_match_rate = 0.800`

### Layer 17 Coarse Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l17_m_up_proj.json`
- Coarse log:
  - `e:\NANO\local_runs\l17c\combo_results.json`
- Verified results:
  - `2048`: pass
  - `4096`: pass
  - `6144`: fail
- Current coarse frontier for `layer17`:
  - best known exact-pass = `4096`
  - first known fail above it = `6144`
- Best layer-17 artifact so far:
  - `e:\NANO\local_runs\l17c\artifacts\chain_17_l17_m_up_proj_4096r_b4`
- Size:
  - `3,047,605,176` bytes
- Delta vs baseline:
  - `-165,698,408` bytes

### Layer 17 Refinement
- Refinement log:
  - `e:\NANO\local_runs\l17mid\combo_results.json`
- Verified additional results:
  - `4608`: pass
  - `5120`: pass
  - `5632`: pass
  - `6144`: fail
- Current exact frontier for `layer17`:
  - best known exact-pass = `5632`
  - first known fail above it = `6144`
- Best layer-17 artifact:
  - `e:\NANO\local_runs\l17mid\artifacts\chain_17_l17_m_up_proj_5632r_b4`
- Size:
  - `3,045,258,176` bytes
- Delta vs baseline:
  - `-168,045,408` bytes

### Full-8192 Sweep After Layer 17 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_18_27\full_sweep_results.json`
- Full-module results:
  - `layer18 = 8192`: fail
- Important conclusion:
  - after fixing `layer17`, the next stopping point is immediately `layer18`.

### Layer 18 Coarse Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l18_m_up_proj.json`
- Coarse log:
  - `e:\NANO\local_runs\l18c\combo_results.json`
- Verified results:
  - `1024`: pass
  - `2048`: fail
  - `4096`: pass
- Important note:
  - `layer18` is non-monotonic even in the first coarse sweep.
- Best layer-18 artifact so far:
  - `e:\NANO\local_runs\l18c\artifacts\chain_18_l18_m_up_proj_4096r_b4`
- Size:
  - `3,038,999,880` bytes
- Delta vs baseline:
  - `-174,303,704` bytes

### Layer 18 Refinement
- Refinement log:
  - `e:\NANO\local_runs\l18mid\combo_results.json`
- Verified additional results:
  - `5120`: pass
  - `6144`: pass
  - `7168`: fail
- Current exact frontier for `layer18`:
  - best known exact-pass = `6144`
  - first known fail above it = `7168`
- Best layer-18 artifact:
  - `e:\NANO\local_runs\l18mid\artifacts\chain_18_l18_m_up_proj_6144r_b4`
- Size:
  - `3,035,870,536` bytes
- Delta vs baseline:
  - `-177,433,048` bytes

### Full-8192 Sweep After Layer 18 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_19_27\full_sweep_results.json`
- Full-module results:
  - `layer19 = 8192`: fail
- Practical conclusion:
  - after fixing `layer18`, the next stop is `layer19`.

### Layer 19 Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l19_m_up_proj.json`
- Logs:
  - `e:\NANO\local_runs\l19c\combo_results.json`
  - `e:\NANO\local_runs\l19mid\combo_results.json`
  - `e:\NANO\local_runs\l19edge\combo_results.json`
- Verified results:
  - `1024`: pass
  - `1280`: fail
  - `1536`: fail
  - `1792`: pass
  - `1856`: pass
  - `1920`: pass
  - `1984`: fail
  - `2048`: fail
  - `4096`: fail
  - `8192`: fail
- Current exact frontier for `layer19`:
  - best known exact-pass = `1920`
  - first known fail above it = `1984`
- Best layer-19 artifact:
  - `e:\NANO\local_runs\l19edge\artifacts\chain_19_l19_m_up_proj_1920r_b4`
- Size:
  - `3,032,937,176` bytes
- Delta vs baseline:
  - `-180,366,408` bytes

### Full-8192 Sweep After Layer 19 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_20_27\full_sweep_results.json`
- Full-module results:
  - `layer20 = 8192`: pass
  - `layer21 = 8192`: fail
- Current best verified artifact:
  - `e:\NANO\local_runs\full8192_sweep_20_27\artifacts\chain_20_l20_m_up_proj_8192r_b4`
- Size:
  - `3,020,420,200` bytes
- Delta vs baseline:
  - `-192,883,384` bytes
- First new failing full-module candidate:
  - `model.layers.21.mlp.up_proj = 8192`

### Runtime Reality Check
- The search quality is improving, but the current method is still expensive in wall-clock time.
- The main cost is not the ranking step. The main cost is:
  - re-exporting a full artifact for each candidate
  - loading the candidate model from disk for each candidate
  - evaluating all prompts even when the candidate is already dead on exact-match criteria

### Runtime Fixes Applied
- Added compact artifact labels in `nano_search_combo_rows.py` to avoid long Windows paths during deep combo searches.
- Added early-stop exact mismatch handling in:
  - `nano_search_subint8.py`
  - `nano_scan_single_modules.py`
  - `nano_scan_row_mixed.py`
  - `nano_search_combo_rows.py`
- Added `nano_sonar_sweep.py` to standardize a layer-by-layer sweep using:
  - a fixed combo base
  - a shared row-count schedule
  - persistent layer logs
  - automatic ranking generation when missing
- New rule:
  - if `next_token`, `greedy_token`, or `full_greedy` exact agreement is already broken under a `1.0` gate, the evaluator stops immediately instead of spending the remaining prompts.

### Sonar Smoke Validation
- `nano_sonar_sweep.py` was validated end-to-end on:
  - fixed base:
    - `layer0=511`
    - `layer1=479`
    - `layer2=1792`
  - variable module:
    - `layer3`
  - schedule:
    - `64`
- Result:
  - pass
  - artifact size `3,208,956,448` bytes
  - log: `e:\NANO\local_runs\sonar_smoke\sonar_results.json`
- Important reality check:
  - even with the new sonar orchestration, a single candidate still took about `1456.61 sec`
  - this confirms again that the dominant cost is full export plus full reload, not the search policy itself

### Next Runtime Target
- The next real speedup is architectural, not cosmetic:
  - stop rebuilding and reloading the full candidate artifact for every single row-count candidate
  - move toward in-memory candidate patching for the variable module inside a fixed combo base
- That is the step that should collapse search time much more aggressively than any prompt-level tweak.

### Layer 21 Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l21_m_up_proj.json`
- Logs:
  - `e:\NANO\local_runs\l21c\combo_results.json`
  - `e:\NANO\local_runs\l21low\combo_results.json`
  - `e:\NANO\local_runs\l21mid\combo_results.json`
  - `e:\NANO\local_runs\l21edge\combo_results.json`
  - `e:\NANO\local_runs\l21micro\combo_results.json`
- Verified results:
  - `64`: pass
  - `128`: pass
  - `256`: pass
  - `320`: pass
  - `324`: fail
  - `328`: fail
  - `332`: fail
  - `336`: fail
  - `352`: fail
  - `368`: fail
  - `384`: fail
  - `448`: fail
  - `512`: fail
  - `768`: fail
  - `1024`: fail
  - `2048`: fail
  - `4096`: fail
  - `6144`: fail
- Current exact frontier for `layer21`:
  - best known exact-pass = `320`
  - first known fail above it = `324`
- Best layer-21 artifact:
  - `e:\NANO\local_runs\l21mid\artifacts\chain_21_l21_m_up_proj_0320r_b4`
- Size:
  - `3,019,931,632` bytes
- Delta vs baseline:
  - `-193,371,952` bytes

### Full-8192 Sweep After Layer 21 Fix
- Sweep log:
  - `e:\NANO\local_runs\full8192_sweep_22_27\full_sweep_results.json`
- Full-module results:
  - `layer22 = 8192`: pass
  - `layer23 = 8192`: pass
  - `layer24 = 8192`: pass
  - `layer25 = 8192`: pass
  - `layer26 = 8192`: pass
  - `layer27 = 8192`: fail
- Current best verified artifact:
  - `e:\NANO\local_runs\full8192_sweep_22_27\artifacts\chain_26_l26_m_up_proj_8192r_b4`
- Size:
  - `2,957,346,736` bytes
- Delta vs baseline:
  - `-255,956,848` bytes
- Practical conclusion:
  - with `layer21=320` fixed, the chain holds full `8192` on layers `22..26`
  - the current tail stop is `model.layers.27.mlp.up_proj = 8192`

### Layer 27 Search
- Built row ranking:
  - `e:\NANO\local_runs\row_scores_l27_m_up_proj.json`
- Coarse search log:
  - `e:\NANO\local_runs\l27c\combo_results.json`
- Verified coarse results:
  - `1024`: pass
  - `2048`: pass
  - `4096`: pass
  - `6144`: pass
- Mid search log:
  - `e:\NANO\local_runs\l27mid\combo_results.json`
- Verified mid results:
  - `6656`: pass
  - `7168`: pass
  - `7680`: pass
- Edge search log:
  - `e:\NANO\local_runs\l27edge\combo_results.json`
- Verified edge results:
  - `7808`: pass
  - `8064`: pass
- Ultra search log:
  - `e:\NANO\local_runs\l27ultra\combo_results.json`
- Verified ultra results:
  - `8176`: pass
  - `8184`: pass
  - `8188`: pass
  - `8192`: fail
- Current exact frontier for `layer27`:
  - best known exact-pass = `8188`
  - first known fail above it = `8192`
- Best up-proj chain artifact:
  - `e:\NANO\local_runs\l27ultra\artifacts\chain_27_l27_m_up_proj_8188r_b4`
- Size:
  - `2,944,835,872` bytes
- Delta vs baseline:
  - `-268,467,712` bytes
- Practical conclusion:
  - `up_proj` is now closed across all 28 layers under the current exact gate
  - this chain is the fixed base for the next family search rather than an intermediate result

### Search Tooling Upgrade
- `nano_search_combo_rows.py` and `nano_sweep_full_rows.py` now accept fixed-spec logs directly.
- Supported log sources:
  - combo logs like `combo_results.json`
  - full sweep logs like `full_sweep_results.json`
- Practical effect:
  - the best verified chain can now be reused directly as a fixed base for the next family search
  - this removes the need to hand-write 20-plus `--fixed-spec` arguments every time a new sweep starts

### Batched Row-Score Builder
- `nano_build_row_scores.py` now supports batched scoring via:
  - repeated `--module-name`
  - or `--module-template` plus `--layer-start/--layer-end`
  - `--out-dir` for multi-module export
- Practical effect:
  - `gate_proj` rankings for layers `1..27` can now be built in a single HF load and a single forward/backward pass
  - this removes the worst scaling failure in the old one-layer-per-run workflow

### Gate-Proj Search
- Initial gate ranking:
  - `e:\NANO\local_runs\row_scores_l0_m_gate_proj_hf.json`
- Full-8192 sweep with fixed up-proj base:
  - `e:\NANO\local_runs\gate_full8192_sweep_0_27\full_sweep_results.json`
  - `layer0 = 8192`: fail
- `layer0` coarse search:
  - log: `e:\NANO\local_runs\gate_l0c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: fail
    - `6144`: pass
  - current best on schedule:
    - `layer0 gate_proj = 6144`
  - artifact:
    - `e:\NANO\local_runs\gate_l0c\artifacts\chain_28_l0_m_gate_proj_6144r_b4`
  - size:
    - `2,935,448,248` bytes

- Full-8192 sweep from `layer1`:
  - `e:\NANO\local_runs\gate_full8192_sweep_1_27\full_sweep_results.json`
  - `layer1 = 8192`: fail
- `layer1` coarse search:
  - log: `e:\NANO\local_runs\gate_l1c\combo_results.json`
  - results:
    - `64`: fail
    - `128`: fail
    - `256`: fail
    - `512`: fail
    - `1024`: fail
    - `2048`: fail
    - `4096`: fail
    - `6144`: pass
  - current best on schedule:
    - `layer1 gate_proj = 6144`
  - artifact:
    - `e:\NANO\local_runs\gate_l1c\artifacts\chain_29_l1_m_gate_proj_6144r_b4`
  - size:
    - `2,926,060,616` bytes

- Batched gate rankings:
  - `e:\NANO\local_runs\gate_row_scores_hf`
  - built for layers `1..27` in one run using the new batch mode

- Full-8192 sweep from `layer2`:
  - `e:\NANO\local_runs\gate_full8192_sweep_2_27\full_sweep_results.json`
  - `layer2 = 8192`: fail
- `layer2` coarse search:
  - log: `e:\NANO\local_runs\gate_l2c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: fail
    - `6144`: fail
  - current best on schedule:
    - `layer2 gate_proj = 2048`
  - artifact:
    - `e:\NANO\local_runs\gate_l2c\artifacts\chain_30_l2_m_gate_proj_2048r_b4`
  - size:
    - `2,922,931,672` bytes

- Full-8192 sweep from `layer3`:
  - `e:\NANO\local_runs\gate_full8192_sweep_3_27\full_sweep_results.json`
  - `layer3 = 8192`: pass
  - `layer4 = 8192`: fail
- `layer4` coarse search:
  - log: `e:\NANO\local_runs\gate_l4c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: fail
    - `256`: fail
    - `512`: pass
    - `1024`: fail
    - `2048`: fail
    - `4096`: fail
    - `6144`: fail
  - current best on schedule:
    - `layer4 gate_proj = 512`
  - artifact:
    - `e:\NANO\local_runs\gate_l4c\artifacts\chain_32_l4_m_gate_proj_0512r_b4`
  - size:
    - `2,909,632,760` bytes

- Full-8192 sweep from `layer5`:
  - `e:\NANO\local_runs\gate_full8192_sweep_5_27\full_sweep_results.json`
  - `layer5 = 8192`: pass
  - `layer6 = 8192`: pass
  - `layer7 = 8192`: fail
- Current best verified combined artifact:
  - `e:\NANO\local_runs\gate_full8192_sweep_5_27\artifacts\chain_34_l6_m_gate_proj_8192r_b4`
  - this includes:
    - up-proj full best chain
    - `gate_proj`: `layer0=6144`, `layer1=6144`, `layer2=2048`, `layer3=8192`, `layer4=512`, `layer5=8192`, `layer6=8192`
  - size:
    - `2,884,598,808` bytes
  - delta vs baseline:
    - `-328,704,776` bytes
- Practical conclusion:
  - `gate_proj` is not monotonic under the current ranking, unlike the smoother early `up_proj` sweep
  - the first unresolved gate frontier is now `model.layers.7.mlp.gate_proj`

### Gate-Proj Search Continued
- `layer7` coarse search:
  - log: `e:\NANO\local_runs\gate_l7c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: fail
    - `256`: fail
    - `512`: fail
    - `1024`: fail
    - `2048`: fail
    - `4096`: pass
    - `6144`: pass
  - current best on schedule:
    - `layer7 gate_proj = 6144`
  - artifact:
    - `e:\NANO\local_runs\gate_l7c\artifacts\chain_35_l7_m_gate_proj_6144r_b4`
  - size:
    - `2,875,211,176` bytes

- Full-8192 sweep from `layer8`:
  - `e:\NANO\local_runs\gate_full8192_sweep_8_27\full_sweep_results.json`
  - `layer8 = 8192`: fail
- `layer8` coarse search:
  - partial run: `e:\NANO\local_runs\gate_l8c`
  - final single-candidate closeout:
    - `e:\NANO\local_runs\gate_l8_6144\combo_results.json`
  - verified results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: fail
    - `1024`: fail
    - `2048`: fail
    - `4096`: fail
    - `6144`: fail
  - current best on schedule:
    - `layer8 gate_proj = 256`
  - best retained artifact from partial run:
    - `e:\NANO\local_runs\gate_l8c\artifacts\chain_36_l8_m_gate_proj_0256r_b4`
  - size:
    - `2,874,820,408` bytes

- Full-8192 sweep from `layer9`:
  - `e:\NANO\local_runs\gate_full8192_sweep_9_27\full_sweep_results.json`
  - `layer9 = 8192`: pass
  - `layer10 = 8192`: pass
  - `layer11 = 8192`: fail
- `layer11` coarse search:
  - partial run: `e:\NANO\local_runs\gate_l11c`
  - final single-candidate closeout:
    - `e:\NANO\local_runs\gate_l11_6144\combo_results.json`
  - verified results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: pass
    - `6144`: fail
  - current best on schedule:
    - `layer11 gate_proj = 4096`
  - best retained artifact from partial run:
    - `e:\NANO\local_runs\gate_l11c\artifacts\chain_38_l11_m_gate_proj_4096r_b4`
  - size:
    - `2,843,918,944` bytes

- Full-8192 sweep from `layer12`:
  - `e:\NANO\local_runs\gate_full8192_sweep_12_27\full_sweep_results.json`
  - `layer12 = 8192`: pass
  - `layer13 = 8192`: fail
- `layer13` coarse search:
  - log: `e:\NANO\local_runs\gate_l13c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: fail
    - `6144`: fail
  - current best on schedule:
    - `layer13 gate_proj = 2048`
  - artifact:
    - `e:\NANO\local_runs\gate_l13c\artifacts\chain_39_l13_m_gate_proj_2048r_b4`
  - size:
    - `2,834,531,312` bytes

- Full-8192 sweep from `layer14`:
  - `e:\NANO\local_runs\gate_full8192_sweep_14_27\full_sweep_results.json`
  - `layer14 = 8192`: pass
  - `layer15 = 8192`: fail
- Current best verified combined artifact:
  - `e:\NANO\local_runs\gate_full8192_sweep_14_27\artifacts\chain_40_l14_m_gate_proj_8192r_b4`
  - this includes:
    - up-proj full best chain
    - `gate_proj`: `layer0=6144`, `layer1=6144`, `layer2=2048`, `layer3=8192`, `layer4=512`, `layer5=8192`, `layer6=8192`, `layer7=6144`, `layer8=256`, `layer9=8192`, `layer10=8192`, `layer11=4096`, `layer12=8192`, `layer13=2048`, `layer14=8192`
  - size:
    - `2,822,014,336` bytes
  - delta vs baseline:
    - `-391,289,248` bytes
- Practical conclusion:
  - after `layer8`, the tail becomes mixed again: some full `8192` passes reappear (`9`, `10`, `12`, `14`), separated by sharp local frontiers (`11`, `13`, `15`)
  - the first unresolved gate frontier is now `model.layers.15.mlp.gate_proj`

### Gate-Proj Search Continued Again
- `layer15` coarse search:
  - log: `e:\NANO\local_runs\gate_l15c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: pass
    - `6144`: pass
  - current best on schedule:
    - `layer15 gate_proj = 6144`
  - artifact:
    - `e:\NANO\local_runs\gate_l15c\artifacts\chain_41_l15_m_gate_proj_6144r_b4`
  - size:
    - `2,812,626,712` bytes

- Full-8192 sweep from `layer16`:
  - `e:\NANO\local_runs\gate_full8192_sweep_16_27\full_sweep_results.json`
  - `layer16 = 8192`: pass
  - `layer17 = 8192`: fail
- `layer17` coarse search:
  - log: `e:\NANO\local_runs\gate_l17c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: pass
    - `256`: pass
    - `512`: pass
    - `1024`: pass
    - `2048`: pass
    - `4096`: pass
    - `6144`: fail
  - current best on schedule:
    - `layer17 gate_proj = 4096`
  - artifact:
    - `e:\NANO\local_runs\gate_l17c\artifacts\chain_43_l17_m_gate_proj_4096r_b4`
  - size:
    - `2,793,851,448` bytes

- Full-8192 sweep from `layer18`:
  - `e:\NANO\local_runs\gate_full8192_sweep_18_27\full_sweep_results.json`
  - `layer18 = 8192`: pass
  - `layer19 = 8192`: fail
- `layer19` coarse search:
  - log: `e:\NANO\local_runs\gate_l19c\combo_results.json`
  - results:
    - `64`: pass
    - `128`: fail
    - `256`: fail
    - `512`: pass
    - `1024`: pass
    - `2048`: fail
    - `4096`: fail
    - `6144`: fail
  - current best on schedule:
    - `layer19 gate_proj = 1024`
  - artifact:
    - `e:\NANO\local_runs\gate_l19c\artifacts\chain_45_l19_m_gate_proj_1024r_b4`
  - size:
    - `2,779,770,208` bytes

- Full-8192 sweep from `layer20`:
  - `e:\NANO\local_runs\gate_full8192_sweep_20_27\full_sweep_results.json`
  - `layer20 = 8192`: fail
  - the fail is immediate and early-stopped on the first prompt:
    - `next_token_agreement = 0.0`
    - `greedy_token_agreement = 0.75`
    - `full_greedy_match_rate = 0.0`

- Current best verified combined artifact:
  - `e:\NANO\local_runs\gate_l19c\artifacts\chain_45_l19_m_gate_proj_1024r_b4`
  - this includes:
    - up-proj full best chain
    - `gate_proj`: `layer0=6144`, `layer1=6144`, `layer2=2048`, `layer3=8192`, `layer4=512`, `layer5=8192`, `layer6=8192`, `layer7=6144`, `layer8=256`, `layer9=8192`, `layer10=8192`, `layer11=4096`, `layer12=8192`, `layer13=2048`, `layer14=8192`, `layer15=6144`, `layer16=8192`, `layer17=4096`, `layer18=8192`, `layer19=1024`
  - size:
    - `2,779,770,208` bytes
  - delta vs baseline:
    - `-433,533,376` bytes
- Practical conclusion:
  - the tail remains mixed rather than monotone: permissive layers (`16`, `18`) are interleaved with narrow frontiers (`17`, `19`, `20`)
  - the first unresolved gate frontier is now `model.layers.20.mlp.gate_proj`

### Fast Tail Policy Update (`min_row_count = 512`)
- Decision applied:
  - stop sub-512 exploration for tail layers (`21..27`) to prevent week-long search cycles
  - only test `row_count = 512`; if it fails, keep layer uncompressed (`0 rows`)
- Tooling update:
  - patched `e:\NANO\nano_sonar_sweep.py` to support `--fixed-spec-log`
  - patched `e:\NANO\nano_sonar_sweep.py` to use short `chain_XX_...` labels (fixes Windows path-length failures)

- 512-only tail checks from `gate_l20c` base:
  - `layer21 @512`: fail (`next=0.0`, `greedy=0.75`, `full=0.0`)
  - `layer22 @512`: fail (`next=0.0`, `greedy=0.75`, `full=0.0`)
  - `layer23 @512`: fail (`next=0.0`, `greedy=0.75`, `full=0.0`)
  - `layer24 @512`: fail (`next=0.0`, `greedy=0.75`, `full=0.0`)
  - `layer25 @512`: pass
    - log: `e:\NANO\local_runs\gate_l25_512\combo_results.json`
    - artifact: `e:\NANO\local_runs\gate_l25_512\artifacts\chain_47_l25_m_gate_proj_0512r_b4`
    - size: `2,778,206,336` bytes
  - `layer26 @512`: pass
    - log: `e:\NANO\local_runs\gate_l26_512\combo_results.json`
    - artifact: `e:\NANO\local_runs\gate_l26_512\artifacts\chain_48_l26_m_gate_proj_0512r_b4`
    - size: `2,777,424,408` bytes
  - `layer27 @512`: pass
    - log: `e:\NANO\local_runs\gate_l27_512\combo_results.json`
    - artifact: `e:\NANO\local_runs\gate_l27_512\artifacts\chain_49_l27_m_gate_proj_0512r_b4`
    - size: `2,776,642,472` bytes

- New current best (fast policy):
  - artifact: `e:\NANO\local_runs\gate_l27_512\artifacts\chain_49_l27_m_gate_proj_0512r_b4`
  - exact gate preserved: `next=1.0`, `greedy=1.0`, `full=1.0`
  - delta vs int8 baseline (`3,213,303,584`): `-436,661,112` bytes
  - improvement vs previous best (`2,779,770,208`): `-3,127,736` bytes

### Ladder Policy Update (`8192 -> 4096 -> 2048 -> 1024 -> 512`)
- User policy applied:
  - no sub-512 search
  - for each layer, test descending ladder and keep the first exact-pass
- Tooling update:
  - patched `e:\NANO\nano_search_combo_rows.py` with `--stop-on-first-pass`
  - ladder logs stored in `e:\NANO\local_runs\gate_l21_ladder` ... `gate_l27_ladder`

- Ladder outcomes from `gate_l20c` base:
  - `layer21`: no pass in `{8192,4096,2048,1024,512}` -> keep uncompressed
    - log: `e:\NANO\local_runs\gate_l21_ladder\combo_results.json`
  - `layer22`: no pass in `{8192,4096,2048,1024,512}` -> keep uncompressed
    - log: `e:\NANO\local_runs\gate_l22_ladder\combo_results.json`
  - `layer23`: no pass in `{8192,4096,2048,1024,512}` -> keep uncompressed
    - log: `e:\NANO\local_runs\gate_l23_ladder\combo_results.json`
  - `layer24`: no pass in `{8192,4096,2048,1024,512}` -> keep uncompressed
    - log: `e:\NANO\local_runs\gate_l24_ladder\combo_results.json`
  - `layer25`: first pass at `4096`
    - log: `e:\NANO\local_runs\gate_l25_ladder\combo_results.json`
  - `layer26`: first pass at `4096`
    - log: `e:\NANO\local_runs\gate_l26_ladder\combo_results.json`
  - `layer27`: first pass at `1024`
    - log: `e:\NANO\local_runs\gate_l27_ladder\combo_results.json`

- New current best (ladder policy):
  - artifact: `e:\NANO\local_runs\gate_l27_ladder\artifacts\chain_49_l27_m_gate_proj_1024r_b4`
  - size: `2,764,907,440` bytes
  - exact gate: `next=1.0`, `greedy=1.0`, `full=1.0`
  - delta vs int8 baseline (`3,213,303,584`): `-448,396,144` bytes
  - improvement vs fast-512 best (`2,776,642,472`): `-11,735,032` bytes

### Pipeline Normalization + Storage Cleanup
- Added one-pass ping controls to avoid micro-search:
  - `e:\NANO\nano_search_combo_rows.py`: `--stop-on-first-pass`
  - `e:\NANO\nano_sonar_sweep.py`:
    - `--selection-mode first-pass|smallest` (default `first-pass`)
    - `--require-size-improve` optional strict gate
    - support for `--fixed-spec-log` already integrated
- Enabled 6-bit path across runtime/export:
  - `e:\NANO\nano_export_from_map_v4.py`:
    - added 6-bit packing for full tensors and row-mixed tensors
    - `embed-bits` now accepts `8,6,4,2,0`
    - row-mixed `low_bits` now accepts `6,4,2`
  - `e:\NANO\nano_inference_direct.py`:
    - added 6-bit unpack/dequant support
  - CLI choices updated for 6-bit in:
    - `e:\NANO\nano_search_combo_rows.py`
    - `e:\NANO\nano_sonar_sweep.py`
    - `e:\NANO\nano_sweep_full_rows.py`
    - `e:\NANO\nano_scan_row_mixed.py`
    - `e:\NANO\nano_scan_single_modules.py`
    - `e:\NANO\nano_search_subint8.py`

- Local runs cleanup (space recovery):
  - pruned obsolete intermediate folders, kept only:
    - baselines (`3B` + `7B`)
    - `gate_l20c` base
    - `gate_l21_ladder` .. `gate_l27_ladder`
    - `gate_row_scores_hf`, `research_logs`
  - removed payload: `249,440,839,737` bytes

### One-Command Ping Pipeline (8/6/4/2/0)
- New unified entrypoint:
  - `e:\NANO\nano_ping_multibit_sweep.py`
- Goal:
  - run all selected layers in one command
  - per-layer lock with fixed ladders:
    - bits ladder (default): `0 -> 2 -> 4 -> 6` then fallback `8` if no pass
    - row ladder (default): `8192 -> 4096 -> 2048 -> 1024 -> 512`
  - lock first exact pass; no sub-512 search
  - keep storage lean: temp artifacts removed; final artifact exported once

- Smoke validation:
  - command run on `layer25 gate_proj` only
  - out: `e:\NANO\local_runs\PING_MB_SMOKE_L25`
  - result:
    - `0-bit`: fail
    - `2-bit @ 8192`: fail
    - `2-bit @ 4096`: pass (first pass lock)
  - final artifact:
    - `e:\NANO\local_runs\PING_MB_SMOKE_L25\final_artifact`
    - size: `2,769,584,264` bytes
    - exact gate preserved (`next=1.0`, `greedy=1.0`, `full=1.0`)

### CPU Timing Probe (Multibit Ping)
- Date: `2026-03-28`
- Goal:
  - measure real wall-time and size impact of one-command ping on CPU with fixed ladders.
- Run:
  - script: `e:\NANO\nano_ping_multibit_sweep.py`
  - output root: `e:\NANO\local_runs\PING_MB_L21_TRY`
  - variable module: `model.layers.21.mlp.gate_proj`
  - seed: `e:\NANO\local_runs\gate_l27_ladder\combo_results.json`
  - ladders:
    - bits: `0 -> 2 -> 4 -> 6`
    - rows: `8192 -> 4096 -> 2048 -> 1024 -> 512`
  - strict policy: `--require-size-improve`
- Outcome:
  - tested candidates:
    - `0-bit`: FAIL
    - `2-bit@8192`: FAIL
    - `2-bit@4096`: FAIL
    - `2-bit@2048`: PASS (locked)
  - accepted update:
    - `model.layers.21.mlp.gate_proj -> 2-bit @ 2048 rows`
  - final artifact:
    - `e:\NANO\local_runs\PING_MB_L21_TRY\final_artifact`
    - size: `2,760,205,640` bytes
  - elapsed:
    - `4,563.64 sec` (~`76.1 min`)
- Delta vs previous best (`e:\NANO\local_runs\gate_l27_ladder\artifacts\chain_49_l27_m_gate_proj_1024r_b4`, `2,764,907,440`):
  - `-4,701,800` bytes (smaller)

### Full INT8 Baseline Sweep (All Gate Layers) - In Progress
- Date: `2026-03-28`
- Objective:
  - execute a fully fresh pipeline from pure INT8 baseline over all gate layers (`0..27`) with fixed ladders.
- Run id:
  - `e:\NANO\local_runs\PING_MB_FULL_GATES_INT8_R2`
- Command profile:
  - script: `e:\NANO\nano_ping_multibit_sweep.py`
  - source: `e:\testmob\out\NB8_NF4_3B_COMPARE\merged_hf`
  - base map: `e:\NANO\local_runs\canonical_map_3b_safe88_v3.json`
  - reference: `e:\NANO\local_runs\Llama-3.2-3B-NANO-Canonical-v3-Safe88-Embed8` (`reference-kind=nano`)
  - module range: `mlp.gate_proj`, layers `0..27`
  - ladders:
    - bits: `0 -> 2 -> 4 -> 6`
    - rows: `8192 -> 4096 -> 2048 -> 1024 -> 512`
  - strict gate: `--require-size-improve`
  - ranking dir: `e:\NANO\local_runs\gate_row_scores_hf`
- Live status snapshot:
  - first candidate finished:
    - `layer0`, `0-bit`: `FAIL`, size `3,188,137,648`
  - second candidate finished:
    - `layer0`, `2-bit@8192`: `FAIL`, size `3,194,495,144`
  - run continues automatically with next ladder step.
- Notes:
  - fixed ranking naming issue for layer0 by adding:
    - `e:\NANO\local_runs\gate_row_scores_hf\row_scores_l0_m_gate_proj.json`
  - previous attempt `R1` aborted on missing layer0 ranking schema; `R2` is the valid full run.

### RAM-First Candidate Evaluation (No Per-Candidate Export)
- Date: `2026-03-28`
- Reason:
  - avoid rewriting full `model.safetensors` for every candidate.
- Implementation:
  - patched `e:\NANO\nano_ping_multibit_sweep.py` with:
    - `--candidate-eval-mode {ram,disk}` (default: `ram`)
    - in RAM mode:
      - candidate modules are patched in-memory on a working NANO model
      - per-candidate quality evaluation runs without artifact export
      - only maps are written during sweep
      - only final artifact is exported at the end
  - added source tensor cache + runtime module builders for:
    - `0-bit` (zeroed module)
    - `row_mixed` low bits (`2/4/6`)
- Validation smoke:
  - run: `e:\NANO\local_runs\PING_RAM_SMOKE_L0`
  - module: `model.layers.0.mlp.gate_proj`
  - ladder tested: `0`, `2@8192`
  - result:
    - both fail, layer kept at 8-bit
  - key verification:
    - no `scratch/` candidate artifacts created
    - only `maps/` + `final_artifact/` produced

### Full INT8 Baseline Sweep (All Gate Layers) - RAM Mode (In Progress)
- Run id: `e:\NANO\local_runs\PING_RAM_FULL_GATES_INT8_R1`
- Status snapshot:
  - `layer0`: `0-bit` FAIL
  - `layer0`: `2@8192` FAIL
  - `layer0`: `2@4096` FAIL
- Mode:
  - RAM-only candidate evaluation (`size=n/a` per attempt)
  - final exact size computed at final export.
