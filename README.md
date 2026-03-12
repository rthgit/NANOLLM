# NANOLLM Repository (Stato Finale)

Stato: **closeout standalone 3B completato** (2026-03-11).

## Documenti canonici

- `PAPER_NANOLLM.md`
- `PHASE2_CANONICAL_LIVE_STATE.md`
- `PHASE2_STANDALONE_PROGRAM.md`
- `RUN_DECISIONS.md`
- `PHASE2_STANDALONE_FINAL_CLOSEOUT.md`
- `TASKS.md`

## Celle finali 3B

- `KAGGLE_QNANOLORA_3B_TEACHER_PHASE1_STYLE_CELL.md` (training teacher -> QNanoLoRA r32)
- `KAGGLE_FINAL_3B_CLOSEOUT_WITH_ADAPTER_CELL.md` (benchmark/decisione finale con tail v13)

## Artefatti finali mantenuti in root

- `phase2_standalone_tail_v13_hard_guard_best.pt`
- `phase2_standalone_tail_v13_hard_guard_summary.json`
- `phase2_standalone_llama3b_final_closeout_20260311.zip`
- `KAGGLE_PHASE2_STANDALONE_TAIL_V13_HARD_GUARD_CELL.md`

## Regola operativa

- Baseline ufficiale resta non-standalone (`nanollm_best_composed_bundle_v2.zip`).
- Ciclo standalone `tail_v1 -> tail_v16` chiuso.
- Nessuna promozione standalone in questo regime.

## Archivio legacy

File e versioni storiche/obsolete sono stati spostati in:
- `archive/2026-03-11_cleanup/`

Con sottocartelle per `docs`, `cells`, `scripts`, `artifacts`, `caches` e manifest.

## Runbook operativo

- RUNBOOK_TOMORROW_3B_V2.md (sequenza pronta per chiusura 3B V2)
