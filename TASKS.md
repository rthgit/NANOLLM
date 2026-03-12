# TASKS

## Fase 1 - fro-core (completata)

- [x] T1.1 Estrarre FRO in modulo stabile (`fro-core/fro_core/optimizer.py`)
- [x] T1.2 Bloccare API minima (`from fro_core import FRO`)
- [x] T1.3 Aggiungere test smoke import (`fro-core/tests/test_optimizer_api.py`)
- [x] T1.4 Definire metadata package (`fro-core/pyproject.toml`)
- [x] T1.5 Congelare licenza proprietaria (`fro-core/LICENSE.md`)
- [x] T1.6 Conferma utente: OK per passare a Fase 2

## Fase 2 - nano-distill-pipeline (completata)

- [x] T2.1 Definire repo pipeline (`nano-distill-pipeline/README.md`)
- [x] T2.2 Fissare config canonica 3B (`nano-distill-pipeline/configs/blend_pilot_safe.yaml`)
- [x] T2.3 Definire schema manifest run (`nano-distill-pipeline/manifests/run_manifest.template.json`)
- [x] T2.4 Collegare runbook preflight (`nano-distill-pipeline/runbooks/preflight.md`)
- [x] T2.5 Congelare policy eval `concise_guard` nel profilo canonico
- [x] T2.6 Conferma utente: OK per passare a Fase 3

## Fase 3 - nano-eval-gates (completata)

- [x] T3.1 Definire regole gate (`nano-eval-gates/README.md`)
- [x] T3.2 Fissare soglie qualità (`nano-eval-gates/configs/thresholds.yaml`)
- [x] T3.3 Congelare prompt core (`nano-eval-gates/prompts/core_prompts.txt`)
- [x] T3.4 Definire area report immutabili (`nano-eval-gates/reports/README.md`)
- [x] T3.5 Conferma utente: OK per passare a Fase 4

## Teacher-Free 3B - execution track

- [x] TF1.1 Phase 1C qkvo conservative completata (`blend_pilot_safe_single-gpu_20260309_122418`)
- [x] TF1.2 Winner selezionato (`final.pt`, step 100, alpha 0.194)
- [x] TF1.3 Gate minimi passati (`han=0 loop=0 short=0`)
- [x] TF2.1 Phase 2A avviata (`qkvo + gate_proj`)
- [x] TF2.2 Phase 2B completata (`qkvo + gate + up`)
- [x] TF2.3 Phase 2C completata (`qkvo + gate + up + down`)
- [x] TF2.4 Prod candidate promosso (`blend_phase2c_single-gpu_20260309_161152/best.pt`)
- [x] TF3.1 Packaging produzione (manifest + hash + export canonical)
- [x] TF3.2a Repo split locale inizializzato (`git init` + `.gitignore` su 4 repo)
- [x] TF3.2b Push remoto dei 4 repo (monorepo `NANOLLM`)


## Standalone Llama 3B - tail cycle closure (completata)

- [x] ST3B.1 Eseguito ciclo `tail_v1 -> tail_v16`
- [x] ST3B.2 Confronto finale contro baseline ufficiale completato
- [x] ST3B.3 Congelato best checkpoint di ricerca standalone: `tail_v13_hard_guard` (`best_step=60`)
- [x] ST3B.4 Reiezione formale `tail_v14_ce_anchor`, `tail_v15_selector_guard` e `tail_v16_final_gate`
- [x] ST3B.5 Chiusura micro-iterazioni stesso regime (`v17+` bloccato)


- [x] ST3B.6 Closeout finale e pacchetto handoff (phase2_standalone_llama3b_final_closeout_20260311.zip)

- [x] ST3B.7 Paper NanoLLM riscritto (`PAPER_NANOLLM.md`)
- [x] ST3B.8 Repository cleanup + archivio legacy (`archive/2026-03-11_cleanup/`)
- [x] ST3B.9 Definito spec QNanoLoRA budget-driven per match size Phase2 (`NANOLORA_BUDGET_SPEC.md`)
- [x] ST3B.10 Creata cell Kaggle rank-auto per QNanoLoRA (`KAGGLE_QNANOLORA_SIZE_MATCH_PHASE2_CELL.md`)
- [x] ST3B.11 Aggiunta cell calibrata QNanoLoRA da run reale (`KAGGLE_QNANOLORA_SIZE_MATCH_CALIBRATED_CELL.md`)
- [x] ST3B.12 Creata cell finale 3B QNanoLoRA sweep fp16 (`KAGGLE_QNANOLORA_FINAL_3B_SWEEP_CELL.md`)
- [x] SCALE.1 Definita policy quantizzazione multi-taglia (`NANOLLM_QUANT_POLICY.md`)
- [x] SCALE.2 Creato runbook scaling 7B->70B (`NANOLLM_SCALE_LADDER_RUNBOOK.md`)
- [x] SCALE.3 Policy aggiornata: INT8 solo per taglie grandi (30B/70B)
- [x] ST3B.13 Decisione 3B: rank fisso 32 per QNanoLoRA
- [x] ST3B.14 Decisione adapter 3B: rank 32 con export fp16 (`NANOLLM_3B_ADAPTER_DECISION.md`)
- [x] ST3B.15 Creata cella training QNanoLoRA stile Fase 1 (teacher base) (`KAGGLE_QNANOLORA_3B_TEACHER_PHASE1_STYLE_CELL.md`)
- [x] ST3B.16 Allineata cella closeout per priorita' adapter teacher-trained (`KAGGLE_FINAL_3B_CLOSEOUT_WITH_ADAPTER_CELL.md`)
- [x] ST3B.17 Preparato runbook operativo domani per chiusura 3B V2 (RUNBOOK_TOMORROW_3B_V2.md)

- [x] ST3B.18 Validazione `V13C + QNanoLoRA merged` passata con gate hard (`gap uniq <= 0.03`, `semantic_fail_count=0`, `short=0`)
- [x] ST3B.19 Bundle production candidate creato e scaricato (`production_candidate_v13c_qnanolora_<timestamp>.zip`)
- [x] ST3B.20 Changelog ripetibilita' in-run salvato (`REPRO_CHANGELOG_20260312_V13C_QNANOLORA.md`)
- [x] ST3B.21 Integrato benchmark esteso 20-prompt nel paper con tradeoff size/quality quantificato
- [x] ST3B.22 Preparata cella benchmark unificata metodi compressione (`KAGGLE_COMPRESSION_METHODS_BENCHMARK_CELL.md`)
- [x] ST3B.23 Eseguito benchmark unificato (RAW/INT8/NF4/QNanoLoRA + opzionali AWQ/GPTQ) e consolidato report finale

- [x] ST3B.24 Integrato benchmark multi-metodo nel paper (sezione unificata con ranking per metrica)
