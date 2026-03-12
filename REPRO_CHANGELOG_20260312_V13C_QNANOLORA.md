# Repro Change Log - V13C + QNanoLoRA (2026-03-12)

Questo log fissa in modo deterministico le modifiche eseguite durante il run Kaggle per garantire ripetibilita'.

## Contesto run

- Base model: `unsloth/Llama-3.2-3B`
- Adapter candidato: `/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_std`
- Summary training adapter:
  - `best_val_step = 160`
  - `best_val_loss = 27.72745966911316`
  - `std_size_bytes = 157073597`

## Modifiche applicate in corsa (ordine reale)

1. **Cambio sorgente cella**
- Errore iniziale: file markdown non trovato in Kaggle input.
- Fix: esecuzione della cella in forma inline (nessuna dipendenza dal file locale `KAGGLE_V13C_QNANOLORA_MERGE_CLEAN_CELL.md`).

2. **Risoluzione artifact V13**
- Errore iniziale: `grouped_mlp_hidden_anchor_v13c_best.pt` non trovato.
- Fix: fallback deterministico su `phase2_standalone_tail_v13_hard_guard_best.pt`.

3. **Normalizzazione checkpoint V13**
- Errore iniziale: `v13['state']` vuoto.
- Fix: `normalize_v13(...)` aggiornata con:
  - fallback da `state` -> `tail_state`;
  - fallback metadati: `best_group_alpha`, `best_step`, `best_cached_kl`;
  - mapping deterministico `wrap_* -> {layer}_{proj}` usando `target_layers` e ordine proiezioni.

4. **Controlli hard prima dell'eval**
- Aggiunti check espliciti:
  - stampa chiavi v13,
  - `len(v13['state']) > 0` obbligatorio,
  - stampa `best_group_alpha`.

5. **Merge adapter**
- Merge LoRA con key parser robusto:
  - pattern supportato: con/senza suffisso `.default`.
- Target modules letti da `adapter_config.json`.
- Risultato run: `merged=196`, `skipped=0`, `scaling=2.0`.

6. **Confronto reale nello stesso run**
- Eseguiti in sequenza:
  - `BASELINE_V13C`
  - `V13C_PLUS_QNANOLORA`
- Stesse prompt, stessa decode policy, stesso ambiente CUDA.

## Risultato deterministico del confronto

- `baseline uniq = 0.8902`
- `merged uniq = 0.8639`
- `gap uniq = 0.0263`
- `baseline semantic_fail_count = 0`
- `merged semantic_fail_count = 0`
- `baseline short = 0`
- `merged short = 0`

Decisione:
- `pass_hard_gates = True`
- `close_enough_vs_baseline = True`
- `better_or_equal_semantic = True`

## Artifact canonici generati

- Eval JSON:
  - `/kaggle/working/v13c_qnanolora_merge_clean_eval.json`
- Bundle produzione scaricato:
  - `production_candidate_v13c_qnanolora_<timestamp>.zip`
- Contenuti minimi bundle:
  - `adapter/blend_teacher_r23_size142_v2_kl_topk_std/*`
  - `reports/v13c_qnanolora_merge_clean_eval.json`
  - `reports/blend_teacher_r23_size142_v2_kl_topk_summary.json`
  - `MANIFEST.json`

## Config/gate congelati per replay

- Prompt set: 4 prompt core closeout
- Decode: `max_new_tokens=64`, `do_sample=False`, `repetition_penalty=1.10`, `no_repeat_ngram_size=3`
- Gate promozione:
  - `semantic_fail_count == 0`
  - `short == 0`
  - `gap_uniq_vs_baseline <= 0.03`

## Note operative

- Warning transformers su `max_length` e `max_new_tokens` e' atteso; non invalida il run.
- Questo replay dipende dalla presenza di `phase2_standalone_tail_v13_hard_guard_best.pt` con `tail_state` valorizzato.
