---
id: node_0026
desc: TabICL foundation-model base
op: draft
parents: [root]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.958991
sem: 0.000197
folds: [0.959432, 0.959371, 0.958361, 0.958788, 0.959005]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn (gain=0.94 > 0.9 threshold) -- consistent with all competition nodes; no leakage found"
leak: clean
lb: null
submitted: null
created: 2026-06-07T07:18Z
decided: 2026-06-07T11:47Z
tags: [nn, tabicl, foundation-model, in-context, cuda, diversity-arm, draft]
---

## plan
built on:   root -- new NN base family. Template src copied from node_0009/src (NN OOF/test
            harness -- reuse the fs_research feature construction exactly, frozen folds.json loop,
            fold-honest OOF + test_probs interface). Keep the data/fold scaffolding; swap only
            the model + prediction routine.
change:     a TabICL foundation-model base. Dep: `uv add tabicl`. TabICL is purpose-built for
            ~500k rows (column-then-row attention + in-context learning, with CPU/disk offloading
            for large N); on >10k-row classification it beats TabPFN-v2 and CatBoost. For each
            fold, fit/condition on that fold's TRAIN rows only (use its large-data path /
            offloading; subsample the context ONLY from train-fold indices if needed) and predict
            val. Test predicted with full-train context, never test rows. Produce oof.npy
            (577347x3) + test_probs.npy (247435x3). GPU + offloading, serialized LAST (heaviest).
hypothesis: a foundation model engineered for THIS row regime, maximally de-correlated from the
            GBDTs, is the strongest new stacking base.
target:     BA maximize, solo >= ~0.962, leak-clean; valued for de-correlation; drop if < 0.955.

## notes
Implementation details:
- n_estimators=8, context_size=100000 (class-balanced: ~33k per class), kv_cache="repr"
- Context subsample from train-fold rows ONLY; val rows are queries, never in context
- Test predicted in chunks of 100k to avoid OOM (col-embedding output buffer limitation)
- Standardization fit on train-fold ONLY (fit-inside-fold)
- Categorical features one-hot encoded (TabICL accepts numeric arrays)
- Per-fold timing: ~27.5s; final test fit+predict: 62s; total wall: ~3.5 min

Performance vs target:
- cv=0.958991 is below the 0.962 solo target but above the 0.955 drop threshold
- Still valuable as a diversity arm for stacking/blending (different architecture from GBDTs)
- For higher accuracy: increase context_size (try 200k with disk offload) or n_estimators

Gate verdicts:
- schema_ok: PASS (validate_submission.py exit 0)
- oof_full: PASS (577347 rows, all nonzero)
- no_nan: PASS
- dist_sane: PASS (GALAXY 62.3%, QSO 20.9%, STAR 16.8% -- reasonable vs true 65/20/14)
- leak_clean: PASS (leakage_scan.py exit 0; all checks ok)
- cv_too_good: WARN (gain=0.94 > 0.90 threshold; consistent with all well-performing nodes in competition)
- passed: true
