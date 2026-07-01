---
id: node_0021
desc: RealMLP base (pytabkit)
op: draft
parents: [root]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.950098
sem: 0.000581
folds: [0.948835, 0.950815, 0.948642, 0.950591, 0.951607]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn (0.950 vs baseline 0.333) — all other nodes similarly flagged; eyeball before first submit"
leak: clean
lb: null
submitted: null
created: 2026-06-07T06:06Z
decided: 2026-06-07T07:02Z
tags: [nn, realmlp, pytabkit, cuda, diversity-arm, draft]
---

## plan
built on:   root — new NN base family. Template src copied from node_0009 (TabM harness:
            fs_research load, frozen folds.json loop, fold-honest OOF + test_probs interface).
            Keep the data-loading/fold scaffolding byte-identical; swap only the model.
change:     add `pytabkit` (`uv add pytabkit`); train `RealMLP_TD_Classifier` as a new base
            model. Standardization/preprocessing instantiated FRESH inside each fold's train
            split only (fit_in_fold) — never on full train or test. Produce fold-honest OOF
            over folds.json → oof.npy + test_probs.npy with the same interface as node_0009.
            GPU (RTX 5090), runs serialized.
hypothesis: RealMLP = a strongly regularized MLP that reaches GBDT-level accuracy solo while
            being de-correlated from TabM and the trees → a fresh, additive stack column.
target:     BA maximize, solo ≥ ~0.962, leak-clean; drop if solo < 0.960.
