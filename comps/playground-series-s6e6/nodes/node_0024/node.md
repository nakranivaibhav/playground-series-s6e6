---
id: node_0024
desc: RealMLP-HPO strong recipe
op: improve
parents: [node_0021]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.949229
sem: 0.000265
folds: [0.949667, 0.950052, 0.948823, 0.948733, 0.948870]
baseline_cv: 0.333333
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false, leak_clean: false, cv_too_good: false, passed: false}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-07T07:18Z
decided: 2026-06-07T09:28Z
tags: [nn, realmlp, pytabkit, hpo, cuda, diversity-arm, improve]
---

## plan
built on:   node_0021 (RealMLP base, pytabkit). Keep the fs_research load, frozen folds.json
            loop, and fold-honest OOF + test_probs interface byte-identical. Template src copied
            from node_0021/src.
change:     replace the bare `RealMLP_TD_Classifier` with the STRONG recipe —
            `RealMLP_HPO_Classifier(hpo_space_name='tabarena-new', device='gpu')`, raise n_epochs
            well above the 256 default (577k rows under-fit at default), wider/deeper hidden
            sizes (e.g. [512]*3), PLR numerical embeddings ON. Dep: `uv add 'pytabkit[models]'`.
            Standardization/preprocessing fit INSIDE each train fold only (fit_in_fold), never on
            full train or test. Fold-honest OOF over folds.json → oof.npy (577347×3) +
            test_probs.npy (247435×3). GPU, serialized.
hypothesis: HPO + more epochs + wider/deeper net lifts RealMLP from 0.950 into the GBDT band
            (≥~0.963) so it becomes an additive, de-correlated stack column instead of dragging.
target:     BA maximize, solo ≥ ~0.962, leak-clean; drop if solo < 0.958.
