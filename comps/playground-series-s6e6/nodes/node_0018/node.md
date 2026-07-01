---
id: node_0018
desc: lgbm + redshift-conditional target encoding
op: draft
parents: [root]
uses_data: [fs_colors, fs_research, fs_tgt_enc]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964839
sem: 0.000035
folds: [0.964952, 0.964783, 0.964880, 0.964755, 0.964826]
baseline_cv: 0.965004
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CV regresses vs parent node_0006 (0.964839 < 0.965004); 3/5 folds negative delta; do not promote"
leak: clean
lb: null
submitted: null
created: 2026-06-06T12:18Z
decided: 2026-06-06T13:21Z
tags: [gbdt, lightgbm, target-encoding, redshift]
---

## plan
built on:   root (a fresh GBDT arm). Reuses node_0006's exact LightGBM params and
            fs_research as the base feature matrix; adds a new fold-local feature-set.
change:     New feature-set fs_tgt_enc (leak-safety fit_in_fold). Inside each train
            fold ONLY: smoothed (m=100) target-encoded posteriors P(class | spectral_type)
            and P(class | galaxy_population), AND a redshift-band conditional encoding —
            bin redshift into ~10 quantile bands with edges computed from the train
            fold, then target-encode P(class | band) for the 3 classes. This yields 8
            columns (3 + 3 redshift-band + 2 categorical, per the 3-class posteriors;
            build exactly the smoothed P(class|·) posteriors described). Append them to
            fs_research and train node_0006's exact LightGBM params.
hypothesis: redshift is the dominant physical separator; explicit smoothed
            P(class | redshift-band) plus categorical target-encoding gives the tree a
            calibrated class signal at the hard QSO↔GALAXY boundary.
target:     balanced accuracy (maximize) · a strong de-correlated GBDT arm beats
            node_0006 0.965004 solo and lifts the 4-arm blend beyond sem.
