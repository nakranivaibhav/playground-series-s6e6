---
id: node_0005
desc: LightGBM heavy-regularized
op: improve
parents: [node_0001]
family: gbdt
uses_data: [fs_colors]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.963698
sem: 0.000237
folds: [0.964543, 0.963868, 0.963207, 0.963423, 0.963448]
baseline_cv: 0.964569
shuffled_cv: 0.33157
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T12:40Z
decided: 2026-06-05T12:50Z
tags: [lightgbm, tuning, regressed]
---

## plan
built on:   node_0001 — same features, different hyperparameters.
change:     heavier regularization + slower learning: min_child_samples=200 (vs default 20),
            learning_rate=0.03, n_estimators=3000 + early_stopping(100), num_leaves=127,
            subsample/colsample=0.8, reg_lambda=1. Saves test_probs.npy.
hypothesis: research said default min_child_samples=20 is under-regularized at 577k rows.
target:     beat node_0001 0.964569.

## notes
HYPOTHESIS REFUTED: cv 0.963698 = -3.7σ, REGRESSED. The signal is so strong that heavy
regularization over-smooths the subtle QSO↔GALAXY boundary; node_0001's light defaults are
near-optimal. Lesson: the lever here is features, not tuning (→ node_0006 confirmed).
