---
id: node_0019
desc: bagged multi-seed TabM arm
op: improve
parents: [node_0009]
uses_data: [fs_colors, fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964466
sem: 0.000328
folds: [0.965267, 0.963264, 0.964655, 0.964507, 0.964640]
baseline_cv: 0.964215
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: fold-2 regressed -0.000384 vs parent; 4/5 folds positive, mean delta +0.000252
leak: clean
lb: null
submitted: null
created: 2026-06-06T12:18Z
decided: 2026-06-06T13:21Z
tags: [nn, tabm, bagging, multi-seed]
---

## plan
built on:   node_0009 (TabM on CUDA, `tabm` library, k=32, PLR embeddings,
            fit-in-fold standardize/bins). The entire TabM config stays byte-identical.
change:     ONE change — train 3 independent TabM models with different init / data-order
            seeds per fold, and average their softmax into one OOF + one test_probs.
hypothesis: TabM is the highest-variance arm (sem 0.000374, fold-2 dip); averaging 3
            seeds shrinks variance, raising solo CV slightly and stabilizing its 0.35
            blend weight.
target:     balanced accuracy (maximize) · solo CV > node_0009 0.964215 with lower sem;
            swapping it for n9 lifts the blend beyond node_0010 0.965889.
