---
id: node_0000
desc: majority-class baseline
op: draft
parents: [root]
family: baseline
uses_data: []
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.333333
sem: 0.000000
folds: [0.333333, 0.333333, 0.333333, 0.333333, 0.333333]
baseline_cv: 0.333333
shuffled_cv: null
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.33333
submitted: 2026-06-05T11:28Z
created: 2026-06-05T11:26Z
decided: 2026-06-05T11:27Z
tags: [baseline, constant]
---

## plan
built on:   root (the dumb floor).
change:     predict the majority class (GALAXY) for every row; constant fit inside each
            train fold only — the label-metric optimum for a constant predictor.
hypothesis: per-class recall (1,0,0) → balanced accuracy = 1/3 by construction; proves the
            data→CV→submit→Kaggle pipe end-to-end before any model.
target:     Balanced Accuracy maximize · any real model must beat 0.333.

## notes
First champion (later demoted). LB 0.33333 == CV exactly → pipe verified, CV mirrors Kaggle.
