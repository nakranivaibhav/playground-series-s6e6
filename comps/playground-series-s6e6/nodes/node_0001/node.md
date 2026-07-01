---
id: node_0001
desc: LightGBM, all feats + colors
op: draft
parents: [root]
family: gbdt
uses_data: [fs_colors]
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964569
sem: 0.000112
folds: [0.964579, 0.964626, 0.964795, 0.964145, 0.964699]
baseline_cv: 0.333333
shuffled_cv: 0.33495
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: true, passed: true}
gate_note: null
leak: clean
lb: 0.96612
submitted: 2026-06-05T11:44Z
created: 2026-06-05T11:32Z
decided: 2026-06-05T11:36Z
tags: [lightgbm, class_weight_balanced]
---

## plan
built on:   root — first real model.
change:     LightGBM multiclass on all raw features (alpha,delta,u,g,r,i,z,redshift) +
            native categoricals (spectral_type, galaxy_population) + 5 color indices;
            class_weight='balanced'. 5-fold OOF under frozen folds.
hypothesis: redshift + colors separate STAR/GALAXY/QSO; balanced weights lift minority recall.
target:     beats baseline 0.333 by a wide margin.

## notes
cv_too_good warn (0.333→0.965) explained by redshift's ~65% importance; shuffled control
collapsed to 0.335 (real signal). LB 0.96612 vs CV 0.964569 → gap +0.0015 (LB>CV, CV trustworthy).
Was champion; demoted to valid when node_0006 (its feature-rich child) beat it.
Parent of node_0002/0005/0006. test_probs regenerated post-hoc via src/dump_test_probs.py for blending.
