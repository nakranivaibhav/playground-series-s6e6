---
id: node_0013
desc: LightGBM + positional features
op: improve
parents: [node_0006]
family: gbdt
uses_data: [fs_colors, fs_research, fs_positional]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.963977
sem: 0.000167
folds: [0.963893, 0.964049, 0.963510, 0.964540, 0.963890]
baseline_cv: 0.965004
shuffled_cv: 0.33233
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: "leak-scan warn at line 147 = full-train fit for test submission (test never in fit); verified safe"
leak: clean
lb: null
submitted: null
created: 2026-06-05T16:36Z
decided: null
tags: [lightgbm, positional-features, sky-cell, knn-density, leak-safe]
---

## plan
built on:   node_0006 (champion-lineage LightGBM, exact config) + its full 28-feature set.
change:     add LEAK-SAFE positional features motivated by the drop-column study (delta = the
            single most irreplaceable feature). All LABEL-FREE:
            - clean.add_positional_features (unit-tested): sin/cos RA (wrap fix), unit-sphere
              cartesian sx/sy/sz, delta×redshift interactions, and sky_cell — a coarse
              (RA 10°×Dec 5°) grid-cell id as a NATIVE CATEGORICAL (the GBDT learns the
              per-region class tendency inside the fold; no manual target encoding).
            - knn_dist5: distance to the 5th-nearest TRAINING object on the unit sphere
              (a local sky-density proxy; KDTree reference = train positions only, no labels).
            sky_cell test categories aligned to the train vocabulary (test-only cells→missing).
hypothesis: richer spatial features let the tree exploit the dominant positional signal that
            raw delta + axis-aligned splits use poorly → beat node_0006 0.965004.
target:     beat node_0006 0.965004 beyond fold-noise; and ideally lift the blend (node_0010).
leak-safety: NO feature uses the target → the shuffled-label control must collapse to 1/3.

## notes
HYPOTHESIS REFUTED (cleanly, leak-free). cv 0.963977 ± 0.000167 = −0.001027 vs node_0006
(~6σ), regressed on ALL 5/5 folds. Shuffled control 0.33233 → leak-safe, so this is genuine
overfitting, not a leak. Diagnosis: the positional signal was ALREADY fully captured by
node_0006's raw delta + gal_l + gal_b; re-encoding it added overfitting surface. Prime suspect:
sky_cell (416-level native categorical) — high-cardinality categoricals let GBDTs carve
train-fold-specific spatial noise. The smooth re-encodings (sin/cos RA, sx/sy/sz, knn_dist5)
are redundant with the raw coords. LESSON: a feature being IMPORTANT (delta, drop-col #1) does
NOT mean MORE features derived from it help — the model already extracts it. Not a blend arm
(weaker + correlated with node_0006). Champion node_0010 unchanged.
