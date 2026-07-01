---
id: node_0015
desc: LightGBM dart boosting variant
op: improve
parents: [node_0006]
uses_data: [fs_colors, fs_research]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.960408
sem: 0.000131
folds: [0.960376, 0.960849, 0.960246, 0.960075, 0.960495]
baseline_cv: 0.965004
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv regressed -0.0046 vs node_0006 on all 5 folds; valid diversity arm for blending only"
leak: clean
lb: null
submitted: null
created: 2026-06-06T08:49Z
decided: 2026-06-06T12:06Z
tags: [lightgbm, dart, gbdt, diversity-arm]
---

## plan
built on:   node_0006 (best single, lgbm + research feats) — byte-identical features,
            folds, and OOF/test plumbing; only the boosting strategy changes.
change:     ONE atomic change — boosting_type 'gbdt' → 'dart'. Set drop_rate ~0.1,
            skip_drop ~0.5, and raise num_iterations to compensate for DART's slower
            convergence. Same fs_research feature matrix, same 5 frozen folds. Save
            oof.npy + test_probs.npy.
hypothesis: DART dropout yields a LightGBM whose per-fold errors are disjoint from n1/n6
            (the tree arms currently err-corr 0.85-0.87), giving the de-correlated tree arm
            the blend lacks.
target:     Balanced Accuracy (maximize) · solo cv within ~1·sem of node_0006 (0.965004)
            AND lower err-corr vs n6; beats node_0006 as a blend arm if it lifts node_0010
            beyond sem.
