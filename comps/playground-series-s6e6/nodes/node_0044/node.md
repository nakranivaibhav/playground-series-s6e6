---
id: node_0044
desc: xgb-v5 zoo port (canary base)
op: draft
parents: [root]
uses_data: [fs_zoo]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.96769
sem: 0.00023
folds: [0.96843, 0.96791, 0.96757, 0.96703, 0.96749]
baseline_cv: 0.966627
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-08T13:51Z
decided: 2026-06-08T20:00Z
tags: [zoo, cdeotte, xgb, canary, original-priors]
---

## plan
built on:   root (a brand-new base; reuses NOTHING from our prior nodes).
change:     Faithfully reproduce CdeOtte's `xgb-v5-for-s6e6` base on OUR folds.json.
            Build the SHARED zoo feature pipeline `fs_zoo` (this node owns it; nodes
            0045/0046/0047 reuse it): rich color FE (pairs+abs, mag stats incl
            argmin/argmax, blue/red slopes, curv_ugr/gri/riz, redshift transforms),
            art floor-cross categorical bins (alpha/delta floors, redshift/10 'tenth',
            alpha/0.2 'deg5', color-half × redshift-tenth crosses), in-fold smoothed
            TargetEncoder (smooth=16, 7 inner splits) on the cat keys, ORIGINAL-SDSS17
            PRIOR features (P(class|key) computed on data/sdss17/star_classification.csv
            ONLY — orig rows NOT appended to training; USE_ORIGINAL_ROWS=False) +
            frequency features, then TOP-370 feature selection by importance. Head =
            XGBClassifier(tree_method=hist, device=cuda, lr 0.012, n_estimators 7000,
            max_depth 0/lossguide, min_child_weight 10, gamma 0.2, reg_alpha 0.30,
            reg_lambda 4.0, subsample 0.82, colsample_bytree 0.74, colsample_bylevel
            0.86, early_stopping on the fold val). Emit oof.npy (N×3, OUR train order),
            test_probs.npy (M×3), submission.csv.
hypothesis: A faithfully-built top-zoo XGB (vs our under-built XGB node_0031 0.96624)
            reaches ~0.969 solo AND injects original-prior signal none of our 15 bases
            have — a stronger, partially-de-correlated arm that lifts the stack toward
            the 0.97126 public cluster.
target:     Balanced Accuracy (maximize) · solo CV beats our XGB node_0031 (0.96624);
            stretch ~0.969 like the reference. Stack value proven in node_0048.

## notes
Reference: refs/xgb-v5-for-s6e6.py (1105-line GPU cudf/cuml notebook — PORT to our
CPU pandas + our GPU xgboost; cuml.TargetEncoder → manual smoothed in-fold TE or
category_encoders). Leakage rules: TE + frequency fit INSIDE each train fold only;
original-prior features computed on orig data only (orig labels disjoint from our
train/val → no fold leak, compute once); NEVER fit any encoder on val/test labels.
Build OOF strictly on OUR folds.json so it aligns with the stack.
