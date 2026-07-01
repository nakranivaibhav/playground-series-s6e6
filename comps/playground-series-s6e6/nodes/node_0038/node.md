---
id: node_0038
desc: ExtraTrees on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: tree
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.961325
sem: 0.000212
folds: [0.961342, 0.961902, 0.960589, 0.961479, 0.961311]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn fires because baseline_cv=0.333333 (dumb constant predictor); all valid nodes in this competition score 0.96+. Not suspicious — expected for this feature set. Warn only, not a blocker."
leak: clean
lb: null
submitted: null
created: 2026-06-07T14:57Z
decided: null
tags: [tree, extratrees, randomized-trees, cross-family-diversity, draft]
---

## plan
built on:   root (new draft, cross-family diversity). Template src to COPY from
            node_0028/src (keeps the fs_realmlp_fe FE pipeline + fold-honest OOF/test_probs
            scaffold over the FROZEN folds.json). The developer SWAPS the model only:
            replace the RealMLP with an ExtraTreesClassifier.
change:     reuse the rich fs_realmlp_fe FE, then fit an ExtraTreesClassifier (sklearn, many trees
            e.g. n_estimators≈500, class_weight='balanced', n_jobs=-1). A randomized-split (extra-
            random) tree family — a DIFFERENT tree inductive bias than the boosted GBDTs
            (LightGBM/XGBoost/CatBoost). CPU. Fold-honest OOF (577347×3) + test_probs (247435×3)
            over the frozen folds.
hypothesis: ExtraTrees uses fully randomized splits + bagging rather than gradient boosting, so
            its OOF errors should be de-correlated from the boosted-GBDT bases and supply the meta-
            stacker a fresh tree perspective the boosted family doesn't cover.
target:     BA maximize; solo ≥ 0.96. Valued for DE-CORRELATION (de-correlated tree base; stack
            lift via restack_probe.py), not solo strength.
