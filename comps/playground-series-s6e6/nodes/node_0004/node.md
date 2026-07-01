---
id: node_0004
desc: XGBoost, native cats
op: draft
parents: [root]
family: gbdt
uses_data: [fs_colors]
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964414
sem: 0.000137
folds: [0.964919, 0.964493, 0.964202, 0.964199, 0.964256]
baseline_cv: 0.333333
shuffled_cv: 0.33461
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: true, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T12:14Z
decided: 2026-06-05T12:18Z
tags: [xgboost, blend-arm, tree-cap-hit]
---

## plan
built on:   root — third model family (toolkit branch C).
change:     XGBoost multiclass (multi:softprob), balanced sample weights, enable_categorical
            (native cats, no one-hot), tree_method='hist', all features + color indices.
            5-fold OOF under frozen folds.
hypothesis: a structurally independent GBDT → de-correlated errors for the blend.
target:     a diverse valid arm.

## notes
cv 0.964414 = -1.1σ vs node_0001 (within noise) — the BEST de-correlated blend arm (err-corr
0.85, lower than CatBoost's). Hit its 800-tree cap (best_iter=799) without early-stopping →
headroom from more trees / lower LR. test_probs regenerated post-hoc via src/dump_test_probs.py
for the blend. Primary partner for node_0006 in the combine node.
