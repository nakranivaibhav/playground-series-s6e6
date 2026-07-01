---
id: node_0003
desc: CatBoost, native cats
op: draft
parents: [root]
family: gbdt
uses_data: [fs_colors]
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.961294
sem: 0.000243
folds: [0.961866, 0.961751, 0.961383, 0.960666, 0.960802]
baseline_cv: 0.333333
shuffled_cv: 0.33761
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: true, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T12:14Z
decided: 2026-06-05T12:38Z
tags: [catboost, blend-arm]
---

## plan
built on:   root — second model family (toolkit branch B).
change:     CatBoost multiclass, auto_class_weights='Balanced', native categoricals via
            cat_features, all features + color indices. 5-fold OOF under frozen folds.
hypothesis: ordered boosting + native cats give errors de-correlated from LightGBM → blend value.
target:     a diverse valid arm; need not beat node_0001.

## notes
UNDERTRAINED: cv 0.961294 = -13.5σ below node_0001 (depth/iterations too low for the strong
signal). Kept valid as a diversity arm, but err-corr 0.80–0.87 with the LGBMs is high and the
weighted blend down-weights it to ~0. Not worth a retune unless decorrelation is needed.
No test_probs saved (only labels) — would need regen to enter a blend submission.
