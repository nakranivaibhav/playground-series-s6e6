---
id: node_0023
desc: CatBoost undertraining fix
op: improve
parents: [node_0003]
uses_data: [fs_colors]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.962737
sem: 0.000239
folds: [0.963455, 0.962953, 0.962562, 0.961993, 0.962722]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-07T06:06Z
decided: 2026-06-07T07:02Z
tags: [catboost, gbdt, blend-arm, improve]
---

## plan
built on:   node_0003 (CatBoost, native cats, fs_colors) — keep its feature set (fs_colors)
            and native categorical handling byte-identical. Do NOT swap features; the full-feature
            CatBoost already exists as node_0012, so swapping would duplicate it.
change:     ONE atomic change vs node_0003 = proper training budget — sufficient `iterations`
            + early-stopping on the fold val (node_0003 was undertrained at cv 0.961294). Keep
            native categoricals, fs_colors. Produce fold-honest OOF over folds.json → oof.npy +
            test_probs.npy.
hypothesis: a properly-trained CatBoost is a de-correlated GBDT column (different split/ordering
            scheme than LightGBM/XGB) the stack currently lacks; distinct from node_0012 by
            feature set (fs_colors, not full).
target:     BA maximize, solo > 0.961294 (node_0003), ideally ≥ ~0.963, leak-clean.
