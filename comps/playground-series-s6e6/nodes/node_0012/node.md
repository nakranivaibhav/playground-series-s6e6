---
id: node_0012
desc: CatBoost full feats (GPU)
op: improve
parents: [node_0003]
family: gbdt
uses_data: [fs_colors, fs_research]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.961957
sem: 0.000149
folds: [0.962037, 0.962501, 0.961664, 0.961787, 0.961798]
baseline_cv: 0.961294
shuffled_cv: 0.33660
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T15:54Z
decided: null
tags: [catboost, full-features, gpu, blend-arm]
---

## plan
built on:   node_0003 (CatBoost) — byte-identical hyperparameters; the ONE change is the
            feature set: base 15 → full 28 (research features). task_type='GPU' only for speed.
            Saves test_probs.npy.
change:     close the feature gap — node_0003 ran on base feats (0.961294, undertrained); add
            the research features.
hypothesis: research features lift CatBoost toward the other GBDTs and add a de-correlated arm.
target:     beat node_0003 0.961294; useful if it earns weight in the combine.

## notes
cv 0.961957 ± 0.000149 = +0.000663 vs node_0003 (base feats). Features helped, but CatBoost
remains the weakest GBDT (~0.962, vs ~0.965 for LGBM/XGB). Leak-clean (shuffled 0.33660). Likely
earns little/zero blend weight given it's dominated — confirmed in reblend_analysis.log.
