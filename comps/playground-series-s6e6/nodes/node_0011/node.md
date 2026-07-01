---
id: node_0011
desc: XGBoost full feats (GPU)
op: improve
parents: [node_0004]
family: gbdt
uses_data: [fs_colors, fs_research]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964918
sem: 0.000180
folds: [0.965188, 0.965233, 0.964493, 0.964462, 0.965215]
baseline_cv: 0.964414
shuffled_cv: 0.33380
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: "leak-scan warn at line 132 = full-train fit for test submission (test never in fit); verified safe, same as node_0004/0006"
leak: clean
lb: null
submitted: null
created: 2026-06-05T15:54Z
decided: null
tags: [xgboost, full-features, gpu, blend-arm]
---

## plan
built on:   node_0004 (XGBoost) — byte-identical hyperparameters; the ONE change is the
            feature set: base 15 → full 28 (research features, same as node_0006/TabM).
            device='cuda' only for speed. Saves test_probs.npy.
change:     close the feature gap — node_0004 ran on base feats (0.964414); add the research
            features (extended/curvature colors, redshift transforms, QSO box, galactic coords).
hypothesis: same +~0.0004 lift LightGBM saw (node_0001→node_0006) → a stronger, still
            de-correlated XGB arm that lifts the blend past node_0010 (0.965889).
target:     beat node_0004 0.964414; and lift the combine beyond 0.965889.

## notes
cv 0.964918 ± 0.000180 = +0.000504 vs node_0004 (base feats) — confirms the research features
lift XGBoost ~the same as LightGBM. Now near the best single (node_0006 0.965004) and GPU-fast.
Leak-clean (shuffled 0.33380). Also the vehicle for the drop-column feature study (drop_column.log).
BLEND TWIST: although stronger solo than node_0004, swapping n4→n11 makes the blend WORSE
(0.965763, −0.000126) — n11 shares n6's full features so err-corr 0.88 (vs n4's 0.85), losing the
base-feature diversity the blend relied on. No arm-set with n11/n12 beats champion node_0010
(0.965889) beyond sem → node_0013 not built. Stronger-but-correlated < weaker-but-diverse for blends.
