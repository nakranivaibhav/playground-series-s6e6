---
id: node_0087
desc: z-conditional residual LightGBM base
op: draft
parents: [root]
uses_data: [fs_zresid]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.963099
sem: 0.000181
folds: [0.963250, 0.963564, 0.962575, 0.962788, 0.963318]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "stack-add to n76 bank18 gives cv=0.970198 vs n76 cv=0.970227; delta vs champion=+0.000045 < 2*sem=0.000520 — does NOT beat champion by >2*sem; mean err-corr vs bank18=0.7046 (NOT decorrelated — hypothesis not confirmed)"
leak: clean
lb: null
submitted: null
created: 2026-06-13T15:51Z
decided: 2026-06-13T17:10Z
---

## plan
built on:   root — fresh draft; reuses the LightGBM rich-FE loader/params from node_0030 (`nodes/node_0030/src`).
change:     Same z-residual lever as node_0086 but with a LightGBM base (n30 recipe) instead of TabM: train on fs_zresid (residual-dominated: z-conditional color/mag z-scores + raw redshift, raw colors DROPPED). Produce OOF + test_probs as a new stack candidate.
hypothesis: A GBDT forced onto z-conditional residuals decorrelates from the z-then-color tree bank; the LightGBM and TabM residual bases together give two orthogonal new stack members.
target:     Balanced Accuracy maximize; stack-add onto n76 must beat champion 0.970153 by >2·sem; decorrelation gate first.

Parallel, independent realization of the research.md z-conditional-residual lever (lines 74-105) on a DIFFERENT family from node_0086, so the two are de-correlated from each other and we attribute the lever to architecture. Trees split on z first then raw colors in coarse z-regions and NEVER construct a continuous z-conditional anomaly, so a tree forced onto residuals is the cleaner test that the decorrelation is real and not NN-specific.

fs_zresid leak-safety = **fit_in_fold**: bin edges + per-bin color/mag mean/std fit train-fold-only, then applied to val+test. Recipe (research.md lines 83-105): ~40 redshift quantile bins (edges train-fold-only); per-bin MEAN/STD of each color (u-g, g-r, r-i, i-z, u-z) and each magnitude fit on train fold; transform every row into z-conditional z-score (color − mean_zbin)/std_zbin; keep raw redshift; DROP raw colors; global mean/std fallback for sparse bins.

IMPORTANT — although fs_zresid is conceptually shared with node_0086, **implement the fs_zresid builder in THIS node's own src/** (identical recipe). Do NOT import node_0086's code; the two nodes build separately/in parallel, so there must be no cross-node file dependency. data.md documents the single canonical recipe.

READ: research.md lines 83-105 (recipe), `nodes/node_0030/src` (LightGBM richFE loader/params — swap fs_realmlp_fe colors for fs_zresid residuals, keep redshift), data.md fs_zresid row.

Same accept gate: OOF error-corr vs the 17-bank, then stack-add vs champion >2·sem.

CPU/GPU LightGBM, minutes. Frozen folds.json.

well: outside
