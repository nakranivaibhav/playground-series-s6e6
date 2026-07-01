---
id: node_0086
desc: z-conditional color-residual TabM base
op: draft
parents: [root]
uses_data: [fs_zresid]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965728
sem: 0.000305
folds: [0.966331, 0.966511, 0.964935, 0.965235, 0.965631]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "Stack-add onto n76 yields +0.000055 delta (2*sem threshold=0.000507). Does NOT beat champion by >2*sem. Mean err-corr vs 17-bank=0.7227 (not decorrelated enough). Solo BA=0.965728 is above cheap-kill (0.965). Node is valid but does not lift stack."
leak: clean
lb: null
submitted: null
created: 2026-06-13T15:51Z
decided: 2026-06-13T17:10Z
---

## plan
built on:   root — fresh draft; reuses the TabM training loop + recipe from node_0033 (`nodes/node_0033/src`).
change:     Train a strong TabM (n33 recipe) on a z-residual-DOMINATED feature set: NEW fs_zresid (fit_in_fold) = for each color (u-g, g-r, r-i, i-z, u-z) and each magnitude, the z-conditional z-score (color − mean_zbin)/std_zbin over ~40 train-fold redshift quantile bins, PLUS raw redshift kept (STAR z≈0). DROP raw colors so the model's primary signal is color-anomaly-at-fixed-z, not z-then-color.
hypothesis: A base whose primary signal is color anomaly at fixed redshift produces errors decorrelated from the z-then-color bank, so adding it to the stack lifts CV where every in-house add washed.
target:     Balanced Accuracy maximize; solo BA need not beat the bank, but the stack-add onto n76 must beat champion 0.970153 by >2·sem (decorrelation gate first).

This is the ONE genuinely un-tried honest inductive bias (research.md 2026-06-13T11:50Z, lines 74-105). Every bank base splits on marginal z first; none is forced onto the z-conditional residual manifold, so this base's OOF errors should DECORRELATE from the 17-bank.

fs_zresid leak-safety = **fit_in_fold**: redshift quantile-bin EDGES and per-bin color/mag mean/std are fit on the TRAIN FOLD ONLY, then applied to val+test. Use ~40 bins; fall back to global mean/std for sparse bins. Recipe (research.md lines 83-105): (1) bin redshift into ~40 quantile bins, edges fit on train fold only; (2) within each z-bin fit per-bin MEAN and STD of each color (u-g, g-r, r-i, i-z, u-z) and each magnitude on the train fold, then transform every row (train+val+test) into its z-conditional z-score (color − mean_zbin)/std_zbin; keep raw redshift; DROP raw colors. This MUST be residual-dominated, not raw+residual — the additive form (n6/n11) already washed.

IMPORTANT — although fs_zresid is conceptually shared with node_0087, **implement the fs_zresid builder in THIS node's own src/** (identical recipe). Do NOT import node_0087's code; the two nodes build separately/in parallel, so there must be no cross-node file dependency. data.md documents the single canonical recipe.

READ: research.md lines 83-105 (exact recipe), `nodes/node_0033/src` (TabM training loop + fs_realmlp_fe loader — keep redshift, swap colors for residuals), data.md fs_realmlp_fe row and the new fs_zresid row.

CHEAP-KILL: run fold-0 first; if fold-0 solo BA < 0.965 (well below the bank-base solo floor) STOP before the remaining folds.

GATE before judging: compute pairwise OOF error-correlation of this base vs the 17-bank; accept only if decorrelated AND, once forward-selected onto the n76 stack, blended OOF BA > champion by >2·sem.

Frozen folds.json; GPU.

well: outside
