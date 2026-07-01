---
id: node_0007
desc: combine blend n6+n4+n1
op: combine
parents: [node_0006, node_0004, node_0001]
family: ensemble
uses_data: []
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965530
sem: 0.000188
folds: [0.966235, 0.965247, 0.965202, 0.965577, 0.965391]
baseline_cv: 0.965004
shuffled_cv: 0.33303
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.96702
submitted: 2026-06-05T14:23Z
created: 2026-06-05T14:14Z
decided: 2026-06-05T14:16Z
tags: [ensemble, blend, weighted-probability-average, fold-honest]
---

## plan
built on:   node_0006 (champion LightGBM+research feats), node_0004 (XGBoost, best
            de-correlated arm, err-corr 0.85), node_0001 (base LightGBM) — uses their
            saved OOF + test probability matrices byte-identically; no model retrain.
change:     weighted probability-average blend. Weights chosen by a FOLD-HONEST nested
            search: for each fold f, the simplex weights are optimized on the OTHER folds'
            OOF and evaluated on fold f (weights never see the scored fold's labels). Final
            test weights are refit on the full OOF. Probability-average, not LR-meta (which
            optimizes logloss, not balanced accuracy — analysis showed it hurts).
hypothesis: n4 (XGB) errors are de-correlated from the LightGBM lineage; averaging their
            class posteriors corrects boundary cases the champion alone gets wrong.
target:     beat champion node_0006 cv 0.965004 beyond fold-noise (analysis OOF ≈ 0.965581).

## notes
WINNER → CHAMPION. Honest nested CV 0.965530 ± 0.000188 = +0.000526 vs node_0006 (>2·sem on
either node). Final weights n6:0.40 / n4:0.40 / n1:0.20 — STABLE: 3/5 folds independently
picked exactly (0.40,0.40,0.20), an interior solution (not a degenerate corner / noise grab).
Champion-only-n6 honest = 0.965004, exactly reproduces the recorded champion CV (sanity ✓).
Shuffled-label control collapsed to 0.33303 ≈ 1/3 (the weight-search itself extracts no signal
from permuted labels). Submission flips only 1,453/247,435 rows (0.59%) vs node_0006 — a small
surgical correction on the QSO/STAR boundary; predicted dist 63.5/20.7/15.8% ~ train prior.
Best clean single = node_0006; this blend is the stronger submit candidate. Not yet submitted.
