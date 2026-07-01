---
id: node_0116
desc: LOO-pruned restack drop 3 harmful bases
op: combine
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970369
sem: 0.000239
folds: [0.971114, 0.970126, 0.970070, 0.969808, 0.970726]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.97110
submitted: 2026-06-18T09:18Z
created: 2026-06-16T13:01Z
decided: 2026-06-16T13:54Z
tags: [stack, loo-prune, exploit, combine]
---

## plan
built on:   node_0091 (champion: C0.003 balanced multinomial LogReg mega-stack over the FULL 63-base pool, cv 0.970355 · lb 0.97121). The meta FAMILY, clipped log-probs, nested in-fold C grid, frozen folds, and plain argmax all stay byte-identical; ONLY the base list changes.
change:     Refit the champion C0.003 balanced-LogReg meta on the SAME pool MINUS the 3 bases the LOO drop-study found have NEGATIVE causal contribution (xgb-6, tabm-0, node_0042; deltas −6.5e-5/−5.8e-5/−4.4e-5, sum −1.67e-4). One atomic change vs n091: the base list shrinks by 3 cols; everything else (clip log-probs, nested in-fold C grid, frozen folds, plain argmax) byte-identical. Optionally A/B a backward-elimination variant that also drops cat-0/node_0030/node_0049 (next-worst, all negative).
hypothesis: n091's shrinkage leaves a small negative-LOO drag in the pool; removing exactly the bases with negative causal contribution lifts honest CV by ~the sum of their drags, the cleanest remaining shot at clearing the 2·sem promote bar.
target:     Balanced Accuracy maximize; beats parent n091 if CV > 0.970355, promote-eligible if > 0.970853 (n091 + 2·sem 0.000249).

This is the ONE combine sub-axis NEVER executed. Every other meta lever is closed with evidence: GBDT-meta overfits (n099 −0.0016), FWLS region-onehot washes (n100 −0.0005), TabPFN-3 meta below (n080), per-base Dirichlet calib washes (n104), per-class DE-threshold washes (n091 probe). But n091 dumps ALL 63 bases under shrinkage and the LOO study (probes/drop_study_ranking.csv, journal 2026-06-16T08:57Z) proved shrinkage does NOT fully zero the harmful tail — 3 bases still have negative LOO delta IN the fitted pool. n091 only beats n070 by +0.000144 (sub-2sem); removing −0.000167 of drag could plausibly cross the 2·sem promote bar. READ: champion/src/solution.py (the exact OOF-ingest + clip+norm + LogisticRegressionCV nested-C loop to copy and just edit the base ID list); probes/drop_study_ranking.csv (the full 63-base ranking — bottom rows 61/62/63 = node_0042/tabm-0/xgb-6 are the drop targets, and the −delta tail extends to ~rank 56); nodes/node_0091/node.md notes (top-|coef| bases, the C-vs-CV curve, the baseline-assert gate that MUST reproduce bank-17+FT-T ≈ 0.970211 first). Run the SAME baseline-assert guard, report the pruned CV/sem + per-fold + the LOO-delta of each dropped base. CPU-only, minutes; uv run --no-sync.

## notes
well = exploit.

RESULT (NULL — does not promote). baseline-assert reproduced bank-17+FT-T 0.970227 (≈0.970211) PASS. REF-FULL n091 0.970355. PRUNED-3 (drop xgb-6/tabm-0/node_0042) 0.970347 = −0.000007 vs ref, NOT the predicted +1.67e-4. PRUNED-6 (also drop cat-0/node_0030/node_0049) 0.970369 = +0.000014 vs ref — the winner saved here, but well under the 2·sem promote bar 0.970853. Lesson: LOO drop-deltas do NOT sum under refit — L2 shrinkage already absorbs the negative-LOO drag, so pruning the harmful tail recovers ~nothing. The one untried combine sub-axis is now closed with evidence.
