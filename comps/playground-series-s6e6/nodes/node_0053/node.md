---
id: node_0053
desc: multi-seed re-partition meta stack (5x10fold)
op: improve
parents: [node_0041]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969845
sem: 0.000280
folds: [0.970887, 0.969385, 0.969504, 0.969477, 0.969970]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [stack, multi-seed, variance-reduction]
---

## plan
built on:   node_0041 CORE+CatBoost 15-base stack (bases byte-identical) + DE threshold meta. Same 15 base OOF/test log-probs.
change:     re-partition the meta stack over SEEDS=[42,63,55555,37,47] with N_FOLDS=10. For each seed, re-fit the
            balanced-multinomial LogReg meta on a fresh 10-fold partition; average the stacked OOF and stacked test
            probs across the 5 partitions; apply the DE threshold to the AVERAGED OOF. Bases unchanged from node_0041.
hypothesis: averaging the meta over 5 independent fold partitions reduces meta-fit variance and lifts the stacked CV
            beyond the single-partition node_0041 without retraining any base.
target:     BA maximize · promote/submit only if CV > 0.969808 by >1·sem AND it holds on the untouched holdout;
            LB-gate before trusting (node_0047 mirage precedent).

## notes
Run date: 2026-06-09. Timing: ~12 min total (5 seeds x 10 folds x ~14s/fold, CPU).

CV comparison (fold-honest DE threshold on frozen 5-fold outer CV):
- node_0053 multi-seed (5x10fold): cv=0.969845 +/- 0.000280
- node_0041 single-seed (parent):  cv=0.969808 (baseline)
- Delta: +0.000037 (within 1 sem -- marginal improvement, not statistically clear)

Per-seed OOF BA (argmax, no DE): seed42=0.969778 seed63=0.969816 seed55555=0.969765 seed37=0.969862 seed47=0.969857 avg=0.969815

Holdout BA (fold 4, inviolable -- DE fit on folds 0-3, applied to fold 4):
- node_0053 multi-seed: 0.969970
- node_0041 single-seed: 0.969963
- Holdout delta: +0.000007 (essentially identical -- confirms no mirage, but also no real lift)

Conclusion: multi-seed averaging delivers negligible CV lift (+0.000037, < 1 sem).
The variance-reduction effect is real but the gap is within noise. The technique
matches the public 0.97105+ cluster's approach but our base OOF diversity/quality
is the binding constraint, not meta re-partitioning variance. Not a mirage (holdout
confirms it), but not a meaningful lift over node_0041 either. Orchestrator: do NOT
promote; the +0.000037 does not beat parent by >1 sem.
