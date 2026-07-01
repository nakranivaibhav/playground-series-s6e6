---
id: node_0069
desc: champion + 5-seed bag (seed-bag + DE-thresh)
op: improve
parents: [node_0063]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970032
sem: 0.000250
folds: [0.970914, 0.969806, 0.969585, 0.969616, 0.970241]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "DE threshold HURTS after bagging: delta=-0.000124; bagged argmax cv=0.970156 beats champion 0.970153 by +0.000003. Bagged sem=0.000250 slightly wider than champion 0.000222 (seed variance adds noise). Per-seed argmax spread std=0.000048 (very tight). Argmax-only bagged is the better slot."
leak: clean
lb: 0.97079
submitted: 2026-06-12T17:55Z
created: 2026-06-12
decided: null
tags: [public-bank, stack, seed-bag, exploit, finals-slot-1-eligible]
---

## plan
built on:   node_0063 champion (17-base balanced multinomial LogReg on clipped log-probs + DE per-class
            threshold). Meta recipe + base set byte-identical.
change:     Add 5-seed bagging — repeat the entire fold-honest stack over 5 seeds (StratifiedKFold seeds
            42..46), average the OOF and test probabilities across seeds, THEN apply the DE per-class
            threshold on the bagged OOF. Single-seed → 5-seed is the ONE change.
hypothesis: seed bagging reduces fold-split variance (the dominant noise on a single seed-42 stack), giving a
            more stable estimate that transfers better to the private 80%; combined with our DE threshold it
            should be ≥ champion and a lower-shake-up finals slot-1.
target:     Balanced Accuracy maximize; beats parent if CV > 0.970153 (node_0063). Even a CV-neutral result
            with tighter sem is a better finals pick (variance reduction).

CPU-only — pure numpy/sklearn over the OOF bank, minutes ×5 seeds.

CONTEXT — this is the seed-bag lever from the cdeotte diff (refs/cdeotte_lr_stacker/):
- Deotte's published stacker uses 5-seed bag (SEEDS=42..46) + plain argmax (NO threshold); raw probs; C=0.1.
- Our champion uses single seed-42 + DE threshold; clipped log-probs; C=1.0.
- The two recipes differ on exactly two substantive levers pulling opposite ways: HIS seed-bag (stability) vs
  OUR threshold (metric edge). This node combines BOTH: his seed-bag + our threshold.
- We tested 5-seed bagging once before (node_0053) and it was CV-neutral (+0.00004) on our recipe — but the
  point here is the FINALS angle: a CV-neutral but variance-reduced stack is a strictly better slot-1 (lower
  private shake-up), and we need to confirm the DE threshold still helps after bagging (or whether the threshold
  was partly fitting single-seed fold noise — that tells us if our public-LB edge transfers).

References to READ: champion/src (a1_full_merge.py ingest+merge, a1_submit.py DE-threshold final fit);
refs/cdeotte_lr_stacker/gpu-logistic-regression-stacker.ipynb (his exact 5-seed loop, cell 3); node_0053
(our prior 5-seed re-partition result).

A/B to report: (1) bagged-CV vs champion 0.970153; (2) per-seed CV spread + the bagged sem vs champion sem
0.000222; (3) DE-threshold-ON vs argmax-only on the bagged OOF (does the threshold still add after bagging?).
Keep folds frozen-anchored at seed-42 for fold0; the extra 4 seeds add re-partitions for bagging only.

## notes
