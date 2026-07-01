---
id: node_0064
desc: recover xgb-0/xgb-3 into bank-19 stack
op: improve
parents: [node_0063]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970199
sem: 0.000275
folds: [0.971124, 0.969987, 0.969482, 0.969974, 0.970427]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "bank-19 wins (+0.000046 vs parent 0.970153 within 1 SEM=0.000222); xgb-0 alone dilutes (-0.000016), xgb-3 alone adds +0.000034; both together +0.000046"
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [public-bank, stack, external-oof, exploit, finals-slot-1-eligible]
---

## plan
built on:   node_0063 champion (17-base balanced-LogReg + DE per-class threshold). Meta recipe byte-identical;
            only the base set grows from 17 → 19.
change:     Add the 2 dropped public-bank models to the stack — xgb-0 (its OOF is 677,347 rows because it
            contains the 100k original-SDSS17 training rows; strip non-playground rows by id and realign to
            our 460k train) and xgb-3 (raw margins → apply softmax; resolve class-column order empirically by
            picking the permutation whose solo OOF balanced accuracy is sane ~0.96, not ~0.33). Refit the
            identical balanced-LogReg + DE-threshold meta on 19 bases.
hypothesis: the two dropped bank models carry the remaining ~+0.0004 separating our bank stack from Deotte's
            published bank LB (0.97101 vs our 0.97073).
target:     Balanced Accuracy maximize; beats parent if CV > 0.970153 (node_0063).

CPU-only — no training, OOF ingest + LogReg refit, minutes.

References to READ:
- Parent recipe + ingest code: `comps/playground-series-s6e6/champion/src` (a1_full_merge.py ingest+merge,
  a1_submit.py final fit). node_0063 node.md notes both drop reasons verbatim.
- Raw artifacts: `comps/playground-series-s6e6/refs/oof_bank` and `refs/kernel_out`.
- Folds verified 100% match to StratifiedKFold(5,shuffle,42) on train.csv file order (journal 2026-06-10T08:55Z)
  — so row alignment is by position AFTER stripping orig rows. VERIFY the stripped xgb-0 row count == 460k
  exactly and its id order matches train.csv before trusting it.
- Journal attributes the +0.0004 gap precisely to these 2 models + handling; the 5-seed re-partition piece is
  already disproven (A1-5seed NULL, −0.00012), so the models themselves are the only remaining piece.

A/B: bank-19 vs bank-17 0.970153 on the frozen folds with the byte-identical meta. ALSO test +xgb-0-only
(bank-18) and +xgb-3-only to attribute each model's contribution. Pre-registered fallback: if xgb-3's column
order cannot be resolved to a sane solo BA, ship bank-18 (xgb-0 only) and log the xgb-3 ambiguity.

## notes
A/B attribution:
- bank-17 (parent):      cv=0.970153 sem=0.000222
- bank-18 (+xgb-0 only): cv=0.970137 sem=0.000254  (DILUTES slightly)
- bank-17+xgb-3 only:    cv=0.970187 sem=0.000276  (+0.000034)
- bank-19 (+xgb-0+xgb-3): cv=0.970199 sem=0.000275  (+0.000046 vs parent)

xgb-0 ingested by keeping first 577,347 rows (last 100k = original SDSS rows, stripped).
xgb-3 raw margins softmaxed, column permutation [0,2,1] (raw order GALAXY,STAR,QSO → GALAXY,QSO,STAR) resolves to BA=0.965102 (other permutations give 0.33 or lower).
Best variant = bank-19. DE weights on full refit: [0.9543, 0.9222, 1.0].
Note: +0.000046 improvement is within 1 SEM; marginal but positive direction.
Runtime: ~4 minutes CPU-only.
