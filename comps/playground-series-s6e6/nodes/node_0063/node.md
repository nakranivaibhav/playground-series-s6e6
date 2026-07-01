---
id: node_0063
desc: public 18-model bank stack (A1)
op: combine
parents: [root]
uses_data: []
family: ensemble
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970153
sem: 0.000222
folds: []
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.97073
submitted: 2026-06-10
created: 2026-06-10
decided: 2026-06-10
tags: [public-bank, stack, external-oof, finals-slot-1]
---

## plan
built on:   root (combine of 17 public base models, NOT our nodes). External OOF from the public Deotte bank
            (cdeotte/s6e6-oof-and-test-preds + 12 kernel outputs), reproduced on OUR frozen folds — verified a
            100% match to StratifiedKFold(5,shuffle,42), so the public OOF aligns row-for-row.
change:     balanced multinomial LogReg meta on the 17 bases' clipped log-probs + honest DE per-class threshold
            (the champion recipe) over the public bank instead of our 15 in-house bases.
hypothesis: the public ensemble carries strong+decorrelated signal our saturated in-house stack lacks.
target:     beat prior champion node_0041 0.969808.

## notes
CV 0.970153 (+0.000345 vs node_0041), LB 0.97073 (+0.0003 vs node_0041 0.97043) — best CV AND best honest LB.
Merging our 15 bases in DILUTES it (0.970025); 5-seed re-partition meta WORSE (0.970032); our decorrelated
weak bases (DAE etc.) don't lift it. Fully fold-honest, NO test fitting. Finals slot-1. Artifacts: oof.npy,
test_probs.npy, submission.csv; build in src/ (a1_full_merge.py ingest+merge, a1_submit.py final fit).
Reproducible from refs/oof_bank + refs/kernel_out. NOTE: depends on external public artifacts (snapshotted in refs/).
