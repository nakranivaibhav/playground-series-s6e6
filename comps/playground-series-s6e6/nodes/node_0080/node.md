---
id: node_0080
desc: TabPFN-3 as L1 meta-stacker
op: draft
parents: [root]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969717
sem: 0.000300
folds: [0.970875,0.969527,0.969124,0.969502,0.969554]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: cv below champion by 0.000436 (within 2·sem=0.000600); subsample of 10k meta-fit rows is the bottleneck
leak: clean
lb: null
submitted: null
created: 2026-06-13T09:13Z
decided: 2026-06-13T09:55Z
---

## plan
built on:   root (a fresh meta-learner draft over the existing bank17+FT-T base OOFs); the base set + a1 ingest reuse champion/src.
change:     Replace the balanced-LogReg L1 meta-stacker with a TabPFN-3 classifier as the L1 meta over the bank17+FT-T base OOFs (stacking, not a primary base). Use TabPFN-3 (the philippsinger tabpfn-3-stacker kernel as reference) as the meta-learner on top of the bank17+FT-T base probability matrix, fold-honest nested, vs the LogReg meta; report CV/sem; CPU/GPU per TabPFN reqs.
hypothesis: a more expressive non-linear meta-learner (TabPFN-3) over the same bases could extract stacking gains the linear LogReg meta leaves on the table.
target:     Balanced Accuracy maximize; beats LogReg-meta champion if CV > 0.970153 by > 2·sem.

TabPFN-3 stacking as an alternative L1 meta-learner over the existing bank17+FT-T bases. READ refs/pull_philippsinger_tabpfn-3-stacker for the meta-stacker recipe; champion/src a1 scripts for the current LogReg meta it replaces. Fold-honest: fit the meta inside each train fold only, never on full train or test. Verify alignment (577347 train rows id-ordered, 247435 test). If TabPFN-3 cannot handle the row count, subsample the meta-fit train per its constraints but keep OOF coverage honest.

well: wildcard.
