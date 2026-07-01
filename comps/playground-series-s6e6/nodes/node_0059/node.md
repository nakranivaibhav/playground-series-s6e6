---
id: node_0059
desc: cleanlab prune RealMLP-ref (C1)
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969115
sem: null
folds: [0.970200, 0.968722, 0.968726, 0.968849, 0.969079]
baseline_cv: 0.969065
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-10
decided: 2026-06-10
tags: [data-centric, cleanlab, label-noise, null]
---

## plan
built on:   node_0028 RealMLP-ref, byte-identical except training rows.
change:     drop cleanlab confident-learning flagged train rows (3224 total, 0.56%; ~2600/fold) before each fold fit; val rows kept so OOF stays comparable. Variant A=prune-all, B=GALAXY-only.
hypothesis: synthetic generator label noise concentrates in the low-z GALAXY/STAR confusion zone (CONFIRMED: flags 2.3x concentrated there); removing it sharpens the boundary.
target:     beat champion 0.969808 by >=2sem in a restack.

## notes
Variant A solo cv 0.969115 (+0.00005 vs n28); variant B 0.969039. Restack: champ15 swap -0.000093, +16th -0.000151, bank17+clean +0.000025 (within noise). NULL — strongly-regularized RealMLP already robust to the (real, concentrated) noise (AQuA). artifacts: oof_all.npy, oof_gal.npy, test_probs_all.npy, test_probs_gal.npy.
