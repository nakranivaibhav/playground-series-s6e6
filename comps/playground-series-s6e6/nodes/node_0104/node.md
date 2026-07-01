---
id: node_0104
desc: Dirichlet-calibrate bases before LogReg meta
op: improve
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970309
sem: 0.000229
folds: [0.971051, 0.970066, 0.970025, 0.969794, 0.970611]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "WASH. Per-base Dirichlet/matrix-scaling calibration (63 bases, fit_in_fold multinomial-LogReg on each base's 3 log-probs) BEFORE the C=0.003 LogReg meta → CV 0.970309 vs champion 0.970355, lift −0.000046. Baseline reproduced EXACTLY (0.970355). The global LogReg meta already subsumes per-base scaling — calibrating each base first adds nothing. Closes the base-INPUT-TRANSFORM axis (the one combine sub-axis the n099/n093/n100/n080 meta-mechanism nodes hadn't touched): combiner is maxed on mechanism AND on per-base preprocessing."
leak: clean
lb: null
submitted: null
created: 2026-06-15T07:06Z
decided: 2026-06-15
---

## plan
built on:   champion n091 full-pool L2-LogReg @C=0.003. Copy node_0099/src/solution.py as the working
            full-pool loader (OOF_mat, nested_cv_arm_logreg, logp/norm, frozen-fold loop). DO NOT re-derive
            the loader and DO NOT read node_0091's solution.py (those reads overflowed n099/n100 devs).
change:     Insert a fit-in-fold Dirichlet / matrix-scaling calibration of EACH base's 3-prob vector
            BEFORE it enters the meta. Calibration fit on train-fold OOF only, applied to val+test.
hypothesis: per-base matrix-scaling fixes base-specific class-confusion miscalibration the single global
            LogReg meta can't disentangle, lifting beyond the maxed combiner ceiling.
target:     BA maximize · OOF CV > 0.970355 (by >2·sem to promote).

HOW: BASELINE-ASSERT FIRST (seconds) — reproduce the plain-OOF_mat LogReg @C=0.003 and confirm CV ==
0.970355 before reading any calibrated delta (global LogReg may already subsume per-base scaling → A/B only
meaningful vs exact baseline). Matrix-scaling = a small multinomial-LogReg per base on its 3 log-probs,
fit_in_fold (train-fold OOF only), or netcal DirichletCalibration if it installs cleanly. Then feed the
calibrated per-base log-probs into the SAME nested-C LogReg arm. A/B the calibrated CV, per-fold, sem.
Kill: OOF CV ≤ 0.970355 within sem = wash. Cheap (no base retraining; reuses saved OOF banks).
Outputs: oof.npy, test_probs.npy, submission.csv, train.log. Leak: calibration fit_in_fold.

## notes
well=exploit. Varies the base INPUT TRANSFORM into the meta — none of the closed combine-mechanism nodes
(n099 GBDT, n093 simplex, n100 FWLS, n080 TabPFN-3) touched that axis.
