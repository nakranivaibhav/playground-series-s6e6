---
id: node_0102
desc: Saerens EM test-prior correction
op: improve
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970233
sem: 0.000283
folds: [0.971197, 0.969868, 0.969755, 0.969783, 0.970563]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "LEVER MOOT (despite apparent 'LIVE' flag). EM OOF CV 0.970233 vs champ 0.970355 = −0.000121 (hurts). The EM test-prior estimate [G 0.623, Q 0.212, S 0.165] looks materially shifted from train [0.654, 0.203, 0.143] (STAR +14.9% rel), BUT the OOF folds are the CONTROL: they are stratified so their TRUE prior == train prior, yet EM estimates the SAME prior [~0.623, ~0.213, ~0.165] on every held-out fold. So the apparent 'shift' is EM-ESTIMATOR BIAS on n091's probability geometry, NOT real label shift — confirmed by the negative OOF delta (correction hurts when there is no real shift). Do NOT submit the EM-corrected test (643 test argmax flips would be spurious). Prior-correction axis fully closed: train-prior multipliers wash (n017/069/078/089), and target-prior EM is an estimator artifact here."
leak: clean
lb: null
submitted: null
created: 2026-06-15T07:06Z
decided: 2026-06-15
---

## plan
built on:   champion n091 stacked probabilities (champion/oof.npy, champion/test_probs.npy — n×3).
change:     Post-process with Saerens-Latinne-Decaestecker EM label-shift: iteratively re-estimate
            the test class prior from the unlabeled test probs and rescale posteriors, then argmax.
hypothesis: the synthetic test prior differs subtly from train; EM corrects the operating point where
            OOF-multiplier threshold search (sees only train prior) cannot.
target:     BA maximize · beats parent if OOF CV > 0.970355 OR EM test-prior shifts materially from train.

HOW: load champion/oof.npy + champion/test_probs.npy + train labels (data/train.csv 'class'). Saerens
fixed-point EM (~20 lines): p_new(y) ∝ mean_x [ p(y|x)·p_new(y)/p_train(y) ] / Z, iterate to convergence;
rescale posteriors, argmax. HONEST eval: the stratified OOF folds carry NO prior shift, so the OOF delta
likely reads ~0 — the DECISIVE signal is the EM-estimated TEST prior vector vs the train prior. LOG both.
Kill: if test prior within 1% of train on every class, lever is moot → drop. If it shifts materially, the
EM-corrected test submission is the candidate even when OOF≈0 (surface for an LB probe, do not auto-submit).
Outputs: oof.npy (EM-corrected OOF), test_probs.npy (EM-corrected), submission.csv, train.log w/ prior vectors.

## notes
well=outside. Cheap (no training; saved stacked probs only).
