---
id: node_0002
desc: per-class threshold tuning
op: improve
parents: [node_0001]
family: gbdt
uses_data: []
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964648
sem: 0.000049
folds: [0.964598, 0.964716, 0.964762, 0.964486, 0.964680]
baseline_cv: 0.964569
shuffled_cv: null
gates: {schema_ok: null, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T11:46Z
decided: 2026-06-05T11:52Z
tags: [postproc, threshold-calibration, weights-saved]
---

## plan
built on:   node_0001 — reuses its saved OOF probs, no model retrain.
change:     post-hoc per-class weight (threshold) tuning: relabel as argmax(prob*w) to
            maximize balanced accuracy; weights tuned FOLD-HONEST (on other 4 folds, scored
            on held fold) so the CV is not optimistically biased.
hypothesis: argmax of softmax is not the balanced-accuracy optimum; tuned thresholds add recall.
target:     beat node_0001 0.964569 beyond fold-noise.

## notes
DEAD-END LEVER: gain +0.000079 = +0.7σ, WITHIN noise. class_weight='balanced' already
calibrated the classes (optimal weights ≈ [0.9,1,1]). No submission. Final weights saved to
weights.json — revisit threshold calibration ON the blend, not the single model.
No submission.csv produced (schema_ok n/a) — a within-noise post-proc, kept as a record.
