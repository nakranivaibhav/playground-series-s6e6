---
id: node_0017
desc: blend OOF per-class threshold tune
op: improve
parents: [node_0010]
uses_data: []
family: ensemble
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966084
sem: 0.000177
folds: [0.966723, 0.965796, 0.966194, 0.965769, 0.965940]
baseline_cv: 0.965889
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.96702
submitted: 2026-06-06
created: 2026-06-06T12:18Z
decided: 2026-06-06T13:21Z
tags: [threshold, post-hoc, blend]
---

## plan
built on:   node_0010 (champion 4-arm combine n6/n4/n1/n9). The saved fold-honest
            OOF probability matrix and the weighted test_probs stay byte-identical;
            no model is retrained.
change:     On the champion blend's fold-honest OOF probability matrix (n6/n4/n1/n9
            weighted average), apply per-class weight (threshold) tuning: relabel
            argmax(prob * w) to maximize balanced accuracy. Tune the 3-vector w
            FOLD-HONEST — fit w on the other 4 folds, score on the held fold (the
            node_0002 protocol, but on the BLEND OOF instead of a single model).
            Apply the averaged w to the blended test probs. No retrain; reuse the
            saved OOF / test_probs.
hypothesis: argmax of the blended softmax isn't the balanced-accuracy optimum; the
            blended posterior may carry a small per-class bias that a per-class
            weight correction removes.
target:     balanced accuracy (maximize) · beats parent if fold-honest CV >
            node_0010 0.965889 beyond 2·sem.
