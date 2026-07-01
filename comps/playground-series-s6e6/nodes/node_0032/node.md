---
id: node_0032
desc: RealMLP-ref seed-2 (bag)
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969119
sem: 0.000274
folds: [0.970197, 0.968669, 0.968875, 0.968873, 0.968982]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good is a warn only (same as node_0028, expected for this competition); all error-severity checks passed"
leak: clean
lb: null
submitted: null
created: 2026-06-07T12:54Z
decided: null
tags: [nn, realmlp, seed-bag, fs_realmlp_fe, improve]
---

## plan
built on:   node_0028 (RealMLP reference recipe, cv 0.969065) — keep its src BYTE-IDENTICAL.
            Template src to COPY from node_0028/src; the FE (fs_realmlp_fe), the model
            architecture, n_ens, embeddings, preprocessing, and the fold-honest OOF scaffold all
            stay unchanged.
change:     re-run node_0028's EXACT recipe with a DIFFERENT random seed (the only edit) — the
            architecture and FE are byte-identical. Produces a 2nd RealMLP OOF/test_probs over
            the frozen folds.json that bags with node_0028; averaging the two seeds' probabilities
            reduces NN-init variance.
hypothesis: node_0028's RealMLP carries init/training variance; a 2nd seed averaged with it lowers
            that variance and may add a touch of accuracy — and a lower-variance RealMLP base can
            only help the stack.
target:     BA maximize; solo ≈ node_0028 (0.969065); the 2-seed average should be ≥ node_0028 and
            valuable if it lifts the re-stack vs champion node_0029 (0.969205).
