---
id: node_0035
desc: RealMLP-ref seed-3 (bag)
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968971
sem: 0.000249
folds: [0.969955, 0.968580, 0.968741, 0.968760, 0.968820]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn fires (expected — baseline=0.333, warn is structural vs random chance; same as n28/n32). Severity: warn only, not a blocker."
leak: clean
lb: null
submitted: null
created: 2026-06-07T13:50Z
decided: null
tags: [nn, realmlp, seed-bag, fs_realmlp_fe, improve, gpu]
---

## plan
built on:   node_0028 (RealMLP reference recipe, cv 0.969065) — keep its src BYTE-IDENTICAL.
            Template src to COPY from node_0028/src; the FE (fs_realmlp_fe), the model
            architecture, n_ens, embeddings, preprocessing, and the fold-honest OOF scaffold
            all stay unchanged. node_0032 is the seed-2 bag partner.
change:     re-run node_0028's EXACT recipe with a 3rd DIFFERENT random seed (the only edit)
            — architecture and FE byte-identical. Produces a 3rd RealMLP OOF/test_probs over
            the frozen folds.json that bags with node_0028 (seed-1) + node_0032 (seed-2) into
            a 3-seed RealMLP bag; averaging three seeds' probabilities further reduces
            NN-init variance. GPU.
hypothesis: the 2-seed RealMLP bag (n28+n32) still carries init variance; a 3rd seed averaged
            with them lowers variance further and a lower-variance RealMLP base can only help
            the stack.
target:     BA maximize; solo ≈ node_0028 (0.969); the 3-seed average should be ≥ the 2-seed
            bag and valuable if it lifts the re-stack vs champion node_0029 (0.969205).
