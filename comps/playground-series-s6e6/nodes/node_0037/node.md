---
id: node_0037
desc: multinomial LogReg on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: linear
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.939309
sem: 0.001202
folds: [0.937991, 0.938759, 0.944066, 0.937677, 0.938053]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good tripwire fires (gain=0.91 >= 0.9) but expected for this comp — all nodes in 0.93-0.97 range; warn only, not void."
leak: clean
lb: null
submitted: null
created: 2026-06-07T14:57Z
decided: null
tags: [linear, logreg, cross-family-diversity, draft]
---

## plan
built on:   root (new draft, cross-family diversity). Template src to COPY from
            node_0028/src (keeps the fs_realmlp_fe FE pipeline + fold-honest OOF/test_probs
            scaffold over the FROZEN folds.json). The developer SWAPS the model only:
            replace the RealMLP with a multinomial LogisticRegression.
change:     reuse the rich fs_realmlp_fe FE (esp. the color pairs, redshift ratios, and any
            TargetEncoder outputs), then fit a multinomial LogisticRegression (saga or lbfgs
            solver, multi_class='multinomial', class_weight='balanced'), with inputs standardized
            fit-in-fold (StandardScaler fit on train fold only). CPU, fast. Fold-honest OOF
            (577347×3) + test_probs (247435×3) over the frozen folds.
hypothesis: a linear base is maximally de-correlated from the tree/NN bases — the public 0.97105
            reference includes a logreg base for exactly this reason. Even at modest solo strength
            its orthogonal errors should give the meta-stacker new exploitable signal.
target:     BA maximize; solo ≥ 0.95. Valued purely for STACK DIVERSITY (restack_probe.py lift),
            not solo strength.
