---
id: node_0036
desc: deep MLP on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965682
sem: 0.000324
folds: [0.966312, 0.966192, 0.964540, 0.965941, 0.965424]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good fires (warn only) — jump from 0.333 baseline is expected; dataset is highly structured with strong signal (redshift). All other nodes on this comp score similarly. Human eyeball confirms not a leak."
leak: clean
lb: null
submitted: null
created: 2026-06-07T14:57Z
decided: null
tags: [nn, mlp, cross-family-diversity, draft]
---

## plan
built on:   root (new draft, cross-family diversity). Template src to COPY from
            node_0028/src (keeps the fs_realmlp_fe FE pipeline + fold-honest OOF/test_probs
            scaffold over the FROZEN folds.json). The developer SWAPS the model only:
            replace the RealMLP/PBLD spec with a plain deep MLP.
change:     reuse the rich fs_realmlp_fe FE, then train a plain but well-regularized DEEP MLP —
            3-4 hidden layers (e.g. [512,512,256]) with batchnorm + dropout, AdamW optimizer,
            inputs standardized fit-in-fold (StandardScaler fit on train fold only), class-balanced
            loss (weighted CE). NO PBLD periodic embeddings, NO internal n_ens ensemble — a
            DIFFERENT NN inductive bias than RealMLP/TabM. GPU, fold-honest OOF (577347×3) +
            test_probs (247435×3) over the frozen folds.
hypothesis: a plain deep MLP carries a different inductive bias than RealMLP (PBLD embeddings)
            and TabM (internal ensemble), so its OOF errors should be de-correlated enough to add
            new signal the balanced-LogReg meta can exploit — even at lower solo strength.
target:     BA maximize; solo ≥ 0.96. Valued for DE-CORRELATION (stack lift via restack_probe.py),
            not solo strength.
