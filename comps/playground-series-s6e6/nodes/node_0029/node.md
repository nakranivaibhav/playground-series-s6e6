---
id: node_0029
desc: 10-base stack (champ9 + RealMLP-ref) + DE thresh
op: combine
parents: [node_0006, node_0004, node_0001, node_0009, node_0011, node_0003, node_0019, node_0016, node_0014, node_0028]
uses_data: []
family: ensemble
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969205
sem: 0.000360
folds: [0.970570, 0.968510, 0.968996, 0.968755, 0.969193]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: null
leak: clean
lb: 0.96993
submitted: 2026-06-07T12:57Z
created: 2026-06-07T12:50Z
decided: 2026-06-07T12:50Z
tags: [stack, breakthrough, realmlp-ref]
---

## plan
built on:   the 9-base champion stack (node_0020) byte-identical, + node_0028 (RealMLP reference recipe, cv 0.969065) added as a 10th base.
change:     add node_0028's OOF/test log-probs as a stack column; balanced multinomial LogReg meta + DE per-class threshold, fold-honest.
hypothesis: node_0028 (a properly-built RealMLP, +0.020 over our broken one) is the strong de-correlated base the stack was missing.
target:     BA maximize · beats champion node_0020 (0.966627) — achieved cv 0.969205 (+0.002578, ~7·sem).

## notes
The breakthrough round. Our prior RealMLP (node_0024) was under-built (bare pytabkit-TD, 0.949); the public 0.97105 meta-stacker revealed the gap. node_0028 ports the proven reference recipe (heavy stateless FE + hand-rolled RealMLP w/ PBLD embeddings + fit-in-fold TargetEncoder) → 0.969065 solo, beating the old champion stack alone. Re-stack: champ9+n28 = 0.969205 (best); adding tabicl/cat on top is redundant (n28 dominates).
