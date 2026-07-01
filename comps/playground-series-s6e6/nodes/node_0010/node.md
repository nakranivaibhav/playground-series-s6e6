---
id: node_0010
desc: combine blend +TabM (n6+n4+n1+n9)
op: combine
parents: [node_0006, node_0004, node_0001, node_0009]
family: ensemble
uses_data: []
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965889
sem: 0.000141
folds: [0.966421, 0.965800, 0.965836, 0.965576, 0.965811]
baseline_cv: 0.965530
shuffled_cv: 0.33298
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.96704
submitted: 2026-06-05T15:43Z
created: 2026-06-05T15:26Z
decided: 2026-06-05T15:28Z
tags: [ensemble, blend, tabm, fold-honest, weighted-probability-average]
---

## plan
built on:   node_0007's blend (n6+n4+n1) + the new TabM arm node_0009. Same fold-honest
            nested weighted-probability-average protocol; uses the four arms' saved OOF +
            test probability matrices byte-identically (no model retrain).
change:     add TabM (node_0009) as a 4th blend arm. It is GBDT-strength solo (0.964215) AND
            de-correlated (err-corr ~0.82 vs trees) — the diagnostic showed the 4-arm honest
            CV = 0.965889 ± 0.000141, beating champion node_0007 (0.965530) by +0.000359 (>2·sem),
            with TabM earning the largest weight (~0.35). node_0008 (plain MLP) is excluded —
            it earns 0 weight.
hypothesis: a strong de-correlated NN arm corrects boundary cases all the GBDTs share.
target:     beat champion node_0007 0.965530 beyond 2·sem (diagnostic OOF ≈ 0.965889).

## notes
WINNER → CHAMPION. Honest nested CV 0.965889 ± 0.000141 = +0.000359 vs node_0007 (>2·sem).
Final weights n6:0.25 / n4:0.15 / n1:0.25 / n9:0.35 — TabM (node_0009) earns the LARGEST
weight, confirming a strong de-correlated NN arm was the missing lever (the user's NN push
paid off). Per-fold weights stable (n9 ∈ 0.30–0.35 every fold). Shuffled control 0.33298 ≈ 1/3.
967 test rows flip vs node_0007. node_0008 (plain MLP) excluded — earned 0 weight. The submit
candidate (beats last-submitted CV node_0007 0.965530 by +0.000359 > 2·sem).
