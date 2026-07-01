---
id: node_0079
desc: honest disjoint-teacher pseudo-label TabM
op: improve
parents: [node_0033]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.967406
sem: 0.000354
folds: [0.968725, 0.967408, 0.966898, 0.967293, 0.966706]
baseline_cv: 0.968053
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CV 0.967406 is BELOW parent n33 0.968053 (-0.000647). Disjoint teacher did NOT lift — regression vs parent. n74 A4-clout teacher got +0.000475; this honest teacher got -0.000647. Teacher quality difference (A4 public LB=0.971 vs external bank ~0.968) may explain the gap. Node is valid but does not beat parent."
leak: clean
lb: null
submitted: null
created: 2026-06-13T09:13Z
decided: 2026-06-13T09:55Z
---

## plan
built on:   node_0033 (TabM on rich fs_realmlp_fe) — kept; adds disjoint-teacher pseudo-labeled test rows to each fold's train.
change:     Retrain TabM-richFE on each train fold augmented with test rows hard-labeled by a DISJOINT EXTERNAL OOF bank's argmax (e.g. pilkwang/ravi primary bases, NOT our stack and NOT the A4 public-vote consensus) at ~0.5 sample weight; honest OOF on true train labels; restack into bank17+FT-T.
hypothesis: an honest disjoint-teacher imports external-cluster knowledge as a base with errors genuinely different from bank-17 (n74 confirmed the solo lift is real), and being clout-free it can promote where n74 could not.
target:     Balanced Accuracy maximize; solo > n33 0.968053 (replicate n74's +0.0004) AND restack bank17+FT-T CV > 0.970211.

node_0074 proved a DISJOINT teacher gives a REAL solo lift (+0.000475 vs n33) where n67's self-distillation washed (teacher==own bank) — but n74 used the A4 PUBLIC-VOTE consensus as teacher, which is CLOUT-quarantined (finals slot-2 only, can never promote). This proposal does the SAME mechanism with an HONEST teacher: hard-labels from a disjoint EXTERNAL primary OOF bank (refs/ext_oof/pilkwang_5090 + ravi_gnn_mlv1 test predictions, argmax consensus), which carries NO public-LB-derived provenance → the resulting base is promotion-eligible.

READ: nodes/node_0074/node.md (mechanism, sample-weight ~0.5, fold-0 kill BA<0.9675, restack pattern), nodes/node_0033/src (TabM-richFE parent), data.md fs_realmlp_fe recipe, refs/ext_oof for teacher test predictions. Fold-honest: pseudo-labels from test rows only, never train labels or val folds; OOF scored only on true train labels. Restack via champion/src a1 scripts onto bank17+FT-T (node_0070 set). GPU ~1h.

well: outside.
