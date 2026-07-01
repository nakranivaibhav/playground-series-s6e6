---
id: node_0106
desc: STAR-gate hierarchical specialist base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966614
sem: null
folds: [0.966614]
baseline_cv: 0.970355
gates: {schema_ok: false, oof_full: false, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "CHEAP-KILL at fold-0: BA=0.9666 >= 0.965 (PASS) but err-corr vs node_0070=0.7960 >= 0.65 (FAIL). Hierarchical framing does NOT decorrelate from 72-base stack. No full OOF/submission produced — partial oof.npy fold-0 only."
leak: clean
lb: null
submitted: null
created: 2026-06-15T07:06Z
decided: 2026-06-15
---

## plan
built on:   root — fresh draft, NEW hierarchical FRAMING. Copy node_0033's TabM-on-fs_realmlp_fe recipe
            VERBATIM for BOTH heads (`nodes/node_0033/src/solution.py`); fs_realmlp_fe recipe in data.md.
change:     Stage-1 STAR-vs-nonSTAR head (redshift≈0 near-trivial); stage-2 GALAXY-vs-QSO TabM specialist
            trained ONLY on stage-1-predicted-nonSTAR rows. Multiply the two heads' probs → calibrated
            3-class output.
hypothesis: splitting off the trivially-separable STAR class first lets a focused GALAXY-vs-QSO specialist
            carve the dominant confusion zone differently than a monolithic softmax → strong decorrelated base.
target:     BA maximize · solo ≥0.965 AND err-corr vs node_0070 < 0.65; stack-add beats 0.970355.

HOW: structurally DIFFERENT from the symmetric OvR/chain decomps (n049/n050/n090, all washed at 0.72) —
the split follows PHYSICS (STAR z≈0). COMPLETE 3-class classifier producing a full prob vector → gate
honestly on CV+err-corr, NOT the n047 mirage rule. LEAK SELF-CHECK #1 (FIRST gate): stage-2's training rows
MUST come from per-fold stage-1 OOF predictions, NOT in-fold — assert per fold that the rows feeding stage-2
are defined by HELD-OUT stage-1 OOF (an in-fold stage-1 leak silently inflates stage-2). CHEAP-KILL: fold-0
solo BA < 0.965 OR err-corr ≥ 0.65 → STOP. Else full OOF + stack-add to n091.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log.

## notes
well=wildcard. GPU. fs_realmlp_fe = stateless, reuse as-is.
