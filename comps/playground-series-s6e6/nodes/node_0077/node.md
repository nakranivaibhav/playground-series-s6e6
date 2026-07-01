---
id: node_0077
desc: source new PRIMARY external OOF bases
op: draft
parents: [root]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970211
sem: 0.000251
folds: [0.971103, 0.970119, 0.969771, 0.969718, 0.970343]
baseline_cv: 0.970211
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CLEAN NEGATIVE: 0 new bases selected; cv=bank17+FT-T baseline. All sourced kernels exhausted: tabpfn3=meta-stacker, kospintr=submission-only, nn2/realmlp0=already in bank17; ravi_realmlp_v1 and ravi_ml_v1 are primary but correlated with existing bank bases (delta<=0)."
leak: clean
lb: null
submitted: null
created: 2026-06-13T09:13Z
decided: 2026-06-13T09:55Z
---

## plan
built on:   root (a fresh sourcing + forward-select draft); the forward-selection mechanics reuse node_0070's approach.
change:     Source NEW primary-model external OOF for s6e6 NOT already in the Deotte bank and NOT derivative meta-stackers — specifically a TabPFN-3-class base, a RealMLP-pytorch base, and an nn-v2 base if their authors published OOF — then forward-select onto bank17+FT-T (node_0070's set).
hypothesis: the Deotte-17 span the decorrelation frontier only at the FT-T accuracy tier; a primary base from a family the bank lacks (true TabPFN-3 primary, pytorch-RealMLP, nn-v2) could be the next FT-T-style real lift.
target:     Balanced Accuracy maximize; a selected base lifts bank17+FT-T CV > 0.970211; promotion-eligible if final CV > 0.970153 by > 2·sem.

The ONLY honest lever that has moved CV is a strong DECORRELATED PRIMARY external base: FT-Transformer added +0.000058 (n70). The lesson (n72) is that public OOF 'banks' scoring ~0.970 are derivative stacks of bank-17 and add nothing — only PRIMARY model OOF (pilkwang/ravi/Deotte) counts. refs/ already snapshots several primary kernels whose OOF we have NOT ingested: refs/pull_philippsinger_tabpfn-3-stacker, refs/ps-s6-e6-realmlp-pytorch.py, refs/nn-v2-for-s6e6.py, refs/pull_kospintr_stellar-catb-hgbc-xgb-lgbm-realmlp-baseline.

First check whether each ships an OOF array (n72 found TabPFN-3 published only a submission, no OOF — if so, that one is out). A sourcing pass should pull any author-shared OOF/test CSVs for these families. Then run node_0070's greedy forward-selection (READ nodes/node_0070/src) starting from bank17+FT-T, eps +0.00003. CRITICAL alignment: every OOF exactly 577347 id-ordered rows (reject any 731273-row externally-concatenated bank like lzsecurity), test 247435; verify each base's solo BA matches its report (not ~0.33) to confirm column order; folds.json; fold-honest. CPU.

well: outside.
