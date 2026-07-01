---
id: node_0052
desc: re-stack discarded OOFs into CORE15
op: combine
parents: [node_0041, node_0042, node_0043, node_0049, node_0050, node_0011]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969808
sem: 0.000279
folds: [0.970818, 0.969226, 0.969580, 0.969454, 0.969963]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [ensemble, re-stack, combine, stack]
---

## plan
built on:   node_0041 (CORE+CatBoost 15-base champion stack, cv 0.969808) — its
            meta-learner + DE-threshold pipeline stays byte-identical.
change:     Re-fit the balanced-LogReg meta on the CORE15 OOF set AUGMENTED with the
            previously-discarded base OOFs: node_0042 (RealMLP config-B), node_0043
            (CatBoost config-B), node_0049 (asymmetric binary chain), node_0050 (OvR
            3-binary), node_0011 (XGBoost full feats). Each was solo-valid but judged
            stack-neutral/washed individually; test whether they JOINTLY lift the
            meta now that diversity is the binding constraint. uses_data=[] — consumes
            OOF, not feature-sets (the combine edges in graph.md).
hypothesis: each candidate washed in isolation, but config-B variants + the binary-
            reframing OOFs carry de-correlated error structure the meta can exploit
            only when added together; a joint re-stack may clear the noise bar.
target:     Balanced Accuracy (maximize). Beats parent if CV > 0.969808 by more than
            ~2·sem (the established promote bar); else null result, champion stands.

## notes
RESULT: No lift found. All 4 eligible candidates hurt the stack. Best config is CORE15 alone (0.969808).

NOTE: node_0011 (XGBoost full feats) is already in CORE15, so only 4 candidates were tested.

Full sweep table (fold-honest balanced-LogReg meta + DE threshold, 5-fold):

| Config                  |     CV   |   ±SEM   |    ΔCV   |
|-------------------------|----------|----------|----------|
| CORE15 (baseline)       | 0.969808 | 0.000279 | +0.000000 |
| CORE15 + node_0043      | 0.969782 | 0.000252 | -0.000027 |
| CORE15 + node_0050      | 0.969781 | 0.000292 | -0.000027 |
| CORE15 + node_0049      | 0.969765 | 0.000316 | -0.000043 |
| CORE15 + node_0042      | 0.969688 | 0.000264 | -0.000120 |

No candidate cleared ~2·sem (~0.0005) lift bar. Greedy forward selection skipped (no individually-positive candidates).

Artifacts emitted for CORE15 baseline config (since no candidate lifted). The hypothesis is rejected.
Leakage: bases were individually gated when built; this node does no .fit() on raw data (pure OOF arithmetic); leakage_scan.py exit=0.
