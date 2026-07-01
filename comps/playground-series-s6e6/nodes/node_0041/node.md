---
id: node_0041
desc: CORE+CatBoost 15-base stack + DE thresh
op: combine
parents: [node_0006, node_0004, node_0001, node_0009, node_0011, node_0003, node_0019, node_0016, node_0014, node_0028, node_0032, node_0035, node_0033, node_0030, node_0039]
uses_data: []
family: ensemble
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969808
sem: 0.000279
folds: [0.970818, 0.969226, 0.969580, 0.969454, 0.969963]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: null
leak: clean
lb: 0.97043
submitted: 2026-06-07T16:41Z
created: 2026-06-07T16:40Z
decided: 2026-06-07T16:40Z
tags: [stack, champion, core, catboost]
---

## plan
built on:   node_0040 CORE stack (14 bases) + node_0039 (CatBoost on rich FE, memory-fixed, 0.96772) added as a 15th base.
change:     add CatBoost-richFE OOF/test log-probs to the CORE stack; balanced multinomial LogReg meta + DE threshold.
hypothesis: the strongest GBDT (CatBoost 0.9677, de-correlated from LightGBM) lifts CORE where weak diverse bases didn't.
target:     BA maximize · cv 0.969808 (+0.00028 vs CORE node_0040; +0.000603 vs last-submitted node_0029 0.969205, >2sem → SUBMIT).

## notes
Best stack of the session. CatBoost is the one strong base that lifted CORE (weak MLP/LogReg/ExtraTrees all hurt). Submit-worthy: beats last-submitted CV by >2sem.
