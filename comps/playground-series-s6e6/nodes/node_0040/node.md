---
id: node_0040
desc: CORE stack (champ9 + 3 RealMLP + TabM + LightGBM-fe) + DE thresh
op: combine
parents: [node_0006, node_0004, node_0001, node_0009, node_0011, node_0003, node_0019, node_0016, node_0014, node_0028, node_0032, node_0035, node_0033, node_0030]
uses_data: []
family: ensemble
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969529
sem: 0.000327
folds: [0.970794, 0.969120, 0.968987, 0.969257, 0.969487]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: best-CV node; +0.00032 (~1sem) over node_0029 — sub-2sem so NOT submitted (LB noise-limited). Finals candidate.
leak: clean
lb: null
submitted: null
created: 2026-06-07T15:49Z
decided: 2026-06-07T15:49Z
tags: [stack, champion-by-cv, core, finals-candidate]
---

## plan
built on:   the node_0029 stacker, base set narrowed/extended to the empirically-best CORE = champ9 + 3 RealMLP seeds (n28/n32/n35) + TabM-richFE (n33) + LightGBM-richFE (n30). 14 bases.
change:     balanced multinomial LogReg meta on base OOF log-probs + DE per-class threshold, fold-honest.
hypothesis: the 3-seed RealMLP bag + the two strong de-correlated richFE bases (TabM, LightGBM) are the best stack; weak diverse bases (MLP/LogReg/ExtraTrees) were proven to HURT.
target:     BA maximize · cv 0.969529 (best CV; +0.00032 vs node_0029's 0.969205, within 1sem — finals pick).

## notes
Best-CV node of the session. node_0029 (cv 0.969205, LB 0.96993) remains the last LB-confirmed submission; node_0040 is +1sem better CV but sub-2sem so not worth a submission on the noise-limited LB. CatBoost (n39) may extend this.
