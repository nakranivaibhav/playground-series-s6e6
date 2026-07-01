---
id: node_0076
desc: FT-T base + bagged argmax stack
op: combine
parents: [node_0070, node_0069]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970227
sem: 0.000244
folds: [0.971110, 0.970035, 0.969895, 0.969730, 0.970364]
baseline_cv: 0.970211
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv=0.970227 > parent node_0070 cv=0.970211 by +0.000016, within 2*sem=0.000488; modest improvement, no anomaly"
leak: clean
lb: null
submitted: null
created: 2026-06-13T09:13Z
decided: 2026-06-13T09:55Z
---

## plan
built on:   node_0070 (bank17+FT-T forward-selected set) + node_0069 (5-seed bag, argmax mechanics) — both kept byte-faithful where reused.
change:     Build ONE clean honest stack that fuses the two independently-LB-positive signals discovered last round: (a) bank-17 + FT-Transformer external base (node_0070), and (b) seed-bagged stacker with plain argmax, DROPPING the DE per-class threshold (node_0069). Reproduce on frozen folds; report CV/sem and a clean submission.
hypothesis: the three LB-positive levers are partly independent, so their combine should exceed node_0070's CV and give a robust, threshold-free champion candidate whose edge survives bagging (unlike the fragile DE-threshold).
target:     Balanced Accuracy maximize; beats node_0070 if CV > 0.970211; promotion-eligible vs champion if CV > 0.970153 by > 2·sem.

This is the consolidation the journal explicitly flagged as 'build next' (2026-06-12T17:56Z): node_0070 (bank17+FT-T) beat champion on BOTH CV (0.970211 vs 0.970153) and LB (0.97087 vs 0.97073), and node_0069 bagged-argmax beat the thresholded champion on LB (0.97079) — DE-threshold confirmed NOT load-bearing and single-seed-fragile (n69 lesson). Stack all three LB-positive moves: +FT-T base, 5-seed bag, argmax-no-threshold.

READ: nodes/node_0070/src (forward-select + bank merge already includes FT-T at refs/ext_oof/pilkwang_5090 ft_transformer_lite, id-aligned, cols proba_GALAXY/QSO/STAR→our 0/1/2 order), nodes/node_0069/node.md (5-seed bag mechanics; SEEDS 42..46 × folds, average OOF+test), champion/src a1_full_merge.py + a1_submit.py (bank-17 ingest, balanced multinomial LogReg on clipped log-probs). Use the SAME LogReg meta but average over 5 seeds and take plain argmax (no DE step). Verify every OOF is exactly 577347 id-ordered rows, test 247435 (ids 577347..824781), folds from folds.json. CPU, minutes; uv run --no-sync.

well: exploit.
