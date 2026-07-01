---
id: node_0061
desc: GCE robust-loss TabM (C8)
op: improve
parents: [node_0033]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966788
sem: null
folds: [0.967317, 0.966996, 0.966754, 0.966373, 0.966498]
baseline_cv: 0.968053
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-10
decided: 2026-06-10
tags: [data-centric, robust-loss, GCE, null]
---

## plan
built on:   node_0033 TabM-richFE, byte-identical except the loss.
change:     CrossEntropy -> Generalized Cross-Entropy (q=0.7), bounded loss robust to label noise (Zhang & Sabuncu 2018), class weights kept.
hypothesis: a bounded loss stops TabM chasing the generator's irreducible overlap noise in the confusion zones.
target:     beat champion 0.969808 via a restack.

## notes
Solo −0.001265 (all folds lower). Robust-loss family dead: on synthetic data train+test share noise, so CE's posterior is what balanced-acc scores; GCE underfits + mis-calibrates the meta's log-prob inputs. No restack.
