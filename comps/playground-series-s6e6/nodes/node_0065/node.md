---
id: node_0065
desc: confusion-zone mixup augmentation for TabM
op: improve
parents: [node_0033]
uses_data: [fs_realmlp_fe]
family: nn
status: buggy
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.968053
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: true, cv_too_good: false, passed: false}
gate_note: "Kill-switch tripped: fold-0 BA=0.961708 < 0.9675 threshold. Mixup degrades model (parent fold-0=0.968562). Full run aborted."
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: 2026-06-12T10:30Z
tags: [data-centric, augmentation, mixup, gpu]
---

## plan
built on:   node_0033 (TabM on rich fs_realmlp_fe, GPU ~10min). Everything byte-identical except the augmentation.
change:     In-fold mixup applied ONLY within the low-z (redshift ≤ 0.1) GALAXY/STAR confusion zone — intra-class
            AND cross-class mixup with soft labels, alpha~0.2, ~2x oversample of zone rows.
hypothesis: mixup densification of the confirmed low-z GALAXY/STAR boundary regularizes exactly where errors
            concentrate, lifting TabM where global tricks (class weights, GCE) failed.
target:     Balanced Accuracy maximize; beats parent if solo CV > 0.968053 (node_0033).

GPU — serialize behind other GPU nodes.

References to READ:
- node_0059/cleanlab proved noise/difficulty concentrates 2.30x in the low-z GALAXY/STAR zone, and PRUNING was
  the wrong response (model already robust). Augmentation attacks the same zone from the opposite direction:
  densify the decision boundary with virtual examples instead of removing rows. It is the one data-centric lever
  (favored well) NOT yet tried — the C-list covered prune/relabel/provenance/robust-loss/DAE, never augmentation.
- Parent src: `comps/playground-series-s6e6/nodes/node_0033/src` (TabM, fs_realmlp_fe — data.md recipe).

Implementation notes: mixup on the NUMERIC feature columns post-encoding (integer-floor categorical views — mix
the embedded/encoded representation or skip mixing categoricals, developer's call); zone mask computed per train
fold only (stateless threshold, no fit); val folds untouched and unweighted so OOF stays honest. A/B solo vs n33
0.968053; report per-class recalls (expect GALAXY/STAR recall to move). If solo improves ≥1 sem, restack into
bank-17. Kill: fold-0 solo < 0.9675.

## notes
Kill-switch tripped at fold-0. Zone mixup (alpha=0.2, 2x oversample, ~148k mixup rows from 74k zone rows)
produced fold-0 BA=0.961708 vs parent node_0033 fold-0=0.968562. Delta=-0.0069 (well below kill 0.9675).
Per-class recalls at fold-0: GALAXY=0.97470, QSO=0.96500, STAR=0.94542.
STAR recall dropped notably (parent STAR recall ~0.970+). The extra mixup gradient steps on the confusion
zone appear to destabilize the model rather than help — the two-pass training (hard then soft) doubles
effective steps per epoch and may be disrupting early-stopped convergence.
Timing: fold-0=202s, projected 5-fold=17min.
Leakage checks all clean before training (checks 1-6 pass). Full run not executed.
