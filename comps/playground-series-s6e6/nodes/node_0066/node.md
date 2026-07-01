---
id: node_0066
desc: RealMLP pretrain on real SDSS17, finetune
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968560
sem: 0.000286
folds: [0.969678, 0.968065, 0.968282, 0.968452, 0.968320]
baseline_cv: 0.969065
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv 0.968560 < parent 0.969065 — does NOT beat parent; only 9 hidden-layer tensors transferred (PBLD dims mismatch)"
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: 2026-06-12
tags: [data-centric, transfer-learning, sdss17-pretrain, gpu]
---

## plan
built on:   node_0028 (RealMLP-ref recipe FE+PBLD, ~5.9min/run GPU). Everything byte-identical except a
            pretrain stage before each fold's fit.
change:     Two-stage transfer: per fold, PRETRAIN the n28 RealMLP-ref on the cleaned original SDSS17 100k rows
            (drop -9999 placeholder rows), then FINE-TUNE on the train fold at a lower LR (~0.1–0.3× n28's).
hypothesis: real-catalog pretraining gives the RealMLP a better-conditioned representation of the true
            astrophysical manifold than random init, lifting it past 0.9691 where concat/priors failed.
target:     Balanced Accuracy maximize; beats parent if solo CV > 0.969065 (node_0028); promotable if bank
            restack > 0.970153.

GPU — serialize behind other GPU nodes.

CRITICAL — this is NOT a re-run of dead levers:
- 2026-06-08T13:34Z killed CONCATENATING SDSS17 into the train side (adversarial AUC 0.909, −0.001 on a GBDT).
- node_0045 killed orig-PRIOR FEATURES.
- Pretraining/transfer is the untested THIRD form — drift hurts when real rows share the loss with synthetic
  ones, but as an INITIALIZATION it only shapes representations. MEMORY/idea-wells favor synthetic-data pretrain
  plays in this regime (here inverted: pretrain on real, fine-tune on the synthetic target distribution).

References to READ:
- Parent src: `comps/playground-series-s6e6/nodes/node_0028/src`; FE recipe `refs/realmlp-v5-for-s6e6.py`
  (fs_realmlp_fe, data.md row, stateless).
- Original SDSS17 csv + cleaning/loader: reuse the node_0044/node_0045 src (orig-prior builders).

Fit-in-fold discipline: the orig data is external and label-complete, fine to use WHOLE in every fold's pretrain;
the in-fold TargetEncoder stays fold-local as in n28; val fold never seen in pretrain OR finetune. A/B solo vs
n28 0.969065; if solo ≥ n28 also re-stack into bank-17 (swap-for-nearest-realmlp and +18th) per the standard
restack probe. Kill: fold-0 solo < 0.9685.

## notes
