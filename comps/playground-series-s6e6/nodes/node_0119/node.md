---
id: node_0119
desc: synthetic generative pretrain then finetune TabM
op: draft
parents: [root]
uses_data: [fs_synthpre]
family: nn
status: dead
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: true, cv_too_good: null, passed: false}
gate_note: "KILL GATE 2 tripped: pretrain fold-0 BA=0.967934 < cold-start fold-0 BA=0.968562; pretrain hurts vs cold TabM; full run skipped per plan"
leak: clean
lb: null
submitted: null
created: 2026-06-16T13:01Z
decided: 2026-06-17T09:37Z
tags: [nn, tabm, synthetic-pretrain, generator, wildcard, draft, fs_synthpre]
---

## plan
built on:   root — a NEW TabM base. Template src to copy verbatim: nodes/node_0033/src/solution.py (the strongest TabM-on-richFE loop, cv 0.968053). The change ADDS a pretrain phase: fit a per-class tabular generator on the train fold, sample synthetic rows, pretrain TabM on them, then fine-tune on the real fold. Uses new artifact fs_synthpre.
change:     Fit a fast per-class tabular generator (Gaussian-copula or a small CTGAN/TVAE) on the TRAIN fold only, sample ~2-4M synthetic labelled rows, PRE-TRAIN a TabM on them, then FINE-TUNE on the real fold. New artifact fs_synthpre (leak-safety: fit_in_fold — the generator is fit on train-fold rows only, sampled rows feed pretraining only, val/test never generated). One coupled hypothesis (wildcard license): generator + pretrain together.
hypothesis: pretraining a NN on millions of rows from a class-conditional generator fit to the train fold regularizes the decision surface toward the true manifold, lifting solo BA and/or decorrelating from the bank trained only on the raw noisy rows.
target:     Balanced Accuracy maximize; cheap-kill if fold-0 fine-tuned solo BA < 0.962 or pretrain hurts vs cold TabM; valuable if solo ≥0.965 AND stack-add to n091 > 0.970355 by > 2·sem.

WILDCARD bundling the favored DATA well's 'synthetic data and synthetic PRE-TRAINING' lever, which has NEVER been tried here and is structurally distinct from every closed augmentation: n065 mixup (tanks STAR recall), n062 swap-noise DAE (26-dim has no within-row manifold), barbagrande jitter-aug (research.md 06-14, exhausted). Those all PERTURB existing rows; a fit generator SAMPLES NEW rows from the learned class-conditional density, a genuinely different inductive bias — pretraining a NN on a smooth generative prior before fine-tuning on the real (noisy, synthetic-generator) data can regularize the decision surface toward the true class manifold. READ: nodes/node_0033/node.md + src (the strongest TabM-on-richFE loop, CV 0.968053 — copy verbatim, add a pretrain phase); data.md fs_realmlp_fe recipe; research.md note that the comp data is itself synthetic (so a copula/CTGAN fit on train approximates the SAME generator family). LIBRARIES FIRST: use sdv (GaussianCopulaSynthesizer / CTGANSynthesizer) or ctgan via uv add — do NOT hand-roll a generator. KILL CRITERIA (cheap, this is long-training): (1) generator fit + 2M-row sample must complete in < ~20 min on fold-0 or downscale the sample; (2) fold-0 fine-tuned solo BA must reach ≥0.965 (cheap-kill < 0.962) — if pretraining HURTS vs cold-start TabM, stop. LEAK: generator fit train-fold-only; verify no val/test row is ever generated or seen by the generator. fs_synthpre = fit_in_fold. If it clears fold-0, full 5-fold + err-corr + stack-add to n091.

## notes
well = wildcard.
