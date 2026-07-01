---
id: node_0073
desc: AutoGluon best_quality base on rich FE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: ensemble
status: dead
stage: decided
cv: null
gate_note: "FOLD-0 GATE KILL: AutoGluon best_quality WeightedEnsemble_L3=0.9575 / L2=0.9569 (internal bagged val BA) << kill 0.9675; ~0.011 below TabM n33 fold-0 0.9686. AutoGluon AutoML tops out far under our hand-tuned single models on this data. Full 5-fold NOT run; proc stopped to free GPU. autogluon-tabular left in lock (harmless)."
decided: 2026-06-12
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.968053
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [ensemble, automl, autogluon, outside, heavy]
---

## plan
built on:   root (new draft). One heavyweight AutoML family never tried.
change:     Train AutoGluon TabularPredictor (presets='best_quality', eval_metric='balanced_accuracy') per
            train fold on fs_realmlp_fe, harvest its internal multi-layer stack as ONE new base OOF; add-one
            restack into bank-17.
hypothesis: a heterogeneous AutoML stack as one base reaches ~0.969 solo with error structure unlike Deotte's
            hand-built bases, lifting the saturated bank.
target:     solo >= 0.969 and bank restack CV > 0.970153.

GPU HEAVY (~1.5-2h/fold, ~10h for 5 folds). `uv add autogluon.tabular` (library-first, rule 8) — verify it
does NOT break the working torch/cu128 build (it may pull conflicting deps; if so, install into an isolated
venv and run there, do NOT corrupt the main lock). SEQUENCING — FOLD-0 GATE FIRST: run fold-0 ALONE, and if
fold-0 val BA < 0.9675 (TabM n33 fold-0 0.9686 minus slack), STOP before launching folds 1-4. Only proceed to
the full 5-fold (overnight) if fold-0 clears.

Fit STRICTLY inside each frozen train fold (folds.json), predict val fold + test; AutoGluon's internal bagging
must operate on the train-fold rows ONLY. SELF-CHECK (reviewer ask): confirm the predictor was fit on exactly
the train-fold index set and that val-fold rows appear NOWHERE in its training data. fs_realmlp_fe recipe:
data.md + refs/realmlp-v5-for-s6e6.py (in src/clean.py). Restack via champion/src a1 scripts, add-one A/B vs
0.970153.

ORCHESTRATOR NOTE: heaviest node — run its fold-0 gate when GPU is otherwise idle; full run is overnight.

## notes
