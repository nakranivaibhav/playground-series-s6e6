---
id: node_0048
desc: Optuna XGB to stacked-OOF objective
op: improve
parents: [node_0031]
uses_data: [fs_realmlp_fe]
family: gbdt
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969762
sem: null
folds: null
baseline_cv: 0.969712
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
decided: 2026-06-09
leak: null
lb: null
submitted: null
created: 2026-06-08T16:11Z
decided: null
tags: [optuna, hpo, stack-objective, decorrelation]
---

## plan
built on:   node_0031 (XGBoost on fs_realmlp_fe, cv 0.966244) — re-tune it.
change:     Optuna HPO of XGBoost on fs_realmlp_fe, but the objective is the STACKED-OOF
            balanced accuracy when this base REPLACES node_0031 in CORE15 — NOT the solo
            base CV (our lesson: solo gains wash if correlation is unchanged). Search depth/
            eta/subsample/colsample/min_child/gamma/reg, ~40-60 trials, time-boxed. Each
            trial: 5-fold train on OUR folds → restack CORE15 (n31→trial) → balanced-acc.
hypothesis: Tuning toward the stack objective can find a config that is stronger AND less
            correlated (e.g. heavier column subsampling) than node_0031, lifting the stack
            where a solo-tuned base washed.
target:     best trial's re-stack beats champion 0.969808 by >2·sem.

## notes
Add optuna via `uv add optuna` if missing. Keep folds frozen (folds.json). Emit the best
config's OOF + test_probs + the saved params. Cache the CORE15 OOF load once; reuse the
restack helper. Self-gate leakage on the final chosen base (TE/encoders fit in-fold).
Honest EV: modest — the stack is saturation-limited; report the trial curve regardless.

## result — WASH (run cut at trial 13/27, but verdict unambiguous)
Across 14 diverse XGB configs the SOLO base CV swung 0.96580→0.96830, yet the STACKED-OOF
balanced accuracy never moved: baseline (un-tuned n31 in CORE15) = 0.969712; best trial
(Trial 1) = 0.969762 = **+0.00005, pure noise**, and it never even reached champion 0.969808.
Confirms the saturated-stack lesson AGAIN: tuning a base that the stack already de-correlates
washes — the 16th XGB variant adds nothing the existing 15 bases don't span. Optuna-to-stack-
objective DEAD for this base set. Run killed mid-trial-13 (session interruption); not restarted
because the curve was flat across all 14 completed trials. No artifacts promoted.
