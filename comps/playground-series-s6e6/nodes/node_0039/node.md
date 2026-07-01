---
id: node_0039
desc: CatBoost on rich FE (memory-fixed debug of n34)
op: debug
parents: [node_0034]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.967723
sem: 0.000205
folds: [0.968361, 0.967691, 0.967432, 0.967180, 0.967951]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn is expected for this comp (redshift is a strong physical separator; same warn on every prior node). Peak RSS 3.32 GB vs 29.5 GB OOM in parent."
leak: clean
lb: null
submitted: null
created: 2026-06-07T15:35Z
decided: null
tags: [gbdt, catboost, debug, oom-fix]
---

## plan
built on:   node_0034 (CatBoost on fs_realmlp_fe) which OOM-killed on fold 4 (29.5GB RSS, depth=8 border_count=254) after 4 strong folds (~0.9684).
change:     reduce memory — border_count 254→128 and depth 8→6 (per the OOM post-mortem) so all 5 folds complete under the 30GB RAM limit. Everything else identical.
hypothesis: CatBoost on the rich FE is a strong de-correlated GBDT (~0.968) that may lift the stack beyond LightGBM alone.
target:     BA maximize · solo ≥ 0.966 · valuable if it lifts the re-stack vs champion node_0029 (0.969205).

## notes
Run completed cleanly in 39.6 min. Peak RSS stayed at 3.32 GB (vs 29.5 GB OOM in parent).
Per-fold: fold0=0.968361 (best_iter=978), fold1=0.967691 (best_iter=1004),
fold2=0.967432 (best_iter=1378), fold3=0.967180 (best_iter=916), fold4=0.967951 (best_iter=1455).
5-fold mean cv=0.967723 ± 0.000205.
The memory reduction (border_count 254→128, depth 8→6) costs ~0.0005 BA vs node_0034's 4-fold
partial mean of 0.968378, but completes all 5 folds — valid and within target ≥ 0.966.
All 5 gates passed. Leakage scan exit=0. dist_sane: GALAXY 63.4% / QSO 20.8% / STAR 15.8%.
