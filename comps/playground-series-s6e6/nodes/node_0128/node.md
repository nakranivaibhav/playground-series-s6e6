---
id: node_0128
desc: z-local color-anomaly GBDT base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_zlocal_dcolor]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966698
sem: 0.000312
folds: [0.967492, 0.967369, 0.966515, 0.965977, 0.966135]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "GALAXY recall +0.0091 but STAR recall -0.0151; no blend benefit vs n091; err-corr vs n070=0.8136 (better than n091 vs n070=0.9408 but not <0.72)"
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T01:33Z
tags: [gbdt, lightgbm, z-conditional, color-anomaly, high-z-qso, data, draft, fs_zlocal_dcolor]
---

## plan
built on:   root — a NEW LightGBM base. Template: nodes/node_0030/src/solution.py. ONE change: add the
            fit_in_fold feature-set fs_zlocal_dcolor.
change:     fs_zlocal_dcolor (leak-safety: **fit_in_fold**): for each row, its colors RELATIVE to the
            median color of its REDSHIFT-NEIGHBOURHOOD. Bin redshift by TRAIN-FOLD quantile edges
            (e.g. 20 quantiles); within each z-bin compute the TRAIN-FOLD median of (u−g),(g−r),(r−i),
            (i−z); emit 4 deltas: color − median(color | zbin). Train one LightGBM on fs_realmlp_fe +
            these 4 deltas; A/B vs the same model without them. Bin edges + per-bin medians are fit on
            the TRAIN FOLD ONLY; val/test rows use the fold's frozen edges+medians.
hypothesis: every prior feature was ABSOLUTE color or a GLOBAL residual. A galaxy at z≈0.9 that is a
            color OUTLIER among objects at the SAME redshift is the physical QSO-impostor signature
            (the GAL→QSO error channel the proposer mapped at z≈0.91). This z-CONDITIONAL anomaly a tree
            cannot synthesize (it splits on raw z and raw color independently, never "color given local
            z"). Targets the high-z GAL↔QSO channel — the redshift regime never separately attacked.
target:     Balanced Accuracy maximize. Cheap-kill fold-0 < 0.965. GATE (validation.md): solo per-class
            check — GALAXY recall rises WITHOUT QSO recall dropping more (pred_diagnostic per-class +
            high-z band); err-corr vs n070 (<0.72 = finally decorrelated?); stack-add to n091 bootstrap
            P(>champ) ≥ 0.90. HONEST ODDS LOW (~10%): n086/n087 showed z-conditional color RESIDUALS
            don't decorrelate (0.70-0.72) — but those were residual-ONLY (info-starved); this is a small
            ADDITIVE block on the full rich FE targeting the specific high-z channel. Cheap GBDT run.

DATA well. READ: nodes/node_0030/src (template); data.md fs_realmlp_fe recipe; nodes/node_0086+0087
(prior z-conditional residual attempts — why they capped). fs_zlocal_dcolor = fit_in_fold (z-bin edges +
per-bin medians train-fold-only). CPU; minutes.

## notes
well = data. The one feature family not built: delta-from-local-z-neighbourhood, aimed at high-z GAL↔QSO.
