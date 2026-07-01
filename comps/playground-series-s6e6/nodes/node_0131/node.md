---
id: node_0131
desc: per-class autoencoder recon-error-gap base
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe, fs_aerecon]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966598
sem: 0.000313
folds: [0.967382, 0.967305, 0.966364, 0.965940, 0.965999]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T12:10Z
tags: [gbdt, lightgbm, autoencoder, recon-error, synthetic-data, fit_in_fold, outside, improve, fs_aerecon]
---

## plan
built on:   node_0030 (LightGBM on fs_realmlp_fe, cv 0.966952). Copy src; add ONLY fs_aerecon.
change:     fs_aerecon (leak-safety: fit_in_fold) — per-class autoencoder reconstruction-error GAPS,
            exploiting that the data is SYNTHETIC. Per fold: train 1 tiny AE per class (GALAXY/QSO/STAR)
            on standardized (ugriz, redshift, derived colors) of the TRAIN-FOLD rows of THAT class only.
            For every row emit its 3-vector of recon errors {err_GAL, err_QSO, err_STAR} + the DIFFS
            (err_STAR−err_GAL, err_QSO−err_GAL, err_QSO−err_STAR) + argmin. FULL recipe in research.md ★3.
hypothesis: a deep tabular generator matches marginals but distorts local/joint structure; a per-class
            AE learns each class's manifold, and a low-z galaxy confusable with a star reconstructs almost
            as well under STAR-AE as GAL-AE → the GAP is a direct "which class-manifold am I closest to"
            signal a GBDT can split (and a difficulty/label-noise score for our 2 channels). This is a
            MANIFOLD-geometry axis from the synthetic nature — a lever the real-sky ceiling never had.
target:     Balanced Accuracy maximize. Cheap-kill fold-0 < 0.965. JUDGE (validation.md gate): solo BA
            vs n030; err-corr vs n070 (<0.72?); STACK-ADD to n091 (bootstrap P≥0.90); pred_diagnostic
            per-class (lift without the GALAXY-for-STAR trade?). fit_in_fold — the AEs fit on train-fold
            class rows ONLY; val/test rows transformed by the fitted AEs.

BUILD (research.md ★3): tiny AEs (~7 inputs → small bottleneck, e.g. [16,4,16]); libraries-first (torch
or sklearn MLPRegressor-as-AE); the value is in the cross-class recon GAPS, not raw errors, so standardize
inputs and report relative gaps. Source: Marks/Griffin/Corso 2024 arXiv:2412.02596. LEAK: each class-AE
fits on train-fold rows of that class only — NEVER on val/test or full train; verify by reading the fold
loop. READ: research.md ★3; nodes/node_0030/src; data.md fs_realmlp_fe. CPU/GPU light.

## notes
well = outside (synthetic-data well). Orthogonal manifold-geometry axis exploiting the generated data.
