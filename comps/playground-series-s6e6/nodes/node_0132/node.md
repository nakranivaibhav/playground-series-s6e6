---
id: node_0132
desc: z-gated QSO bump-excess GBDT base
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe, fs_sedshape]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966675
sem: 0.000317
folds: [0.967636, 0.967223, 0.966310, 0.966093, 0.966115]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T12:05Z
tags: [gbdt, lightgbm, sed-shape, mgii-bump, high-z-qso, z-gated, outside, improve, fs_sedshape]
---

## plan
built on:   node_0030 (LightGBM on fs_realmlp_fe, cv 0.966952). Copy src; add ONLY fs_sedshape.
change:     fs_sedshape (leak-safety: STATELESS) — z-gated continuum-relative SED-shape, surgical on the
            high-z GALAXY↔QSO channel. DISTINCT from our z-conditional color RESIDUAL (n128, which subtracted
            the POPULATION-median color in a z-bin): this subtracts each object's OWN log-flux continuum
            interpolated at the physical MgII line position. Feature: bump_excess = mag_r − (w·mag_g +
            (1−w)·mag_i), w≈0.55 (= r minus the continuum interpolated from non-line neighbours g,i — a
            sign-definite QSO marker). z-GATED form: band B nearest 2800·(1+z), emit mag_B −
            continuum_interp(neighbours of B), so the rest-2800Å complex moving g→r→i over z=0.45→1.5 reads
            as one clean axis. Optionally also the D4000 break proxy (★2b). FULL recipe in research.md ★2a.
hypothesis: a quasar carries MgII λ2800 + the small-blue-bump; as z slides it across the filters, QSO
            colours make non-monotonic excursions a galaxy's 4000Å ABSORPTION break cannot reproduce
            (opposite sign). At z≈0.9 the rest-2800Å lands in r → bump_excess is sign-definite for QSOs,
            a per-object SED-line measurement no single color encodes. Targets the z≈0.9 GAL↔QSO channel.
target:     Balanced Accuracy maximize. Cheap-kill fold-0 < 0.965. JUDGE (validation.md gate): solo BA vs
            n030; err-corr vs n070; STACK-ADD to n091 (bootstrap P≥0.90); pred_diagnostic — QSO recall up
            WITHOUT GALAXY dropping more (per-class + high-z band). HONEST prior: cheapest of the 3, surgical;
            broadband smears the line so it may be noisy — but z-gating the band is what makes it usable.

BUILD (research.md ★2a): pure stateless formula from raw ugriz + z (log-flux continuum interpolation +
the z-gated band selection). Source: Richards 2001 arXiv:astro-ph/0012449 (quasar λ_eff). fs_sedshape =
STATELESS (no fit/target/cross-row). READ: research.md ★2a; nodes/node_0030/src; nodes/node_0128 (the
z-conditional-residual that this is DISTINCT from). CPU; minutes.

## notes
well = outside. Cheapest of the 3 — a surgical per-object MgII-line axis on the z≈0.9 GAL↔QSO channel.
