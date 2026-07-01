---
id: node_0130
desc: per-class template-chi2 GBDT base
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe, fs_tmplchi2]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966680
sem: 0.000155
folds: [0.967143, 0.966866, 0.966534, 0.966226, 0.966632]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "WASH (analytic-template build). solo 0.966680 (<n030); err-corr 0.8107; stack-add +0.000013 (noise). Same GAL+0.0095/STAR-0.0155 entanglement as every feature."
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T12:16Z
tags: [gbdt, lightgbm, template-chi2, sed-fit, redshift-gated, outside, improve, fs_tmplchi2]
---

## plan
built on:   node_0030 (LightGBM on fs_realmlp_fe, cv 0.966952). Copy src; add ONLY fs_tmplchi2.
change:     fs_tmplchi2 (leak-safety: STATELESS) — per-class SED template-fit χ², GATED on the known
            redshift z. For each row, fit its 5 ugriz fluxes to 3 template families (galaxy CWW, QSO
            Vanden Berk composite, star Pickles), each redshifted to the row's z + integrated through
            SDSS bandpasses, closed-form amplitude α*=Σ(f·T/σ²)/Σ(T²/σ²), χ²_t=Σ(f−α*T)²/σ². Emit
            chi2_gal, chi2_qso, chi2_star + pairwise diffs (chi2_star−chi2_gal, chi2_qso−chi2_gal,
            chi2_qso−chi2_star) + argmin-class + a softmax "template posterior". FULL recipe + sources
            in research.md 2026-06-18 ★1.
hypothesis: a star fits a 1-param stellar template tightly but galaxy/QSO poorly; a z≈0.9 QSO fits a
            power-law+lines composite well, galaxy poorly; a galaxy fits the 4000Å-break CWW well —
            DIFFERENT 5-band residual structures, NOT recoverable from any single color. Knowing z
            removes the redshift degeneracy that limits photometric-only template fitting = our edge.
            Attacks BOTH confusion channels (low-z GAL↔STAR via χ²_star, high-z GAL↔QSO via χ²_qso).
target:     Balanced Accuracy maximize. Cheap-kill fold-0 < 0.965. JUDGE (validation.md gate): solo BA
            vs n030; err-corr vs n070 (<0.72 = finally decorrelated?); STACK-ADD to n091 (bootstrap
            P≥0.90 bar); pred_diagnostic per-class — does it lift GALAXY recall WITHOUT the usual STAR/QSO
            trade (i.e. is it a NEW axis, not the entanglement)? HONEST prior: broadband ceiling may bind
            (research: optical-only template star-score ~96.5%), but the z-gated whole-SED-fit is a real
            untried axis. If it wins, ablate {chi2_gal,chi2_qso,chi2_star} to attribute.

BUILD (research.md ★1): `uv add speclite` (SDSS sdss2010 Doi 2010 filter curves). Templates: CWW
(E/Sbc/Scd/Im), Vanden Berk 2001 QSO composite (astro-ph/0105231), Pickles 1998 stellar (VizieR VI/61).
Precompute synthetic ugriz on a Δz≈0.01 grid per template → per-row lookup + linear algebra (NO training
run for the features). If a template library can't be fetched, FALL BACK to physically-motivated analytic
templates (blackbody grid for stars, power-law f_ν∝ν^−0.5 +/− a broad MgII bump for QSO, a 4000Å-break +
red-continuum model for galaxies) and SAY SO explicitly. fs_tmplchi2 = STATELESS (deterministic row-wise;
no .fit, no target, no cross-row). READ: research.md ★1; nodes/node_0030/src; data.md fs_realmlp_fe. CPU.

## notes
well = outside. The research's top pick — a whole-SED model test using our redshift edge; hits both channels.
