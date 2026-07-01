---
id: node_0124
desc: photometric-metallicity + stellar-locus base
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe, fs_physloc]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966727
sem: 0.000235
folds: [0.967063, 0.967204, 0.966800, 0.965854, 0.966716]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv slightly below parent n030 (0.966727 vs 0.966952); solo gain not achieved; physics features hurt STAR recall (-0.0152) while lifting GALAXY recall (+0.0091); stack-add verdict: bootstrap P=0.000 REAL loss vs champion n091"
leak: clean
lb: null
submitted: null
created: 2026-06-17
decided: 2026-06-17T18:34Z
tags: [gbdt, lightgbm, physics, photometric-metallicity, stellar-locus, low-z-galaxy-star, outside, improve, fs_physloc]
---

## plan
built on:   node_0030 (LightGBM on fs_realmlp_fe, cv 0.966952) — copy its src; keep FE + fold loop +
            tree params identical. ONE atomic change: ADD the new stateless feature block fs_physloc.
change:     Augment fs_realmlp_fe with fs_physloc (leak-safety: STATELESS — row-wise deterministic,
            no fit, no target, no cross-row stat) — a physics block targeting the low-z GALAXY↔STAR
            bottleneck: photometric metallicity [Fe/H]_phot (Ivezic 2008), stellar-locus principal
            colors P1s/P2s, and a redshift warp. EXACT formulas (use raw ugriz + redshift z):
              x = (u-g) if (g-r) <= 0.4 else (u-g) - 2*(g-r) + 0.8
              y = (g-r)
              feh_phot = -13.13 +14.09*x +28.04*y -5.51*x*y -5.90*x**2 -58.68*y**2 \
                         +9.14*x**2*y -20.61*x*y**2 +0.00*x**3 +58.20*y**3
              P1s = 0.910*(u-g) + 0.415*(g-r) - 1.28
              P2s = -0.249*(u-g) + 0.545*(g-r) + 0.234        # perp distance from stellar locus
              z_warp = log10(z + 3e-4)                         # expand the z~=0 boundary
              feh_in_range = 1.0 if -3 < feh_phot < 0.6 else 0.0
            Do NOT clip feh_phot to a physical range — out-of-range values ARE the galaxy signal.
            Use the colours fs_realmlp_fe already defines if present, else compute u-g/g-r from raw.
hypothesis: stars obey a tight (u-g,g-r)->[Fe/H] surface (real values) and sit on the stellar locus
            (P2s~=0); low-z GALAXIES (composite SED + 4000A break) fall OFF the single-star locus, so
            feh_phot goes out-of-range and |P2s| is large — a calibrated stars-vs-low-z-galaxy axis a
            GBDT CANNOT synthesize from raw colours (it's a specific nonlinear surface). z_warp turns
            the sub-0.01 stellar-noise-z vs smallest-galaxy-z crush into a wide stable margin. This is
            genuinely NEW INFORMATION (physics), not a representation/reweight retread — the first real
            test of whether the decorrelation wall yields to new physical signal.
target:     Balanced Accuracy maximize. Cheap-kill fold-0 BA < 0.965 (should >= n030 0.9666 if it helps).
            JUDGE (validation.md structural gate): (a) solo BA vs n030; (b) err-corr vs n070 — is it
            finally a strong-AND-decorrelated base?; (c) stack-add to n091 (the real test — does it beat
            0.970355 by the bootstrap-P>=0.90 bar?); (d) tools/pred_diagnostic.py vs n091 — a CAPTURABLE
            low-z GALAXY/STAR fix-block (fixes >> breaks there, holds on holdout)? If solo wins, ABLATE
            the block next round (feh_phot & P2s likely carry it — they encode the physics two ways).

OUTSIDE well (method literature). READ: nodes/node_0030/src (the LightGBM base to copy); data.md
fs_realmlp_fe recipe; research.md 2026-06-17 entry (full physics derivation + sources Ivezic 2008
tomographyII.pdf eq.4, Bond 2010 signed coeffs, Gu 2015). fs_physloc = stateless (compute once on
train+test, identical transform, no fit) -> uses_data includes it but no fold-fit needed. CPU; minutes.
Emit oof/test/submission; run pred_diagnostic.py vs n091 after scoring.

## notes
well = outside. First genuinely-new-INFORMATION lever since the wall was confirmed; physics, not representation.

## results (post-run)
Solo 5-fold BA: 0.966727 ± 0.000235  vs  n030 parent: 0.966952 (delta = -0.000225; marginal decrease)
  Per-fold: [0.967063, 0.967204, 0.966800, 0.965854, 0.966716]
  best_iters: [637, 640, 574, 615, 586] — similar to n030

Error correlation vs n070: 0.8139 (above the 0.72 decorrelation bar — NOT decorrelated enough to blend)

Stack-add vs n091 champion (0.970355): bootstrap P=0.000 — REAL LOSS (-0.003627), not a gain.
  Confusion delta summary: GALAXY +3445 net fixes, but QSO -561 and STAR -1259 — trades worse.
  STAR recall drops -0.0152, QSO recall drops -0.0048. GALAXY recall improves +0.0091.

Holdout low-z (0.0025-0.15) net fix: +854 (fixes 2251, breaks 1397) — physics features help in low-z
  band but the STAR class is badly hurt (-1259 net STAR fixes across all bands), which dominates BA.

z_warp guard: 8673 rows had z+3e-4 ≤ 0 (clipped to 1e-10). These are stellar sources with tiny
  synthetic negative redshifts — the log10 guard handled them safely.

Verdict: physics features (feh_phot, P1s, P2s) HELP GALAXY recall but badly HURT STAR recall and QSO.
  The feh_phot surface is computed for (u-g,g-r) which are degenerate for some STAR locus vs galaxy
  regions — the LightGBM can't extract a net gain from these 5 new features on top of fs_realmlp_fe.
  The hypothesis was correct mechanistically (GALAXY falls off stellar locus) but the representation
  is not a net improvement — the GBDT may already synthesize similar discriminants from raw colours.
  ABLATION IMPLICATION: feh_phot alone or P2s alone might help; the flat indicator feh_in_range and
  z_warp likely add confusion. These should be tested as a debug/improve child if value is desired.
  Status: valid (no leak, artifacts clean) but below parent — not promoted.
