---
id: node_0126
desc: physics ablation feh_phot + P2s only
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe, fs_physloc_min]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966535
sem: 0.000260
folds: [0.967089, 0.967181, 0.966218, 0.965835, 0.966353]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-17T18:44Z
tags: [gbdt, lightgbm, physics, ablation, stellar-locus, fs_physloc_min, outside, improve]
---

## plan
built on:   node_0030 (LightGBM on fs_realmlp_fe, cv 0.966952) — same template as n124. ABLATION of
            n124's fs_physloc: keep ONLY the two core star-galaxy discriminants, drop the noise.
change:     Add fs_physloc_min = {feh_phot, P2s} ONLY (STATELESS). Drop P1s, z_warp, and feh_in_range
            from n124's block — the agent post-mortem (node_0124 results) flagged z_warp (which clipped
            8673 negative-z stars to one value, an artifact) and the flat feh_in_range boolean as likely
            STAR-confusion noise. Exact formulas (same as n124, raw ugriz):
              x = (u-g) if (g-r) <= 0.4 else (u-g) - 2*(g-r) + 0.8 ;  y = (g-r)
              feh_phot = -13.13 +14.09*x +28.04*y -5.51*x*y -5.90*x**2 -58.68*y**2 +9.14*x**2*y -20.61*x*y**2 +58.20*y**3
              P2s = -0.249*(u-g) + 0.545*(g-r) + 0.234
            Everything else byte-identical to n030.
hypothesis: n124 proved the physics MOVES the low-z GALAXY boundary (+0.0091 GALAXY recall, +854 low-z
            fixes) but the full block hurt STAR (-0.0152) more, partly from z_warp/feh_in_range noise.
            Stripping to the two clean discriminants may keep the GALAXY gain while shedding the STAR
            damage — testing whether the physics lever nets positive once de-noised, or whether the
            STAR cost is intrinsic to the star-galaxy axis (entanglement, like n118).
target:     Balanced Accuracy maximize. Cheap-kill fold-0 < 0.965. JUDGE (validation.md gate): solo BA
            vs n030 (0.966952) AND vs n124 (0.966727) — did de-noising recover the STAR loss?; err-corr
            vs n070; stack-add to n091 (bootstrap-P>=0.90 bar); pred_diagnostic vs n091 — did the STAR
            net-fix recover while keeping the low-z GALAXY fix-block? If STILL net-negative/entangled,
            the physics lever is CLOSED (boundary tradeoff is intrinsic, not a noise artifact).

OUTSIDE well, ablation. READ: nodes/node_0124/node.md results section (the full-block post-mortem);
nodes/node_0030/src (template); research.md physics entry. fs_physloc_min = stateless. CPU; minutes.

## notes
well = outside. De-noise ablation of the physics lever — decides if the GALAXY/STAR tradeoff is intrinsic.

## results (2026-06-17T18:44Z)
CV 0.966535 ± 0.000260 (5 folds: 0.967089, 0.967181, 0.966218, 0.965835, 0.966353)
  vs n030 (0.966952): -0.000417 BA — worse than parent
  vs n124 (0.966727): -0.000192 BA — also worse than n124 full block

Per-class recall (vs n124):
  GALAXY: 0.969718 (delta -0.000127 vs n124)
  QSO:    0.971616 (delta -0.000196 vs n124)
  STAR:   0.958271 (delta -0.000254 vs n124)

Per-class recall (vs n030 parent):
  GALAXY: 0.969718 (delta -0.000026)  — near-flat
  QSO:    0.971616 (delta -0.000162)
  STAR:   0.958271 (delta -0.001064) — STILL hurt vs n030

Diagnostic vs champion n091 (BA 0.970355):
  n126 BA 0.966535 — delta -0.003820 (all rows); holdout -0.004275
  GALAXY: +0.0090 recall vs n091 (low-z GALAXY fix block confirmed)
  QSO:    -0.0050 vs n091
  STAR:   -0.0155 vs n091 (STAR damage essentially identical to n124's -0.0152)
  Net fixes vs n091: +1,533 (4,323 fixes / 2,790 breaks); McNemar p=2.67e-74
  Stack-add bootstrap P(n126 > n091) = 0.000 — REAL loss, no stack value

Error correlation n126 vs n070: 0.8158 (vs n030 vs n070: 0.8175 — nearly identical)

VERDICT: The GALAXY/STAR tradeoff is INTRINSIC. Removing z_warp and feh_in_range
(the suspected noise) did not recover STAR recall — it got marginally worse (-0.0003
vs n124, -0.0011 vs n030). feh_phot and P2s still push stars into the galaxy locus,
and removing P1s/z_warp did not help. The physics stellar-locus lever is CLOSED for
this dataset — the star-galaxy boundary confusion is not a noise artifact but a
fundamental feature of the photometric space. The physics lever does not net-help BA
in any ablation variant (n124 or n126).
