---
id: node_0140
desc: redshift-error-aware RealMLP base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_zsoft]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969305
sem: 0.000376
folds: [0.970781, 0.968731, 0.968936, 0.968896, 0.969181]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "stack-cheap-killed (err-corr 0.87 vs n070 — wall) but FORCE_FULL-run for the user-directed solo LB probe; valid best-solo base (cv 0.969305, LB 0.97009); full clean 5-fold OOF + submission produced"
leak: clean
lb: 0.97009
submitted: 2026-06-21T08:38Z
created: 2026-06-21T07:36Z
decided: 2026-06-21T08:42Z
tags: [nn, realmlp, redshift-warp, z-error, decorrelation, outside, gpu]
---

## plan
built on:   root (a new draft — a RealMLP base on the same rich FE but with a
            redshift-ERROR-aware re-expression of the z≈0 neighbourhood, the exact
            bottleneck boundary). Copy nodes/node_0028/src as the scaffold (the
            RealMLP reference recipe + fold-honest OOF/test loop over frozen
            folds.json + fs_realmlp_fe build); the ONLY change is the added
            fs_zsoft feature block.
change:     ONE atomic change (the boundary label-smoothing is SPLIT OUT and is NOT
            part of this node — this is the FEATURE block only): add the NEW
            stateless feature-set fs_zsoft, a redshift-error-aware re-expression
            of the SDSS z-error floor (recipe source: research.md L244,
            z_warp = log10(z + 3e-4) / asinh(z/ε), ε≈SDSS z error floor 3e-4):
              - z_snr  = z / 3e-4  (redshift signal-to-noise vs the error floor);
              - asinh(z / 3e-4)    (well-conditioned z-warp, expands the z≈0 cloud);
              - log10(z + 3e-4)    (the log z-warp, same neighbourhood expansion);
              - a soft STAR-likelihood scalar = a smooth membership of the z≈0
                stellar-noise band (e.g. a logistic/Gaussian bump centred at z≈0
                with width ~the error floor) — NOT a hard flag, a soft score.
            All four are row-wise deterministic on raw redshift (the synthetic data
            has no extinction; apply to raw z) → fs_zsoft leak-safety = stateless.
            Fed ALONGSIDE fs_realmlp_fe to the SAME RealMLP recipe as n028. NO
            label smoothing in this node (that is a separate atomic change for a
            later node).
hypothesis: the GAL↔STAR bottleneck lives at z≈0 where the stellar-noise-z cloud
            and the smallest real galaxy z (~0.02) are crushed into a sub-0.01
            interval. Re-expressing z relative to its ERROR FLOOR gives that
            neighbourhood a wide, stable margin (z_snr / asinh / log-warp) plus a
            soft "is this stellar noise?" axis — information about z that the raw
            log1p(redshift) in fs_realmlp_fe does not encode. This attacks the exact
            boundary the physics/spatial/SED bases all failed to crack, and an
            error-relative z representation is a new axis that MAY decorrelate.
target:     Balanced Accuracy maximize. CONCRETE CHEAP-KILL (run on fold-0 BEFORE
            any full 5-fold): continue ONLY if fold-0 err-corr vs node_0070 < 0.65
            AND solo fold-0 BA ≥ 0.965; else STOP (this is the wall's verdict —
            either correlated like every prior z-feature, or below tier). If it
            passes, run the full 5-fold → oof.npy, then the stack-add to n091 via
            the structural gate (bootstrap P ≥ 0.90 + holdout fix-block).

## build protocol (cost-staged cheap-kill)
1. Build fs_zsoft (cheap, stateless, CPU) + append to the n028 fs_realmlp_fe matrix.
2. SMOKE the RealMLP: small subsample, few epochs — pipeline + timing + VRAM.
3. FOLD-0 (background + marker /tmp/s6e6_node_0140.done): compute BOTH solo fold-0
   BA AND err-corr vs node_0070 (load node_0070/oof.npy on the fold-0 val rows).
   APPLY THE CHEAP-KILL: err-corr < 0.65 AND BA ≥ 0.965 to continue, else STOP.
4. FULL 5-fold only if the cheap-kill passes → oof.npy + test_probs.npy +
   submission.csv over the frozen folds.

## leakage discipline (same standard as parent-scaffold n28)
- fs_zsoft is stateless: all four features are row-wise deterministic on raw z with
  a FIXED constant 3e-4 (NOT data-fit) — no target, no cross-row stat, no fit. Safe
  to compute once on train+test together. Verify the soft STAR-likelihood uses a
  fixed centre/width, not a train-fitted one (a fitted width would make it
  fit_in_fold).
- Stateless fs_realmlp_fe once; factorize/KBins/TargetEncoder fit train-fold-only;
  folds from frozen folds.json. OOF covers every train row once.

## references to READ
- research.md L244 (the z_warp recipe: log10(z+3e-4) / asinh(z/ε), ε≈3e-4 SDSS
  error floor, "re-express NOT remove redshift to EXPAND the z≈0 neighbourhood,
  attacks the exact bottleneck boundary") — the source recipe for fs_zsoft.
- nodes/node_0028/src/solution.py + features.txt — the RealMLP FE + fold-honest
  OOF/test scaffold to copy (only fs_zsoft added).
- nodes/node_0124|0125|0128/node.md + journal 2026-06-17/18 (physics-locus,
  spatial-kNN, z-local-color all entangled at err-corr ~0.81) — the prior every
  z-attack lands in the wall; the cheap-kill is calibrated to stop fast if this one
  does too.
- nodes/node_0070/oof.npy + tools/pred_diagnostic.py — the err-corr reference for
  the cheap-kill and the structural stack gate.
