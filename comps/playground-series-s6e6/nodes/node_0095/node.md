---
id: node_0095
desc: strict z-residual-only LightGBM base
op: draft
parents: [root]
uses_data: [fs_zresid_strict]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.970153
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "KILLED at fold-0: solo BA=0.720574 < 0.96 threshold. err_corr_vs_n70=0.1772 (BELOW 0.65 — the strict residual-only view DOES decorrelate, but it is a hopelessly weak classifier: 6 features, 2000 trees without convergence, log-loss still falling at iteration 2000). No full OOF produced. Hypothesis confirmed on one axis (decorrelation real at 0.177) but refuted on the other (solo BA 0.72 is far below any useful base tier). Strictly dropping raw-z + magnitudes removes essentially all discriminative signal from the feature vector."
leak: clean
lb: null
submitted: null
created: 2026-06-14T10:41Z
decided: 2026-06-14
---

## plan
built on:   root (the LightGBM richFE recipe from node_0030 is the *training* template, but this is a draft off root because the INPUT REPRESENTATION is wholly replaced — no fs_realmlp_fe, no raw mags/colors/z-continuous)
change:     Train a residual-only LightGBM on a NEW, STRICTER feature representation `fs_zresid_strict` (fit_in_fold): ONLY the z-conditional color residuals + a single binary STAR-flag (z≈0 bin). DROP everything the existing residual nodes kept — raw redshift-as-continuous, raw magnitudes, raw colors, AND the per-magnitude z-scores that node_0086/n87's `fs_zresid` retained. The hypothesis is that the prior residual bases (n86 err-corr 0.72, n87 0.70) failed to decorrelate because they still carried the raw redshift/magnitude axes that every champion base already exploits; stripping them forces the model onto a purely shape-of-residual geometry.
hypothesis: A residual-ONLY representation (no raw z/mag/color leakage back into the model) yields a base whose errors are genuinely de-correlated from the bank-17 + FT-T stack (err-corr below the ~0.70 wall that n86/n87/n90/n94 all hit), so it can add signal a restack would select.
target:     Balanced Accuracy Score, maximize · beats parent if CV > 0.970153 (champion baseline). REALISTICALLY this is a decorrelation probe, not a CV-beater — judged by the fold-0 err-corr gate, not by topping the champion.

### LOW-EV — honest flag
This is HONESTLY low expected-value. The decorrelation axis "z-conditional residuals" has already been tried THREE times and washed: node_0086 (TabM, err-corr 0.72), node_0087 (LGBM, err-corr 0.70), and the closed-list verdict in graph.md header ("err-corr 0.79 — instance-weighting does NOT decorrelate; joins residuals/OvR/families on the closed list"). The bet here is narrow: that the prior residual bases washed because they LEAKED the raw redshift/magnitude axes back in, and a STRICT residual-only view might finally drop below the err-corr wall. If it doesn't clear the gate it dies cheaply at fold-0. Do not over-invest.

### The cheap fold-0 err-corr KILL gate (run FIRST, before any full 5-fold run)
COPY the gate harness from `nodes/node_0094/src/solution.py` (it already implements exactly this pattern against node_0070). Procedure:
1. Build `fs_zresid_strict` on fold-0's train split only (bin edges + per-bin color mean/std fit on train-fold, global fallback for sparse bins).
2. Train the residual-only LightGBM on fold-0 train, predict fold-0 val.
3. Compute solo Balanced Accuracy on fold-0 val AND the error-correlation vs node_0070's fold-0 OOF slice (`nodes/node_0070/oof.npy`, rows aligned to frozen folds — take the fold-0 val indices).
4. **KILL the node (status dead, one journal line) if err-corr ≥ 0.65 OR solo BA < 0.96.** Only if BOTH pass, proceed to the full 5-fold OOF build + restack-candidacy.

### HOW to build fs_zresid_strict (fit_in_fold)
- Start from `fs_zresid`'s canonical recipe (`add_zconditional_residuals`, described in data.md line 89 and research.md lines 83-105): ~40 redshift quantile bins, per-bin per-color MEAN/STD fit on the TRAIN FOLD ONLY, applied to val+test; global mean/std fallback for sparse bins.
- STRICT differences vs fs_zresid: (a) DROP raw redshift-as-continuous; (b) DROP raw magnitudes; (c) DROP raw colors; (d) DROP the per-MAGNITUDE z-scores — keep ONLY the per-COLOR residual z-scores; (e) ADD one binary STAR-flag = 1 if the row falls in the z≈0 (lowest) redshift bin, else 0. Final feature vector = {color-residual z-scores} ∪ {star_flag}.
- Implement the recipe INSIDE node_0095's own `src/` (single canonical recipe per data.md convention for fs_zresid — no cross-node import).
- LightGBM training recipe: reuse node_0030's hyperparameters/loop verbatim (`nodes/node_0030/src/` — multiclass, balanced handling, the same early-stopping/num_leaves it used on richFE) so the ONLY variable vs the residual-LGBM baseline (n87) is the strict representation.

### References to READ
- `nodes/node_0094/src/solution.py` — copy the fold-0 err-corr gate harness (already wired to node_0070).
- `nodes/node_0030/src/` — the LightGBM richFE training recipe to reuse verbatim.
- `nodes/node_0087/node.md` — the prior residual-LGBM (err-corr 0.70) this strictens.
- `nodes/node_0086/node.md` — the residual-TabM sibling (err-corr 0.72).
- data.md line 89 (fs_zresid recipe) + research.md lines 83-105 (z-conditional residual derivation).
- `nodes/node_0070/oof.npy` — the fold-honest OOF the err-corr gate measures against.
- graph.md header — the closed-list decorrelation verdict (why this is LOW-EV).
