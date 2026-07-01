---
id: node_0097
desc: stronger Nystroem RBF base + stack-add
op: improve
parents: [node_0096]
uses_data: [fs_rbf_nystroem]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: false, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "FOLD-0 KILL: solo_BA=0.9466 < 0.955 threshold (relaxed gate). Decorrelation preserved: err_corr=0.541 vs node_0070 (well below 0.65 floor). n_components=2000 (4000 OOM'd; 461k x 4000 = ~7.4GB matrix, killed by OS). gamma=0.0556 (1/n_features, best of 5 candidates). Gamma sweep on 1000 components gave 0.9447; full n_components=2000 gave 0.9466. The BA gap vs threshold is ~0.85pp — higher capacity (4000+) might clear it but memory is the bottleneck on this hardware. Full OOF not produced — oof.npy has fold-0 rows only (rest zeros). Stack-add test not reached. schema_ok/no_nan/dist_sane all pass on partial artifacts. Runtime ~2.1min total."
leak: clean
lb: null
submitted: null
created: 2026-06-14T10:57Z
decided: 2026-06-14
---

## plan
built on:   node_0096 (Nystroem RBF kernel-feature base). node_0096 was KILLED at fold-0 only
            by the solo-BA<0.96 floor — but it scored err-corr 0.530 vs node_0070, the FIRST genuinely
            decorrelated base since FT-Transformer (every other reframing floored at 0.70-0.79). The
            kernel head was simply under-capacity (1000 components on a 13-dim input).
change:     STRENGTHEN the RBF kernel base to lift solo BA toward/above ~0.955 WHILE preserving the
            decorrelation, then test the STACK-ADD. Three coupled knobs (this is ONE hypothesis —
            "a higher-capacity RBF kernel machine becomes useful without collapsing onto the bank's
            error structure"): (1) feed the FULL rich FE (fs_realmlp_fe, ~26 feats) into the Nystroem
            map, not the 13-dim core; (2) raise n_components to ~4000-8000; (3) re-sweep gamma on the
            richer input. Optionally a small 1-hidden-layer MLP head instead of LogReg if the linear
            head caps BA. Everything else (fit_in_fold scaler+landmarks, balanced weighting) stays.
hypothesis: RBF decision geometry (similarity to landmarks / radial boundaries) is fundamentally
            different from trees (axis splits) and NNs/RealMLP (learned linear projections), which is
            why it decorrelated at low capacity. Raising capacity should raise accuracy along that
            different geometry rather than forcing convergence onto the bank's errors — so err-corr
            should stay well below the 0.70 floor even as BA climbs.
target:     Balanced Accuracy maximize. This is a DIVERSITY-FEEDER for the champion stack — the
            decisive test is the STACK-ADD, not solo BA. Promote the resulting STACK only if its
            fold-honest OOF BA beats champion node_0091 (0.970355) by > 2·sem.

HOW (hand the developer the concrete experiment):
- Reuse node_0096's pipeline (nodes/node_0096/src — StandardScaler → Nystroem(rbf) → balanced
  multinomial LogReg, all fit_in_fold on a train-fold subsample for tractability at 577k rows).
  Library-first: sklearn.kernel_approximation.Nystroem — do NOT hand-roll. The only changes are the
  input vector (rich FE), n_components, gamma sweep, and (optional) head.
- Input: the full fs_realmlp_fe feature-set (read data.md fs_realmlp_fe recipe; it's the rich FE our
  strong bases use). Standardize fit_in_fold.
- FOLD-0 GATE FIRST (cheap): copy node_0096's err-corr gate (nodes/node_0096/src vs
  nodes/node_0070/oof.npy, shape (577347,3), GALAXY=0/QSO=1/STAR=2). Proceed to full 5-fold ONLY if
  fold-0 err-corr < 0.65 AND solo BA >= 0.955 (RELAXED from 0.96 — the exceptional decorrelation
  justifies a lower solo floor; the stack-add is the real test). KILL otherwise (record BA + err-corr).
- ON PASS: full 5-fold OOF (577347,3) + test_probs (247435,3). Then THE DECISIVE TEST — stack-add:
  take the champion node_0091 recipe (balanced multinomial LogReg on clipped log-probs over the full
  pool; nodes/node_0091/src/solution.py) and add THIS base's OOF as one more column; refit the meta
  fold-honest (nested C as in n091). Report: champion stacked CV 0.970355 baseline, the new stacked CV
  with this base added, the delta, and per-fold deltas. PROMOTE the new stack (a combine node, register
  it) only if stacked OOF BA > 0.970355 by > 2·sem.
- References to READ: nodes/node_0096/node.md + src (the base to strengthen + the gate); node_0091/
  src/solution.py (the champion stack recipe to add into); data.md fs_realmlp_fe row; nodes/node_0070/
  oof.npy (err-corr baseline); MEMORY [ensemble] n62 (weak+decorrelated 0.655/0.958 HURT the stack —
  the cautionary prior; this base must clear the stack-add test, not just decorrelate).
- Self-gate (kaggle-leakage): scaler/landmarks/gamma fit train-fold-only; folds frozen; OOF full/no-NaN;
  dist sane; submission schema. Write gate booleans + leak. stage: built. Do NOT submit.

## notes
node_0096 numbers to beat: solo BA 0.9418 / err-corr 0.530 (gamma 0.077 = 1/n_features on 13-dim).
The honest risk (state plainly): a stronger kernel head may converge toward the bank's accuracy AND
its error structure (err-corr climbs back to 0.70+), or may plateau below 0.955 — either way the
fold-0 gate kills it cheaply. The upside is the first base-set expansion since FT-T.
