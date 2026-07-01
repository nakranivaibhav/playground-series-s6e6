---
id: node_0099
desc: LightGBM meta-stacker over full pool
op: improve
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968784
sem: 0.000381
folds: [0.969802, 0.968489, 0.968493, 0.967671, 0.969464]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "NULL — GBDT meta over full pool CV 0.968784 (FULL arm; TIGHT 0.968620) vs champion LogReg 0.970355, delta −0.001571 (all 5 folds worse by 0.001-0.0023). LogReg sanity baseline reproduced exactly (0.970355). The GBDT meta overfits the OOF as MEMORY n80 predicted (meta-capacity is not the lever). Combine-mechanism axis now fully closed: LogReg(0.9704) > Caruana(0.9700) > GBDT(0.9688) > simplex/TabPFN-3. Built under orchestrator marker-takeover after the developer agent overflowed context."
leak: clean
lb: null
submitted: null
created: 2026-06-14T12:48Z
decided: 2026-06-14
---

## plan
built on:   champion node_0091 (balanced multinomial LogReg over the FULL OOF pool — bank-17 + FT-T +
            ~54 in-house bases — at C=0.003). Same pool, byte-identical loading.
change:     Replace the LINEAR LogReg meta with a LightGBM MULTICLASS meta over the same pooled OOF
            matrix — the one combine MECHANISM never tested over the full pool (we have LogReg [best],
            Caruana ensemble-selection [A2, lost], non-negative simplex [n093, lost], TabPFN-3 [n080,
            lost]). A gradient-boosted meta can model non-linear interactions between base predictions
            that the linear meta cannot.
hypothesis: IF there is residual non-linear structure in how the base probabilities should combine
            (e.g. "trust base A only when base B is uncertain"), a GBDT meta captures it where LogReg
            can't. HONEST CAVEAT / low-EV flag: MEMORY [stack] n80 found meta-capacity is NOT the lever
            on this saturated bank (TabPFN-3 meta LOST to LogReg by −0.0004); a GBDT meta is even more
            capacity and the classic failure mode is overfitting the OOF (optimistic CV that washes on
            LB). Expect a wash; the value is closing the combine-mechanism axis completely.
target:     Balanced Accuracy maximize. Promote ONLY if fold-honest stacked OOF CV beats champion
            node_0091 (0.970355) by > 2·sem. Because a GBDT meta can overfit OOF, treat any CV "win"
            with extra suspicion (cv_too_good) — it must look plausible per-fold, and would need an LB
            probe before any finals move.

HOW (hand the developer the concrete experiment):
- Reuse node_0091's pool-loading VERBATIM (read nodes/node_0091/src/solution.py — the MANIFEST/bank
  loader, FT-T loader, the ~54 in-house OOF loader, the norm()/logp() helpers, and the frozen-fold
  loop). The FEATURE MATRIX for the meta = the same full pool of per-base class-probs (or clipped
  log-probs) it builds. ONLY the meta changes.
- Meta: lightgbm multiclass (objective='multiclass', num_class=3), class-balanced (sample/class
  weights), MODEST capacity to fight OOF overfit (e.g. num_leaves ~15-31, learning_rate ~0.03,
  n_estimators with early-stopping on the inner val, feature/bagging fraction < 1). Fit fold-honest:
  train the meta on the 4 training folds' OOF, predict the held-out fold; rotate all 5 folds. NEVER
  fit the meta on the fold being scored. Score = balanced accuracy of argmax.
- A/B: report the GBDT-meta fold-honest CV vs the champion's 0.970355 (reproduce the LogReg baseline
  first as a sanity assert), per-fold deltas, and whether it clears 0.970355 + 2·sem.
- Produce nodes/node_0099/{oof.npy (577347,3 meta OOF), test_probs.npy (247435,3), submission.csv,
  train.log}. Self-gate (kaggle-leakage): meta fit fold-honest; folds frozen; OOF full/no-NaN; dist
  sane; schema; cv_too_good judgment (a GBDT meta beating LogReg on CV is a red flag — eyeball it).
  Write gate booleans + leak; stage: built. Do NOT submit. `uv run` (lightgbm already a dep). CPU/GPU
  minutes — no marker file needed unless it runs long.

## notes
This closes the combine-mechanism axis (linear / greedy-selection / convex / prior-fitted-net / GBDT
all tested over the full pool). If it washes, the combiner is exhaustively confirmed maxed and the
champion node_0091 stands as the honest ceiling.
