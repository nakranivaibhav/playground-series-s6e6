---
id: node_0114
desc: NCL aggressive-gamma frontier trace
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "WALL CONFIRMED. NCL cliff: gamma=1.0→BA=0.9675,err_corr=0.9857 (flat); gamma=5.0→BA=0.0271(COLLAPSED),err_corr=-0.1766; gamma=20.0→BA=0.0195,err_corr=-0.2410. No gamma achieves err_corr<0.65 with BA≥0.965. Stopped after fold-0 sweep per plan — no oof/submission written."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:33Z
decided: 2026-06-15
---

## plan
built on:   root — completes node_0113. n113's NCL penalty at γ∈{0.1,0.3} NEVER ENGAGED (err-corr unmoved
            at ~0.996 = unconstrained baseline; the two γ gave identical corr → entirely in the flat region).
            That run is inconclusive, NOT a frontier trace. Reuse n113's src VERBATIM; only change the γ grid.
change:     Aggressive γ sweep {1, 5, 20} (and the standard PER-CLASS ERROR-correlation metric vs node_0070,
            same as every other node — NOT the prob-corr that gave n113's anomalous 0.996) to actually PUSH
            correlation down and trace what BA it costs. Report the full (γ, BA, err-corr) frontier at fold-0.
hypothesis: a high-enough γ WILL drop err-corr below 0.65; the question is what BA survives. If BA stays
            ≥0.965 at err-corr<0.65 → NCL FOUND a strong-decorrelated base (break the wall, the prize). If
            BA collapses as soon as corr<0.7 → the wall is confirmed OPTIMALLY (NCL is the tightest probe).
target:     BA maximize · GATE any γ giving fold-0 BA ≥ 0.965 AND err-corr < 0.65 → run that γ full 5 folds
            + restack onto n091 (>2·sem to promote). Else record the frontier and close the base-search.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0113/src/solution.py → nodes/node_0114/src/solution.py. Keep EVERYTHING; change only the
  γ grid to {1.0, 5.0, 20.0}. CRITICAL: ensure the err-corr REPORTED is the standard mean per-class
  ERROR-correlation vs nodes/node_0070/oof.npy (the metric every other node uses, range ~0.48-0.83) — if
  n113 reported probability-correlation (0.996), fix it to error-correlation so the frontier is comparable.
- For EACH γ at fold-0, log (BA, err-corr). Pick the γ that FIRST achieves err-corr<0.65; if its BA≥0.965,
  run full 5 folds at that γ. If NO γ reaches err-corr<0.65, OR the one that does has BA<0.965, STOP after
  fold-0 and record the full frontier (this is the definitive negative).
- Outputs nodes/node_0114/{oof.npy, test_probs.npy, submission.csv, train.log} (only if a γ clears the gate;
  else just train.log + the frontier table). Self-gate: NCL bank-OOF train-fold-only; folds frozen; if OOF
  written, full/no-NaN/each-row-once/dist-sane/schema. Write gates + cv/sem/folds + leak + the (γ,BA,corr)
  frontier in gate_note; stage: built. Do NOT submit. tabm lib, GPU — marker
  DONE=/tmp/playground-series-s6e6_node_0114.done.

## notes
well=wildcard. The TRUE definitive frontier probe (n113 was under-powered). Either it breaks the wall
(strong-decorrelated base = the session's prize) or it confirms the wall optimally with a clean (γ,BA,corr)
curve. Use the standard error-corr metric so the result is comparable to the rest of the bank.
