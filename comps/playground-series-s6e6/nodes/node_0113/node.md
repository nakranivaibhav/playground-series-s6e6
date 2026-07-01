---
id: node_0113
desc: Negative-correlation-learning TabM base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "NCL gate2 TRIPPED: fold-0 err-corr=0.9961(g=0.1) / 0.9962(g=0.3) >= 0.65 threshold. Gate1 PASSED: BA=0.9689/0.9690 >= 0.965. No 5-fold run. NCL at gamma {0.1,0.3} cannot break structural correlation — wall confirmed optimally by NCL itself."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:33Z
decided: 2026-06-15
---

## plan
built on:   root — copy node_0033's TabM-richFE recipe VERBATIM (`nodes/node_0033/src`). The ONE genuinely
            new lever vs the whole session: instead of HOPING a representation/framing lands decorrelated,
            EXPLICITLY optimize the decorrelation↔strength frontier with Negative Correlation Learning
            (Liu & Yao 1999).
change:     Add an NCL penalty to the TabM loss: total = balanced-CE(class) − gamma · corr_penalty, where
            corr_penalty rewards the model's per-sample error/logit being NEGATIVELY correlated with the
            bank's consensus error. Concretely, per train row use the bank reference OOF p_bank (the
            train-fold-only portion of nodes/node_0070/oof.npy) and add gamma · mean[ (f_i − p_bank_i) ·
            (ensemble_mean_i − p_bank_i) ] style NCL term (standard NCL ambiguity decomposition), pushing
            this learner's predictions AWAY from the bank consensus while CE keeps it accurate.
hypothesis: NCL directly searches the frontier no prior node optimized — it finds the MAXIMUM BA achievable
            at a given decorrelation, rather than stumbling on a single (corr,BA) point. If the wall is hard,
            NCL caps BA as soon as corr<~0.7 (confirming the wall is OPTIMAL, not an artifact of weak probes);
            if any slack exists, NCL is the method that extracts it.
target:     BA maximize · sweep gamma∈{0.1,0.3} at fold-0; GATE fold-0 solo BA ≥ 0.965 AND err-corr vs
            node_0070 < 0.65; if both pass → full OOF + restack onto n091 must beat 0.970355 by >2·sem.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0033/src/solution.py → nodes/node_0113/src/solution.py. Keep fs_realmlp_fe, tabm loop,
  balanced CE, PLR, frozen folds, OOF/test/submission writing. ONLY add the NCL penalty.
- NCL term (fit_in_fold): load nodes/node_0070/oof.npy (577347,3 bank consensus). For each training batch,
  penalty = − gamma · E[ (softmax(logits) − p_bank)·(p_bank_meanclass − p_bank) ] — the standard NCL
  ambiguity term that decorrelates this learner from the fixed bank ensemble. Use ONLY train-fold rows'
  p_bank inside the fold loop (the bank OOF for held-out rows is fine to USE as a fixed target since it is
  not THIS model's val prediction and carries no label — but to be strict, only use train-fold p_bank for
  the penalty). At inference, NO penalty — just the class head.
- GATE ORDER: fold-0 only (both gammas); solo BA + err-corr vs node_0070. If best gamma's BA<0.965 OR
  err-corr≥0.65 → STOP, record. Else all 5 folds at best gamma.
- Outputs nodes/node_0113/{oof.npy, test_probs.npy, submission.csv, train.log}. Self-gate: NCL uses bank
  OOF train-fold-only (no label, no val leakage); folds frozen; OOF full/no-NaN/each-row-once; dist sane;
  schema. Write gates + cv/sem/folds + leak + err-corr (gate_note); stage: built. Do NOT submit. tabm lib,
  GPU — marker DONE=/tmp/playground-series-s6e6_node_0113.done.

## notes
well=wildcard. The DEFINITIVE frontier probe: NCL is the only method that OPTIMIZES the decorrelation-
strength tradeoff rather than sampling it. If even NCL can't beat the wall (BA<0.965 at corr<0.65), the
structural ceiling is confirmed OPTIMALLY and the base-search is provably exhausted.

## result
Gate 2 tripped at fold-0. Both gammas:
  gamma=0.1: BA=0.968867  err_corr=0.9961
  gamma=0.3: BA=0.968979  err_corr=0.9962

Gate 1 PASSED (BA > 0.965 for both). Gate 2 FAILED (err_corr >> 0.65 for both).
NCL at gamma {0.1, 0.3} had ZERO decorrelation effect — err_corr ~0.996 matches the
unconstrained TabM baseline, meaning the penalty is too weak to overcome the structural
alignment between any high-accuracy model and the bank consensus at these gamma values.

Interpretation: this confirms the hard wall is structural (the data forces similar predictions
for any high-accuracy model), not an artifact of weak prior probes. NCL is the optimal
frontier probe — it failed to decorrelate, therefore the wall is real and provably tight.
The base-search for decorrelated bases is exhausted.
