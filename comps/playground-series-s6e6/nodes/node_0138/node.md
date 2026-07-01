---
id: node_0138
desc: GBDT meta on curated complementary subset
op: combine
parents: [node_0091, node_0039]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969691
sem: 0.000332
folds: [0.970885, 0.968839, 0.969475, 0.969621, 0.969637]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv=0.969691 vs champion 0.970355; delta=-0.000664 (all 5 folds worse). Bootstrap P(cand>champ)=0.000 — definitively WORSE. Holdout delta=-0.000989. GALAXY recall +0.0021 but STAR recall -0.0038 nets to loss. LogReg sanity on curated subset=0.970135 (confirmed ingest OK). Closes shallow-GBDT x curated-subset cell — nonlinear-meta family now fully closed."
leak: clean
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: null
tags: [stack, gbdt-meta, curated-subset, nonlinear-meta, exploit]
---

## plan
built on:   the champion stack n091 (the OOF pool + the proven balanced-LogReg meta
            as the reference to beat) and n039 (load-bearing CatBoost, the top
            single-base contributor by the drop-study). uses_data [] — this is a
            combine over pre-computed base OOF, not a feature-set consumer.
change:     ONE atomic change vs the closed "GBDT-meta over the FULL pool" result
            (n099 overfit −0.0016): fit a SHALLOW GBDT meta (LightGBM, small depth
            / strong regularization, monotone-free) over a CURATED, SMALL
            complementary SUBSET of bases — NOT the full ~63-col pool. The subset =
            the de-correlated, individually-strong bases the drop-study + err-corr
            analysis flag as carrying distinct signal (e.g. the CatBoost family
            n039/n043, RealMLP n028/n032/n035, TabM n033, FT-T, the strongest
            bank-17 members) — small enough that a nonlinear meta cannot memorise
            the OOF. The hypothesis being isolated: n099's GBDT-meta failure was
            CAPACITY×WIDTH (a deep tree meta over 63 redundant cols overfits the
            OOF), NOT that nonlinearity is useless — a shallow GBDT over a curated
            handful might capture a base-INTERACTION the linear meta cannot, without
            the overfit.
hypothesis: the linear L2 meta (n091) is additive in base log-probs; a base
            INTERACTION (e.g. "trust CatBoost's GALAXY call only when TabM also
            leans GALAXY") is invisible to it. A shallow regularized GBDT over a
            SMALL decorrelated subset can express that interaction with few enough
            params to generalise — the narrow gap n099's wide overfit could not
            test.
target:     Balanced Accuracy maximize. Promote iff CV beats champion 0.970355 by
            the structural gate (tools/pred_diagnostic.py bootstrap P(>champ) ≥ 0.90
            AND a holdout fix-block that holds). Expectation is honest: the
            combine-MECHANISM axis is largely closed (n099 GBDT-meta, n080 TabPFN
            meta, n100/n122/n127 region/MoE) — this is the one untested cell
            (shallow GBDT × curated subset), and a clean WASH here closes the
            nonlinear-meta family for good with attribution.

## build protocol
- SANITY ASSERT FIRST (seconds): a LINEAR meta over the SAME curated subset must
  reproduce ≈ that subset's known LogReg stack CV — if the OOF ingest /
  column-order / clip-norm is off, the GBDT reading is meaningless. STOP and fix
  before tuning the GBDT (the n070-v1 / n122 misbuild guard).
- NESTED honesty: any GBDT meta hyperparameter (n_estimators via early stop, depth)
  must be selected on an INNER split of the outer-train portion, never on the outer
  val fold — exactly as n091's nested C grid does for the linear meta. The meta
  trains on fold-honest base OOF; verify the columns are OOF, not refit-on-full.
- CPU, minutes (no base retrain).

## references to READ
- nodes/node_0099/node.md + journal entry (LightGBM meta over FULL pool overfit
  −0.0016) — the failure this node deliberately narrows (subset + shallow, not
  full + deep).
- nodes/node_0080/node.md (TabPFN-3 L1 meta-stacker, below champ) — the other
  nonlinear-meta data point.
- champion/src/solution.py + a1_full_merge.py / a1_submit.py — the OOF-ingest +
  the linear-meta reference the sanity-assert must reproduce; reuse the exact clip
  ±30 / normalize.
- the n091 FULL-pool base list (champion/src/solution.py) + the drop-study
  (probes/drop_study_ranking.csv, journal 2026-06-16 DROP-STUDY) — to pick the
  curated complementary subset (top causal contributors, not |coef|, which the
  drop-study showed lies).
- tools/pred_diagnostic.py — the structural + holdout gate.

## results (post-run)
### pred_diagnostic.py output (node_0091 vs node_0138)
- All rows: BA champ=0.970355 cand=0.969691 delta=-0.000663
- Per-class recall (champ/cand/delta): GALAXY +0.0021 / QSO -0.0003 / STAR -0.0038
- Holdout (fold 4): BA champ=0.970626 cand=0.969637 delta=-0.000989
- Flips: 2100 fixes / 1654 breaks / net +446 (McNemar p=3.56e-13, real difference)
- Net fix by class: GALAXY +794 / QSO -32 / STAR -316
- Bootstrap B=3000: P(cand>champ)=0.000  95%CI=[-0.000906,-0.000414]
- VERDICT: REAL LOSS — promote gate FAILS

### interpretation
Sanity check confirmed: LogReg over the SAME curated 8-base × 24-col matrix = 0.970135
(close to champion 0.970355), so the ingest was correct. The GBDT meta underperforms
the linear meta on this curated subset by −0.000664. Specifically: GBDT helps GALAXY
(+794 net fixes, recall +0.002) but badly hurts STAR (−316 net, recall −0.004). The
nonlinear meta picks up some GALAXY-specific structure but overshoots on STAR.
This fully closes the "shallow GBDT × curated subset" axis (n099 closed FULL×deep;
n138 closes narrow×shallow). Nonlinear-meta family is now exhaustively confirmed
below LogReg for this task — no further GBDT-meta variants are warranted.
