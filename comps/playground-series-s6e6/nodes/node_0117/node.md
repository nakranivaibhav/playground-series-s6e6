---
id: node_0117
desc: max-decorrelated rank-vote finals hedge
op: combine
parents: [node_0091, node_0039, node_0070]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969272
sem: 0.000235
folds: [0.970168, 0.969016, 0.968908, 0.968953, 0.969317]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "Finals hedge only: cv=0.969272 is -0.001083 below champion n091 (0.970355); 0.76% row-level disagreement vs n091 is the hedge value"
leak: clean
lb: 0.97003
submitted: 2026-06-17T11:14Z
created: 2026-06-16T13:01Z
decided: 2026-06-16T13:54Z
tags: [stack, rank-vote, finals-hedge, decorrelation, outside, combine]
---

## plan
built on:   node_0091 (LogReg mega-stack champion, lb 0.97121) + node_0039 (CatBoost rich-FE base) + node_0070 (bank-17 + FT-T stack, the prior best-honest-LB). This is a finals slot-2 candidate built on a DIFFERENT combiner form (rank/argmax vote) than the LogReg champion — the parents anchor which load-bearing bases the vote draws on; the champion's decision rule is deliberately NOT kept.
change:     Build ONE honest finals-hedge candidate whose error geometry is maximally DIFFERENT from the LogReg champion: a fold-honest rank/argmax-VOTE ensemble over the top causal-family bases (cat-3 + node_0039 CatBoost, realmlp-2, lgbm-5 + node_0003, FT-T) using a per-class-balanced soft-rank average + DE per-class weights, NOT a dense LogReg on log-probs. Report its honest CV and its prediction-disagreement vs n091.
hypothesis: a tier-strength ensemble built on a different decision rule (rank-vote) over the load-bearing bases produces private errors more independent of the LogReg champion than any weight-blend, giving a genuine shake-up hedge the deferred finals selection needs.
target:     Balanced Accuracy maximize; honest CV within ~2·sem of n091 (0.970355) AND maximal row-disagreement vs n091 — a robustness hedge, not a CV win.

DISCUSSIONS-DRIVEN, not a CV-chase. discussions.md topic 704512 (Siddhesh/Deotte variance analysis): public σ≈0.00087, private σ≈0.00035, top-25 inside a 0.0002 window — the private ordering is DRAW-DOMINATED. The 'two decorrelated picks' Monte-Carlo says ~0 uplift on the MEAN, but for a draw-dominated private split the SECOND finals pick is a VARIANCE hedge against shake, and the standard move is to pair the LogReg champion (n091 LB 0.97121) with a comparable-CV candidate built on a DIFFERENT combiner FORM so their private errors are as independent as the bank allows. n093 already showed a convex simplex blend is structurally weaker (0.963) — so this must be a VOTE/rank ensemble (different decision rule), not a weight-blend, and restricted to the LOO-load-bearing bases (CatBoost #1 by 4×, RealMLP #2, LGBM #3 per probes/drop_study_ranking.csv) so it stays at tier. READ: champion/src/solution.py (OOF ingest + DE-threshold pattern from a1_submit lineage); nodes/node_0017/node.md (the prior per-class threshold-tune on a prob-average — this is the rank-vote analogue); refs/a4_vote + refs/vote_bank (public vote-blend recipes for the rank-average mechanics); probes/drop_study_ranking.csv (which bases are load-bearing). Deliverable: oof/test/submission + the honest CV/sem + a row-level disagreement-rate vs n091 (the hedge value). Do NOT promote on CV; this is a finals slot-2 candidate, judged on robustness not on beating n091. CPU-only, minutes.

## notes
well = outside.
