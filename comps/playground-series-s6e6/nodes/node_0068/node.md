---
id: node_0068
desc: refresh public-sub hard-vote bank (CLOUT)
op: draft
parents: [root]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: null
gates: {schema_ok: true, oof_full: "n/a", no_nan: "n/a", dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CLOUT artifact: no OOF/honest CV. oof_full and no_nan are n/a by construction. schema_ok=true (validated). dist_sane=true (GALAXY/QSO/STAR in expected proportions). leak_clean=true (pure public-sub ensemble, no target/train data used). cv_too_good=false (no CV). QUARANTINED: finals slot-2 ONLY, never champion, never honest slot-1."
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [clout, public-vote, finals-slot-2-ONLY, quarantined]
---

## plan
built on:   root (new draft). Rule-9 registration of the A4 hard-vote clout line as a proper node.
change:     Re-pull the CURRENT top public submissions/notebooks (the board has moved since 2026-06-10),
            patch-strip duplicates, and rebuild the plain hard-vote with bank17 tie-break — the EXACT A4 recipe
            on fresh inputs. Weighted/anchored variants are NOT part of this node.
hypothesis: a fresher, larger public-sub consensus beats the 06-10 vote's LB 0.97123 as the max-of-two upside.
target:     Public LB maximize (slot-2 ONLY); beats A4 if LB > 0.97123 when the human spends the probe; CV n/a
            (no OOF).

CPU-only. ⚠️ QUARANTINE: this is a clout/slot-2 artifact. Per MEMORY it is eligible for the finals slot-2 clout
line ONLY — NEVER honest finals slot-1, NEVER champion promotion. It has no OOF / honest CV.

References to READ:
- A4-vote: `refs/a4_vote/` (journal 2026-06-10T12:55Z) — our best public LB 0.97123 and the finals slot-2 swing;
  built from the top-7 subs as of 06-10 and NOT a node yet (rule 9 requires any finals candidate be a node).
- The A4 script in `refs/a4_vote` already does the .b/.c micro-patch-family strip — READ it and reuse.

Build: kaggle CLI to list/pull current top public kernels' submission outputs into `refs/` (snapshot them),
strip .b/.c micro-patch families to one representative each, tie-break to bank17 (champion/submission.csv).
The node's artifact is the plain refreshed hard-vote ONLY.

Separately + cheaply, run two probes/ scripts (one journal line each, NOT nodes): (a) rank-weighted vote (weight
by public LB), (b) bank17-anchored vote that only flips when ≥k externals agree against bank17 — report each by
flip-counts vs bank17 and vs the prior A4 vote, plus agreement with bank17 high-confidence rows. The human
decides whether to spend a PROBE slot on any of the three. No LB probing for selection (budget guard).

## build result
- Pulled 11 fresh kernel outputs (a4_refresh_2026-06-12/). All matched within 99.996%+ agreement to existing
  vote_bank files — board barely moved since 06-10. The refresh confirmed the vote_bank is current.
- vote_bank: 81 CSVs → 70 unique families after .b/.c strip.
- Top-10 selected (LB 0.97108..0.97135): same top tier as A4, extended from 7 to 10 for more consensus.
- Hard vote with bank17 tie-break: 61 ties (0.025%).
- Flips vs bank17: 582 (0.235%). Flips vs prior A4 (LB 0.97123): 25 (0.010%).
- Class distribution: GALAXY=156952, QSO=51464, STAR=39019.

## probe summaries
probe_0068a (rank-weighted vote by LB):
  623 flips vs bank17, 41 flips vs node_0068.
  Bank17 high-confidence rows (all 10 agree): 246254/247435 (99.5%). Zero hc_flips.

probe_0068b (anchored vote, sweep k=2..10):
  k=6 → 582 flips vs bank17, 0 flips vs node_0068 (equivalent to plain majority vote of 10).
  k=7 (strict majority) → 531 flips vs bank17, 51 flips vs node_0068.
  k=5 → 643 flips, k=4 → 694 flips (more aggressive).
  All k: 0 flips in high-confidence rows. Anchored vote at k=8 (490 flips) is most conservative option.

## notes
