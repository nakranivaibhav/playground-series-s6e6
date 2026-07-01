---
id: node_0074
desc: TabM on A4 public-consensus pseudo-test (CLOUT)
op: improve
parents: [node_0033]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: decided
cv: 0.968528
sem: 0.000186
folds: [0.969150, 0.968523, 0.968029, 0.968621, 0.968318]
leak: clean
gate_note: "CLOUT/slot-2-ONLY (A4 public-consensus pseudo-labels) — NEVER honest slot-1 or champion. Solo CV 0.968528 (+0.000475 vs n33 0.968053): disjoint-teacher pseudo-label gave a REAL solo lift (unlike n67 self-distill which washed). Restack into bank-17 not cleanly measured (LogReg+DEthresh probe too slow); prior=wash given round-wide pattern (n67 sibling restacked +0.000098) and solo barely above n33 which itself only +0.0002 into CORE. Restack EV moot anyway — clout lineage can't promote."
decided: 2026-06-12
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.968053
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [nn, tabm, pseudo-label, wildcard, clout, finals-slot-2-ONLY, quarantined]
---

## plan
built on:   node_0033 (TabM-richFE). Byte-identical except pseudo-test augmentation.
change:     Retrain TabM-richFE on each train fold AUGMENTED with all test rows hard-labeled by the A4
            public-vote consensus (refs/a4_refresh_2026-06-12 / refs/a4_vote, downweighted ~0.5); honest OOF on
            train val-folds; add-one restack into bank-17.
hypothesis: a student taught by the DISJOINT public consensus imports public-cluster knowledge as an
            honest-CV base with genuinely different errors from bank-17, where self-distilled n67 could not
            (n67's failure: teacher==stack). Teacher here is public-submission-derived, NOT our bank.
target:     solo > n33 0.968053 and bank restack CV > 0.970153.

GPU ~1h. ⚠️ CLOUT-PROVENANCE — MANDATORY: the student's labels derive from the A4 public-LB consensus
(LB 0.97123). Record clout-provenance in BOTH ## notes AND gate_note so a finals slot-1 pick can NEVER silently
inherit the quarantined A4 lineage. This node is eligible for finals slot-2 / diversity ONLY, NEVER honest
slot-1 or champion promotion — even if its CV beats champion.

Fold-honest: pseudo-labels come from public TEST predictions, never our train labels or val folds; OOF scored
only on true train labels. Distinct from n46 (champion-teacher, GBDT) and n67 (soft-distill own bank): different
teacher, hard labels, sample-weighted ~0.5. Parent src nodes/node_0033/src; fs_realmlp_fe per data.md;
A4 labels in refs/a4_refresh_2026-06-12 or node_0068 artifact. Kill: fold-0 val BA < 0.9675. Restack via
champion/src a1 scripts; also try bank17+n67+this.

## notes
