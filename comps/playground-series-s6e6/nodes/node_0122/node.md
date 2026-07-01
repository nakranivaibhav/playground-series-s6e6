---
id: node_0122
desc: region-interacted meta (redshift-band x base)
op: improve
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970286
sem: 0.000286
folds: [0.971289, 0.970010, 0.969808, 0.969781, 0.970543]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-17
decided: 2026-06-17T17:03Z
tags: [stack, meta, region-interaction, redshift-band, exploit, improve, structural-gate]
---

## plan
built on:   node_0091 (champion: C-tuned balanced multinomial LogReg mega-stack over the
            FULL 63-base pool of clipped log-probs, nested in-fold C grid, frozen folds, plain
            argmax · cv 0.970355 · lb 0.97121). The meta FAMILY, the clip+normalise of base
            log-probs, the nested-in-fold C selection, the frozen folds, and argmax ALL stay
            byte-identical. ONLY the meta feature matrix changes.
change:     Augment the meta input with REGION-INTERACTION columns: for each base's class
            log-prob column, add its product with a redshift-band one-hot (stateless bands,
            fixed edges [-inf, 0.0025, 0.15, 0.5, 1.0, 2.0, inf]). This lets the LogReg learn a
            PER-REGION (per redshift band) trust weight for each base instead of one global
            weight. One atomic change vs n091: the feature matrix gains the interaction block;
            everything else identical.
hypothesis: the champion meta is REGION-BLIND — one global coefficient per base. The
            hidden-signal sweep (probes/hidden_signal_sweep.csv, journal 2026-06-17) proved 12
            bases carry HOLDOUT-significant complementary fix-blocks that are region-conditional
            (n118 fixes 4098 GALAXY rows concentrated low-z; McNemar p 1e-40..1e-11) which a
            global-linear meta structurally cannot capture. Region-interacted features give the
            meta the capacity to take a base's signal WHERE it helps and drop it where it hurts —
            the soft, probability-based version of the hard region-gate that failed
            (probes/capturability_check.py: the crude argmax-vote override couldn't beat working
            OR holdout). Lower odds after the hard gate failed, but this is the expressive,
            DEFINITIVE test of the region-blindness hypothesis.
target:     Balanced Accuracy maximize. Judged on the NEW structural gate (validation.md
            2026-06-17), NOT the raw 2*sem scalar: promote-eligible iff paired-bootstrap
            P(node > n091) >= 0.90 on the OOF AND the holdout net-fix vs n091 is positive +
            McNemar-significant (must hold on the inviolable fold-4 holdout, not just working).
            SAFE WASH if entangled: L2 shrinks the interaction coefficients to ~0 and it
            reproduces n091 (a clean wash, not an n0047 mirage).

EXPLOIT well — the one expressive combine sub-axis left after the hard region-gate failed.
READ: champion/src/solution.py (the exact OOF-ingest + clip+norm-of-log-probs +
LogisticRegressionCV nested-C loop to copy — edit ONLY the feature-matrix construction to append
the interaction block); probes/hidden_signal_sweep.csv (which bases carry holdout-sig complementary
signal — the interaction is most likely to bite on n118/n030/n060/n085/n094/n099 and the
near-champion n084/n104); probes/capturability_check.py (why the HARD gate failed — fixes/breaks
interleaved; the soft meta is the more-expressive retry); validation.md (the structural gate to
judge by). redshift is a leak-safe raw input present in BOTH train and test; the band one-hot is
STATELESS (fixed edges, no fit) so uses_data stays []. The meta's nested-C is fit per-fold exactly
as n091. GUARD: first reproduce n091's 0.970355 with the interaction block ZEROED (a sanity assert
that the base pipeline is byte-identical) before adding interactions. CPU-only, minutes; uv run
--no-sync. After scoring, run tools/pred_diagnostic.py vs n091 (working + holdout + bootstrap) and
report whether L2 kept or shrank the interaction coefficients.

## notes
well = exploit. The definitive close/open on the region-conditional combine lever.

RESULT (WASH — does not promote; region-conditional combine lever CLOSED). Multi-arm A/B:
sanity assert (TIGHT pool, interactions zeroed) reproduced 0.970301 ≈ n091 0.970355 (PASS,
base pipeline byte-identical). WITH redshift-band × base-logprob interactions: TIGHT+interact
cv 0.970286 (sem 0.000286), FULL+interact cv 0.970257 — both BELOW champion 0.970355 and below
the matched zeroed-assert 0.970301 (interactions cost ~-0.000015 same-pool). Structural gate
(tools/pred_diagnostic.py vs n091): bootstrap P(>champ) = 0.185, flip net -33 (McNemar p=0.382,
pure churn), and critically NET-NEGATIVE in the low-z band (-20) where n118's hidden signal lives.
KEY: L2 did NOT shrink the interaction coefficients (interact/base |coef| ratio = 0.5152) — the
meta judged the regional structure worth using IN-FOLD, yet it nets flat-to-negative OOF → the
regional signal is in-fold structure that does NOT generalize. With the hard region-gate
(probes/capturability_check.py, also failed), this CLOSES the region-conditional combine lever
from both the crude (vote) and expressive (soft interacted meta) sides. The 12-node holdout-
significant complementary signal (probes/hidden_signal_sweep.csv) is REAL but genuinely
UNCAPTURABLE — fixes and breaks entangled below any region/class conditioning. Decorrelation wall
now confirmed at the meta/combine level with the most expressive available tool.
