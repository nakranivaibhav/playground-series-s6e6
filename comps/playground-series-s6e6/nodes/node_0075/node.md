---
id: node_0075
desc: 5-seed bagged TabM-richFE, then restack
op: improve
parents: [node_0033]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968244
sem: 0.000225
folds: [0.969052, 0.968310, 0.967694, 0.968033, 0.968132]
baseline_cv: 0.968053
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "KILL triggered: 2-seed avg cv=0.968244 < threshold 0.968353 — seeds 3-5 stopped. Restack: bank17+n75bag=0.970168, bank17+n75bag+n67=0.970237. Neither beats champion (0.970153) by >2·sem=0.00055."
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [nn, tabm, seed-bag, exploit]
---

## plan
built on:   node_0033 (TabM on rich fs_realmlp_fe). Byte-identical except seed-bagging.
change:     Bag n33's TabM-richFE over 5 seeds (average per-fold probs) → one stronger TabM OOF; add-one
            restack into bank-17 and the bank17+n67 pair.
hypothesis: seed-averaging lifts TabM solo ~+0.0005–0.001, enough to push its already-positive stack marginal
            past fold-noise.
target:     2-seed-avg solo CV >= 0.968353 to continue; final solo > 0.968053; restack CV > 0.970153.

GPU ~2-3h (5× n33). INTERIM KILL: after seed-2, if 2-seed-avg solo CV < 0.968353 (n33 + 0.0003), STOP seeds
3-5 and record the wash (priors weak: n19 TabM bag washed, RealMLP 3-seed +0.00003).

CONTEXT: n33 is our strongest, most stack-relevant in-house de-corr base (single seed 0.968053, the only
in-house add-one ever positive: +0.0002 into CORE, bank17+n33+n67 hit +0.000098). rich-FE TabM has never been
bagged. Parent src nodes/node_0033/src; fs_realmlp_fe per data.md (src/clean.py); folds.json frozen. After 5
seeds: score solo CV, then A/B restacks via champion/src a1 scripts (bank17+bag; bank17+bag+n67). Promote bar
2·sem ~0.00055 vs 0.970153.

## notes
KILL triggered after 2 seeds: seed1 (42) cv=0.968053, seed2 (123) cv=0.967958, 2-seed avg=0.968244 < 0.968353.
Seed1 exactly matches n_0033 (same fold scores: 0.968562, 0.968216, 0.967941, 0.967740, 0.967804).
Restack probes: bank17=0.970153 (champion), bank17+n75bag=0.970168 (+0.000015), bank17+n75bag+n67=0.970237 (+0.000084).
None beats champion by fold-noise (2·sem=0.00055). Bag is a confirmed wash — priors correct.
