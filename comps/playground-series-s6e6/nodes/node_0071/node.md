---
id: node_0071
desc: 5-seed bagged DCN, restack best-LB base
op: improve
parents: [node_0055]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966273
sem: 0.000226
folds: [0.967102, 0.966168, 0.965784, 0.965993, 0.966319]
baseline_cv: 0.966037
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "2-seed interim kill triggered: cv=0.966273 < threshold 0.966337 (diff -0.000064); +0.000236 vs parent n55; restack not run; 2-seed only artifact"
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [nn, dcn, seed-bag, exploit, best-lb-base]
---

## plan
built on:   node_0055 (DCN/CrossNet on fs_realmlp_fe). Byte-identical except seed-bagging.
change:     Bag n55's DCN over 5 seeds (average per-fold probs) → one stronger DCN OOF; add-one restack into
            bank-17 (champion a1 scripts). Interim kill: after seed-2, if 2-seed-avg solo CV < 0.966337, stop.
hypothesis: seed-bagging lifts DCN solo enough that its already-LB-positive de-correlation finally clears CV
            fold-noise in the stack.
target:     2-seed-avg solo >= 0.966337 to continue; restack CV > 0.970153.

GPU ~1-2h. CONTEXT: n55 is the standout LB signal in the whole graph — solo CV 0.966037 (CV-neutral in stack)
yet its bank restack PROBE hit LB 0.97083, the BEST LB (+0.0004 over champion 0.97073) → its error structure
is genuinely de-correlated where the public test lives. The weak half is solo strength; seed-bagging is the
cheapest solo lift, never applied to DCN. Parent src nodes/node_0055/src; fs_realmlp_fe per data.md; folds.json
frozen. Score solo CV, then A/B restacks (bank17+DCNbag; bank17+DCNbag+n33). If CV again neutral but a finals
candidate on LB grounds, SURFACE — never auto-spend a slot.

## notes
