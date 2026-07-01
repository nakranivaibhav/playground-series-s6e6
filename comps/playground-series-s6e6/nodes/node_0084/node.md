---
id: node_0084
desc: Revive strong discards into best-honest stack
op: combine
parents: [node_0076, node_0067, node_0074]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970240
sem: 0.000157
folds: [0.970851, 0.970133, 0.969965, 0.970171, 0.970083]
baseline_cv: 0.970227
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CLOUT flag: node_0074 (A4-vote teacher) was fwd-selected into clout stack (cv=0.970299). Clout stack is slot-2 only. Honest stack (n076+n067) cv=0.970240 does NOT beat champion 0.970153 by >2*sem (threshold=0.970466). No promote."
leak: clean
lb: null
submitted: null
created: 2026-06-13T11:56Z
decided: 2026-06-13T14:34Z
---

## plan
built on:   node_0076 (bank17+FT-T+bagged-argmax stack, the best-honest base) — reproduced EXACTLY first; node_0067 and node_0074 contribute only their SAVED OOF/test_probs as candidate additions.
change:     Cheap re-stack: A/B-add the SAVED OOF of strong discards — node_0067 (transductive-distill TabM, solo lifted +0.00136) and node_0074 (disjoint-teacher TabM, solo +0.000475, the strongest de-correlated TabM-tier OOF we have) — as candidate fwd-select additions onto the node_0076 bank17+FT-T+bagged-argmax stack. No retraining; promote only if a discard lifts CV > 2·sem.
hypothesis: n67/n74's de-correlated TabM-tier OOFs were judged against the weaker bank-17 baseline; against the stronger n076 stack one may now clear fwd-select where it previously washed.
target:     balanced accuracy maximize; beats parent if cv > node_0076 0.970227 by >2·sem (~0.000488).

Revival habit (re-stack arm). The best-honest base (n076) consolidated FT-T + bagging +
argmax AFTER n67/n74 were judged. Their verdicts (washes/clout-only) were against the
OLDER bank-17 baseline, so they are stale vs the current best stack — exactly the
residual-shift case the policy says to re-examine. n67 (cv 0.969414, +0.00136 solo vs
parent) and n74 (cv 0.968528, +0.000475 solo — disjoint-teacher that genuinely worked) are
the two strongest de-correlated complete classifiers among discards.

Load their nodes/<id>/oof.npy + test_probs.npy and forward-select onto n076's stacked
baseline using the exact n70 protocol (journal 2026-06-12T15:01Z): reproduce the n076
baseline EXACTLY first (HARD assert ~0.970227), then add one candidate at a time, keep only
if delta>0. n74 is CLOUT-tainted (A4-vote teacher) — if it is fwd-selected it can be slot-2
only, NOT honest slot-1; keep that flag. Trust the CV (these are complete honest
classifiers, not narrow specialists — NOT a node_0047-style error-pocket revival).

Read node_0067/node.md and node_0074/node.md.

well: exploit
