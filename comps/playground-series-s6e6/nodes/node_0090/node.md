---
id: node_0090
desc: OvR STAR-then-QSO/GALAXY chained RealMLP
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.967670
sem: 0.000224
folds: [0.968335, 0.967430, 0.967092, 0.967470, 0.968025]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "stack-neutral: n76+n90 delta=+0.000048 < 2*sem=0.000508; mean error-corr vs 18-bank=0.7203 (high correlation, chain NN not decorrelated)"
leak: clean
lb: null
submitted: null
created: 2026-06-13T15:51Z
decided: 2026-06-13T17:10Z
---

## plan
built on:   root — fresh draft; reuses the RealMLP-ref recipe + fs_realmlp_fe + PBLD from node_0028 (`nodes/node_0028/src`).
change:     Build the one-vs-rest STRUCTURAL decomposition (research.md highest-leverage idea) as a fresh RealMLP-ref base: model A STAR-vs-rest → P1, model B QSO-vs-GALAXY on the (1−P1) mass → P2, recombine P(STAR)=P1, P(GALAXY)=(1−P1)(1−P2), P(QSO)=(1−P1)P2. Add the recombined OOF as a new decorrelated stack base.
hypothesis: A chained STAR-then-QSO/GALAXY RealMLP concentrates NN capacity on the hard boundary, decorrelating from the monolithic-softmax bank where the GBDT chain (n49/n50) could not.
target:     Balanced Accuracy maximize; stack-add onto n76 must beat champion 0.970153 by >2·sem; decorrelation gate first.

Research.md lines 27-32 flags OvR as the highest-leverage STRUCTURAL idea (mirrors s6e4 1st place); n49/n50 tried it as GBDT chains (solo +0.0011 but stack-neutral). It has NEVER been tried on the STRONG RealMLP-ref family (n28, the 0.969 breakthrough base) — a chained NN concentrates capacity on the hard QSO↔GALAXY boundary differently than a monolithic softmax, which is exactly the de-correlated-base hypothesis (research.md retrain-on-current-features revival logic: n49/n50 may have stack-washed because the GBDT chain re-derives the same tree splits, not because the decomposition caps).

READ: research.md lines 27-32 + 50-52, `nodes/node_0028/src` (RealMLP-ref recipe + fs_realmlp_fe, PBLD), `nodes/node_0049/node.md` + `nodes/node_0050/node.md` (chain mechanics, ordering doesn't matter, recombination formula).

Two binary RealMLP heads on fs_realmlp_fe; recombine per the formula above.

CHEAP-KILL: run fold-0 first; if fold-0 recombined BA < 0.9675 STOP before the remaining folds.

GATE: OOF error-corr vs the 17-bank, then stack-add vs champion >2·sem. GPU. Frozen folds.json.

well: wildcard
