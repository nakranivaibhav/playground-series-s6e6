---
id: node_0072
desc: widen bank — harvest non-Deotte public OOFs (TabPFN-3 etc.)
op: improve
parents: [node_0063]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970211
sem: 0.000251
folds: [0.971103, 0.970119, 0.969771, 0.969718, 0.970343]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-12
decided: null
tags: [public-bank, stack, external-oof, look-outside, finals-slot-1-eligible]
---

## plan
built on:   node_0063 champion (+ node_0070's widened set once it lands — coordinate: this node should start
            from node_0070's selected base set, NOT re-from bank-17, so the two widens compose).
change:     Hunt and ingest ADDITIONAL public OOF artifacts beyond Deotte's 17 AND beyond node_0070's
            pilkwang/ravi set — esp. philippsinger's TabPFN-3 stacker, kospintr's multi-model baseline,
            flexonafft blender, and the meta-stacker-0.97105 kernel. Verify each is a 100% row-match to our
            frozen StratifiedKFold(5,shuffle,42) like n63 did, then forward-select into the widened bank with
            the same balanced-LogReg + DE-threshold meta.
hypothesis: the public ecosystem holds strong de-correlated bases (esp. full-scale TabPFN-3) absent from
            Deotte's 17; widening the bank is the proven mechanism that made n63 champion.
target:     Balanced Accuracy maximize; beats champion if CV > 0.970153 by >2·sem.

CPU mostly. Already-pulled candidates may sit in refs/ (pull_philippsinger_tabpfn-3-stacker,
pull_kospintr_..., pull_flexonafft_blender..., meta-stacker-0.97105.py) — read them first for OOF outputs or
reproducible per-fold recipes; also `kaggle kernels list -s s6e6 --sort-by voteCount` for fresh ones (auth:
source .env; export KAGGLE_KEY=$KAGGLE_TOKEN; uv run --no-sync kaggle ...). Fold-match verify recipe in
champion/src/a1_full_merge.py — a mismatched OOF is VOID (discard, NEVER re-index/re-CV ad hoc). A foundation-
model base (TabPFN-3 at full public scale) is exactly the de-correlated-AND-strong arm our in-house TabPFN
(n22/25/27 capped 0.949) never delivered. If a kernel must be re-run per-fold, cap at ONE model. Add-one A/B
each new base before the full widened restack.

ORCHESTRATOR NOTE: build AFTER node_0070 so they compose into one widest bank (don't double-count pilkwang/ravi).

## notes
