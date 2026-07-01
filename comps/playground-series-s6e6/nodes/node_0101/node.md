---
id: node_0101
desc: kNN-graph GraphSAGE base
op: draft
parents: [root]
uses_data: [fs_knngraph]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.898208]
baseline_cv: 0.970355
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "CHEAP-KILL tripped: fold-0 BA=0.898208 < 0.962 threshold. PyG SAGEConv path (torch-geometric 2.8.0 + torch 2.11.0+cu128, CUDA verified). Full-graph GNN with 461k nodes / 5.5M edges converges too slowly / too weakly — internal val BA=0.898 at early-stop ep=40. Graph family axis closed: too weak to reach tier (same pattern as ravi's external GNN, 0.9345). No decorrelation check (fold-0 only ran). err_corr_vs_bank=N/A (cheap-kill)."
leak: clean
lb: null
submitted: null
created: 2026-06-15T06:52Z
decided: 2026-06-15
---

## plan
built on:   root — fresh DRAFT, a genuinely new model FAMILY (graph neural net / message-passing)
            not present anywhere in the bank. Reuses node_0086's data-load + frozen-fold +
            OOF/test-output skeleton (`nodes/node_0086/src/solution.py`) — same I/O contract.
change:     Train a kNN-graph GraphSAGE-style base: build a k-nearest-neighbour graph in
            standardized feature space (colors u-g/g-r/r-i/i-z/u-z + redshift + magnitudes),
            then a 2-3 layer message-passing net that, for each object, AGGREGATES its
            neighbours' features/embeddings (mean + max pool) before classifying. The
            prediction for a row depends on its NEIGHBOURHOOD, not just its own row — an
            inductive bias no tree/MLP/transformer/RBF base in the bank has.
hypothesis: ravi's external GNN was logged as the "only graph model anywhere, top
            decorrelation" but failed forward-select ONLY on strength (solo BA 0.9345,
            ~3.5pp below tier — journal 2026-06-12T14:11Z/14:31Z). The graph family
            decorrelates; the lever is a graph base STRONG enough to clear the tier
            (BA ≥ ~0.965, within 1pp of the ~0.969 base tier) while keeping that
            decorrelation. That is the exact RBF/n062 pattern (decorrelated-but-too-weak)
            but with a family that has real headroom over ravi's weak impl.
target:     Balanced Accuracy maximize. Two-part gate (decorrelation FIRST, then stack-add):
            (1) solo BA ≥ 0.965 (cheap-kill floor; ravi's 0.9345 is the failure to beat);
            (2) mean OOF error-correlation vs the 17-bank (`nodes/node_0070/oof.npy`) < 0.65
            — the RBF closure rule (decorrelation necessary, not sufficient);
            (3) stack-add: LogReg-blend [champion pool + this base] must beat champion
            0.970355 by > 2·sem (≈0.970851). Promote ONLY if all three clear.

HONEST EV: low. The SDSS literature converges ~0.965-0.97 across families (research.md
2026-06-13T11:50Z) and a tabular kNN graph may just relearn the same z-then-color split
trees already own. But it is the single genuinely-untried STRONG family, the decorrelation
evidence is concrete (ravi's graph base was the most decorrelated external candidate), and
the cheap-kill + decorrelation gates bound the wasted compute. Closes the graph-family axis.

HOW (TIGHT — single base, NO full-pool loader; this node must NOT touch the 72-base pool):
- cp nodes/node_0086/src/solution.py → nodes/node_0101/src/solution.py as the I/O SKELETON
  (data load of train.csv/test.csv, frozen-fold loop from folds.json with
  `[np.asarray(f["val_idx"], dtype=int) for f in folds["folds"]]`, class-balanced training,
  OOF/test-prob/submission writing, the log() helper). REPLACE the TabM model + fs_zresid
  feature build with the GraphSAGE pipeline below. Do NOT read node_0091's solution.py.
- FEATURES (fs_knngraph, leak class fit_in_fold): standardized numeric features
  (colors u-g,g-r,r-i,i-z,u-z + raw redshift + raw magnitudes u,g,r,i,z + the two engineered
  categoricals one-hot/ordinal). StandardScaler FIT ON TRAIN-FOLD ROWS ONLY, applied to
  val+test. Keep raw signal here (unlike n086) — a strong base needs z+colors; decorrelation
  comes from the GRAPH aggregation, not from dropping features.
- GRAPH: for each row, its k≈10-15 nearest neighbours in the scaled feature space. CRITICAL
  leak rule — the neighbour INDEX is built on TRAIN-FOLD ROWS ONLY (sklearn NearestNeighbors
  or faiss fit on train-fold); train-fold rows query the index excluding self; val and test
  rows query the same train-fold index. NEVER let a val/test row see another val/test row or
  its own label. This is the fit_in_fold discipline for the graph.
- MODEL — libraries-first (rule 8): TRY `uv add torch-geometric` and verify `import torch`
  still works AND torch.cuda.is_available() is still True (torch is 2.11.0+cu128 — PyG wheels
  may not exist for it). If PyG resolves cleanly, use a 2-3 layer SAGEConv/GATConv. If PyG
  FAILS to build / breaks the torch+cuda import (likely on torch 2.11), FALL BACK to a thin
  pure-torch SAGE: precompute neighbour index lists with sklearn, mean+max aggregate neighbour
  feature vectors via torch index ops, 2-3 Linear+ReLU layers with self+neighbour concat.
  SAY EXPLICITLY in train.log which path was taken and why (rule 8). class_weight='balanced'.
- CHEAP-KILL: run fold-0 ONLY first. If fold-0 solo BA < 0.962, STOP (graph base can't reach
  tier — log it and mark the node valid-but-dead-on-strength). Else run all 5 folds.
- DECORRELATION: after OOF is built, compute mean per-class error-correlation of this base's
  OOF vs nodes/node_0070/oof.npy (the bank-17 reference). Report it in the gate_note. This is
  the n062/RBF rule — a strong base that is NOT decorrelated (<0.65) does not help the stack.
- Produce nodes/node_0101/{oof.npy (577347,3), test_probs.npy (247435,3), submission.csv,
  train.log}. Self-gate (kaggle-leakage): scaler + kNN index + any encoder fit train-fold-only
  (fit_in_fold); folds frozen; OOF full / no-NaN / each train row once; dist sane; schema
  matches sample_submission (uv run tools/validate_submission.py --submission ... --sample ...);
  cv_too_good eyeball. Write gate booleans + cv/sem/folds + leak + the err-corr in gate_note;
  stage: built. Do NOT submit. Everything `uv run`. GPU minutes — background with marker
  DONE=/tmp/playground-series-s6e6_node_0101.done and touch on completion.

## notes
fs_knngraph (fit_in_fold): kNN neighbour index + StandardScaler fit on train-fold rows only;
graph aggregation features for val/test query the train-fold index. If the graph base clears
the tier (BA ≥0.965) AND decorrelates (<0.65) AND lifts the stack, it is the first new base
since FT-Transformer. If it clears the tier but does NOT decorrelate, or decorrelates but is
too weak, the graph-family axis is closed (record which).
