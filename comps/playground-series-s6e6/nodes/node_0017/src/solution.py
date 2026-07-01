"""node_0017 — improve: per-class weight tuning on champion blend OOF (fold-honest).

Built on: node_0010 (champion 4-arm blend n6/n4/n1/n9, cv=0.965889). The saved
fold-honest blended OOF probability matrix (node_0010/oof.npy) and test probabilities
(node_0010/test_probs.npy) are reused byte-identical — no model is retrained.

Change: Instead of argmax(prob), predict argmax(prob * w) where w is a 3-vector
(one per class: GALAXY, QSO, STAR). Grid-search w over a uniform grid to maximise
balanced accuracy. Tuning is FOLD-HONEST: for each of the 5 folds, w is fit on the
other 4 folds' OOF and scored on the held-out fold. The final test-time w is fit
on the full blended OOF. This post-hoc correction removes any per-class bias in
the blended posterior that argmax alone cannot fix.

Efficiency: argmax(p*w) is invariant to global scale, so we fix w[2]=1 and search
the 2D grid (w[0], w[1]) in [0, 4] with step 0.1 = 41x41=1681 combos. Per-combo
argmax is a single numpy call over n rows. Total: ~1681*6 = ~10k numpy argmax calls.

Metric: Balanced Accuracy Score = macro-average per-class recall (maximize).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

TARGET, IDC, DIRECTION = "class", "id", "maximize"
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
N_CLASSES = 3

# 2D grid: fix w[2]=1, search w[0] and w[1] independently
W_LO, W_HI, W_STEP = 0.0, 4.0, 0.1  # 41x41 = 1681 combos


def fast_balacc(y_int: np.ndarray, pred_int: np.ndarray) -> float:
    return float(np.mean([(pred_int[y_int == c] == c).mean()
                          for c in range(N_CLASSES) if (y_int == c).any()]))


def search_weights_2d(probs: np.ndarray, y_int: np.ndarray, rows: np.ndarray) -> tuple:
    """Pure-Python outer loop over grid; inner argmax vectorized over rows."""
    pts = np.arange(W_LO, W_HI + W_STEP * 0.5, W_STEP)
    p = probs[rows]   # view, no copy
    y = y_int[rows]
    best_score, best_w = -1.0, (1.0, 1.0, 1.0)
    for w0 in pts:
        for w1 in pts:
            w = np.array([w0, w1, 1.0], dtype=np.float32)
            preds = np.argmax(p * w, axis=1)
            s = fast_balacc(y, preds)
            if s > best_score + 1e-12:
                best_score, best_w = s, (float(w0), float(w1), 1.0)
    return best_w, best_score


print("Loading saved blended OOF + test probability matrices from node_0010 ...")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
n = len(train)

oof_probs = np.load(COMP_DIR / "nodes/node_0010/oof.npy").astype(np.float32)
test_probs = np.load(COMP_DIR / "nodes/node_0010/test_probs.npy").astype(np.float32)

assert oof_probs.shape == (n, N_CLASSES), f"oof shape: {oof_probs.shape}"
assert test_probs.shape == (len(test), N_CLASSES), f"test shape: {test_probs.shape}"

y_int = train[TARGET].map(LABEL2IDX).to_numpy()
all_idx = np.arange(n)
fold_val = [np.asarray(fi["val_idx"]) for fi in folds_list]

pts = np.arange(W_LO, W_HI + W_STEP * 0.5, W_STEP)
n_combos = len(pts) ** 2
print(f"2D grid (w[2]=1): [{W_LO},{W_HI}] step {W_STEP} -> {n_combos} combos")

print("Fold-honest per-class weight search ...")
per_fold, per_fold_w = [], []
for fi, val_idx in zip(folds_list, fold_val):
    other = np.setdiff1d(all_idx, val_idx)
    w_f, _ = search_weights_2d(oof_probs, y_int, other)
    w_arr = np.array(w_f, dtype=np.float32)
    preds_val = np.argmax(oof_probs[val_idx] * w_arr, axis=1)
    score_f = fast_balacc(y_int[val_idx], preds_val)
    per_fold.append(score_f)
    per_fold_w.append(w_f)
    print(f"  fold {fi['fold']}: w=({w_f[0]:.1f},{w_f[1]:.1f},{w_f[2]:.1f}) "
          f"-> balanced_accuracy = {score_f:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}  (mean+/-sem: {mean_cv:.6f}+/-{sem_cv:.6f}  HONEST per-class weight)")

print("Fitting final w on full OOF ...")
w_final, full_oof_s = search_weights_2d(oof_probs, y_int, all_idx)
print(f"  final w=({w_final[0]:.1f},{w_final[1]:.1f},{w_final[2]:.1f})  "
      f"full_oof_balacc={full_oof_s:.6f}")

np.save(NODE_DIR / "oof.npy", oof_probs.astype(np.float64))
np.save(NODE_DIR / "test_probs.npy", test_probs.astype(np.float64))

w_arr_final = np.array(w_final, dtype=np.float32)
test_preds = np.argmax(test_probs * w_arr_final, axis=1)
labels = np.array([LABEL_ORDER[i] for i in test_preds])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

(NODE_SRC / "features.txt").write_text(
    "blend_oof_probs_GALAXY\nblend_oof_probs_QSO\nblend_oof_probs_STAR\n")
print("Done.")
