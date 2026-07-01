"""node_0000 — dumb baseline: predict the majority class for every row.

Metric = Balanced Accuracy Score (maximize). The label-metric optimum for a
CONSTANT predictor is the majority class; by construction it yields per-class
recall (1, 0, 0) -> balanced accuracy = 1/3. This proves the data->CV->submission
pipe end-to-end before any model is trained. The constant is fit inside each
train fold only (fit-inside-fold discipline, even for a constant).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

D = Path(__file__).resolve().parents[3]            # comps/<slug>
TARGET, IDC = "class", "id"

tr = pd.read_csv(D / "data/train.csv")
te = pd.read_csv(D / "data/test.csv")
samp = pd.read_csv(D / "data/sample_submission.csv")
folds = json.loads((D / "folds.json").read_text())["folds"]


def const_from(y):                                  # the dumb predictor
    return pd.Series(y).mode().iloc[0]              # majority class label


scores = []
for f in folds:
    va = np.array(f["val_idx"])
    mask = np.ones(len(tr), bool)
    mask[va] = False
    c = const_from(tr.loc[mask, TARGET].values)     # FIT INSIDE FOLD
    pred = np.full(va.shape, c, dtype=object)
    scores.append(balanced_accuracy_score(tr.loc[va, TARGET].values, pred))

cv = float(np.mean(scores))
sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in scores))
print(f"cv={cv:.6f}±{sem:.6f}")

# submission: fit constant on ALL train (test never used), broadcast to test ids
c_full = const_from(tr[TARGET].values)
sub = pd.DataFrame({IDC: te[IDC].values, TARGET: c_full})
sub = sub[list(samp.columns)]                       # byte-match sample column order
sub.to_csv(D / "nodes/node_0000/submission.csv", index=False)
print("majority_class=", c_full, "wrote submission.csv rows", len(sub))
