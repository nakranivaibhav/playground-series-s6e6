#!/usr/bin/env python3
"""node_0129 — ensemble-of-ensembles: LogReg meta over our top 6 stacks.

A combine node over registered stack nodes (n091/n070/n116/n063/n041/n040), each
itself a meta over the 63-base bank. The meta form is the champion's: a balanced
multinomial LogReg @C=0.003 on clipped log-probs, fit per-fold (frozen folds) for
the honest OOF, then refit on ALL train for the test prediction.

Leak-safety: each input is a fold-honest registered node's oof.npy (train) +
test_probs.npy (test); the meta is fit on outer-train only per fold. Frozen folds.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

NODE = Path(__file__).resolve().parent.parent
COMP = NODE.parent.parent
STACKS = ["node_0091", "node_0070", "node_0116", "node_0063", "node_0041", "node_0040"]
C = 0.003

y = pd.read_csv(COMP / "data" / "train.csv", usecols=["class"])["class"].astype(str)
classes = sorted(y.unique())                       # ['GALAXY','QSO','STAR']
y = y.map({c: i for i, c in enumerate(classes)}).to_numpy()

folds = json.load(open(COMP / "folds.json"))["folds"]
foldid = np.empty(len(y), int)
for f in folds:
    foldid[np.asarray(f["val_idx"], int)] = f["fold"]

def clog(o):                                        # clipped log-probs
    return np.log(np.clip(o.astype(np.float64), 1e-6, 1.0))

Xtr = np.hstack([clog(np.load(COMP / "nodes" / n / "oof.npy")) for n in STACKS])
Xte = np.hstack([clog(np.load(COMP / "nodes" / n / "test_probs.npy")) for n in STACKS])

# honest nested-fold OOF
oof = np.zeros((len(y), len(classes)))
for f in range(len(folds)):
    tr, va = foldid != f, foldid == f
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr[tr], y[tr])
    oof[va] = clf.predict_proba(Xtr[va])
per_fold = [balanced_accuracy_score(y[foldid == f], oof[foldid == f].argmax(1)) for f in range(len(folds))]
cv = float(np.mean(per_fold)); sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print(f"cv={cv:.6f} sem={sem:.6f} per_fold={[round(x,6) for x in per_fold]}")

# test prediction: refit on ALL train
clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
clf.fit(Xtr, y)
test_probs = clf.predict_proba(Xte)

np.save(NODE / "oof.npy", oof.astype(np.float32))
np.save(NODE / "test_probs.npy", test_probs.astype(np.float32))
sub = pd.read_csv(COMP / "data" / "sample_submission.csv")
sub["class"] = [classes[i] for i in test_probs.argmax(1)]
sub.to_csv(NODE / "submission.csv", index=False)
print(f"wrote oof {oof.shape}, test_probs {test_probs.shape}, submission {len(sub)} rows")
print(f"test pred dist: {pd.Series(sub['class']).value_counts(normalize=True).round(4).to_dict()}")
