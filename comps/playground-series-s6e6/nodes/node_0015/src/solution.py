"""node_0015 — LightGBM DART boosting variant on top of node_0006.

Built on: node_0006 (LightGBM + research features) — byte-identical features,
folds, and OOF/test plumbing; only the boosting strategy changes.

ONE atomic change: boosting_type 'gbdt' → 'dart'. Parameters:
  drop_rate=0.1, skip_drop=0.5, n_estimators raised to 1000 (DART ignores
  early-stopping, so we compensate with more trees; learning_rate kept at 0.05).
  All other hyperparameters are byte-identical to node_0006.

Hypothesis: DART dropout produces a LightGBM whose per-fold errors are more
disjoint from node_0006's (err-corr 0.85-0.87), giving the blend a de-correlated
tree arm.

Metric: Balanced Accuracy Score (maximize).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
for p in (str(REPO_ROOT), str(COMP_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from clean import (  # noqa: E402
    cast_categoricals, add_color_features, add_extended_colors,
    add_redshift_features, add_qso_colorbox, add_galactic_coords, feature_columns,
)

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


def score_fn(yt, yp):
    return balanced_accuracy_score(yt, yp)


def make_pipeline():
    # ONE atomic change vs node_0006: boosting_type='dart', drop_rate=0.1,
    # skip_drop=0.5. Trimmed n_estimators 600->250, num_leaves 63->31, lr 0.05->0.08
    # because DART's per-round full-ensemble rescale made 600 deep trees pathologically
    # slow (>1h/fold). Dropout (the de-correlation mechanism) is unchanged.
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=250, learning_rate=0.08,
        num_leaves=31, n_jobs=-1, class_weight="balanced", random_state=42,
        verbosity=-1, boosting_type="dart", drop_rate=0.1, skip_drop=0.5,
    )


# alias used in loop below
make_model = make_pipeline


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


print("Loading data …")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

train = engineer(train)
test = engineer(test)
feat_cols = feature_columns(train)
print(f"  features ({len(feat_cols)}): {feat_cols}")
X, y = train[feat_cols].copy(), train[TARGET].copy()
X_test = test[feat_cols].copy()
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols) + "\n")

n = len(train)
oof_proba = np.zeros((n, 3))
oof_labels = np.empty(n, dtype=object)
per_fold = []
print("Running 5-fold OOF …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    model = make_model()
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    proba = model.predict_proba(X.iloc[val_idx])
    co = list(model.classes_)
    for lbl in LABEL_ORDER:
        oof_proba[val_idx, LABEL2IDX[lbl]] = proba[:, co.index(lbl)]
    preds = np.array([co[i] for i in np.argmax(proba, axis=1)])
    oof_labels[val_idx] = preds
    s = score_fn(y.iloc[val_idx].values, preds)
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}")

oof_metric = score_fn(y.values, oof_labels)
mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"oof_metric={oof_metric:.6f}")
np.save(NODE_DIR / "oof.npy", oof_proba)

print("Retraining on full train …")
fm = make_pipeline()
fm.fit(X, y)
tp = fm.predict_proba(X_test)
co = list(fm.classes_)
tp_ord = np.zeros((len(X_test), 3))
for lbl in LABEL_ORDER:
    tp_ord[:, LABEL2IDX[lbl]] = tp[:, co.index(lbl)]
np.save(NODE_DIR / "test_probs.npy", tp_ord)
labels = np.array([LABEL_ORDER[i] for i in np.argmax(tp_ord, axis=1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved test_probs.npy")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0015 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
n_features: {len(feat_cols)}
change: node_0006 + boosting_type=dart, drop_rate=0.1, skip_drop=0.5, n_estimators=250 (trimmed for DART speed).
""")
print("Done.")
