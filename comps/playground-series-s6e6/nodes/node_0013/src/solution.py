"""node_0013 — champion LightGBM (node_0006 config) + LEAK-SAFE positional features.

Motivated by the drop-column study: `delta` (declination) is the single most irreplaceable
feature, i.e. the synthetic data has strong class-vs-sky-position structure that raw coords +
axis-aligned splits exploit poorly. This node adds positional features, ALL LABEL-FREE:

  clean.add_positional_features (unit-tested, row-wise stateless):
    sin/cos RA (0°/360° wrap fix), unit-sphere cartesian sx/sy/sz, delta×redshift interactions,
    and sky_cell = a coarse (RA 10° × Dec 5°) grid-cell id as a NATIVE CATEGORICAL — LightGBM
    learns the per-region class tendency INSIDE each fold (no manual target encoding → no leak).
  knn_dist5 (here): distance to the 5th-nearest TRAINING object on the unit sphere — a local
    sky-density proxy. The KDTree reference is the TRAIN positions only and uses NO labels.

Leak-safety is structural: no feature touches the target, so the shuffled-label control must
collapse to 1/3. sky_cell's test categories are aligned to the TRAIN vocabulary (train-defined;
test-only cells → missing), the correct (test-never-informs-train) direction.

Metric = Balanced Accuracy Score (macro-average per-class recall) — maximize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent
_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
for p in (str(_r), str(COMP_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from clean import (  # noqa: E402
    cast_categoricals, add_color_features, add_extended_colors, add_redshift_features,
    add_qso_colorbox, add_galactic_coords, add_positional_features, feature_columns,
)

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


def score_fn(yt, yp):
    return balanced_accuracy_score(yt, yp)


def make_model():
    # identical to node_0001 / node_0006 (CV-proven near-optimal)
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
        num_leaves=63, n_jobs=-1, class_weight="balanced", random_state=42, verbosity=-1,
    )


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    df = add_positional_features(df)        # NEW: leak-safe positional features
    return df


print("Loading + engineering …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

# align sky_cell test categories to the TRAIN vocabulary (test-only cells -> missing)
train["sky_cell"] = train["sky_cell"].astype("category")
test["sky_cell"] = test["sky_cell"].cat.set_categories(train["sky_cell"].cat.categories)
print(f"  sky_cell: {train['sky_cell'].cat.categories.size} train cells; "
      f"{int(test['sky_cell'].isna().sum())} test rows in unseen cells -> missing")

# knn_dist5: distance to 5th-nearest TRAINING object on the unit sphere (label-free density)
pos_tr = train[["sx", "sy", "sz"]].to_numpy()
pos_te = test[["sx", "sy", "sz"]].to_numpy()
tree = cKDTree(pos_tr)
d_tr, _ = tree.query(pos_tr, k=6)          # col 0 = self (dist 0)
d_te, _ = tree.query(pos_te, k=5)
train["knn_dist5"] = d_tr[:, 5]
test["knn_dist5"] = d_te[:, 4]
print(f"  knn_dist5: train median {np.median(train['knn_dist5']):.4g}, test median {np.median(test['knn_dist5']):.4g}")

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
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}  (oof_metric={oof_metric:.6f})  parent node_0006=0.965004")
np.save(NODE_DIR / "oof.npy", oof_proba)

print("Retraining on full train …")
fm = make_model()
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
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0013 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})   parent node_0006=0.965004
n_features: {len(feat_cols)}
change: node_0006 LightGBM + leak-safe positional features (sin/cos RA, unit-sphere xyz,
delta×redshift, sky_cell native categorical, knn_dist5 sky-density). All label-free.
""")
print("Done.")
