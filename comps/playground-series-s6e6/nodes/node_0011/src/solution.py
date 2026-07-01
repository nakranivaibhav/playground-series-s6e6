"""node_0011 — XGBoost on the FULL 28-feature set (research features) — GPU.

One atomic change vs node_0004: use the full engineered feature set (same as node_0006 /
the NNs) instead of the base 15 features. Hyperparameters are byte-identical to node_0004
(n_estim=800, lr=0.06, max_depth=7, subsample/colsample=0.8); only device='cuda' (RTX 5090)
for speed. Closes the feature gap: node_0004 (base feats) = 0.964414; the same +~0.0004 lift
LightGBM got from these features should make this a stronger de-correlated blend arm.
Saves test_probs.npy for the blend.

Metric: Balanced Accuracy Score (macro-average per-class recall) — maximize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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
    cast_categoricals, add_color_features, add_extended_colors,
    add_redshift_features, add_qso_colorbox, add_galactic_coords, feature_columns,
)

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
IDX2LABEL = {i: lbl for lbl, i in LABEL2IDX.items()}


def score_fn(yt, yp):
    return balanced_accuracy_score(yt, yp)


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


def make_model():
    # byte-identical to node_0004 except device='cuda'
    return XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=800, learning_rate=0.06,
        max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", enable_categorical=True, random_state=42,
        eval_metric="mlogloss", early_stopping_rounds=50, verbosity=0,
    )


print("Loading + engineering (full feature set) …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

feat_cols = feature_columns(train)
print(f"  features ({len(feat_cols)}): {feat_cols}")
X, y = train[feat_cols].copy(), train[TARGET].copy()
y_enc = y.map(LABEL2IDX).astype(np.int32)
X_test = test[feat_cols].copy()
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols) + "\n")

n = len(train)
oof_proba = np.zeros((n, 3))
oof_labels = np.empty(n, dtype=object)
per_fold = []
print("Running 5-fold OOF (XGBoost, CUDA) …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    m = make_model()
    sw = compute_sample_weight("balanced", y_enc.iloc[tr_idx].values)
    m.fit(X.iloc[tr_idx], y_enc.iloc[tr_idx], sample_weight=sw,
          eval_set=[(X.iloc[val_idx], y_enc.iloc[val_idx])], verbose=False)
    proba = m.predict_proba(X.iloc[val_idx])              # cols 0,1,2 = LABEL_ORDER
    oof_proba[val_idx] = proba
    preds = np.array([IDX2LABEL[i] for i in np.argmax(proba, axis=1)])
    oof_labels[val_idx] = preds
    s = score_fn(y.iloc[val_idx].values, preds)
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}  (best_iter={m.best_iteration})")

oof_metric = score_fn(y.values, oof_labels)
mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}  (oof_metric={oof_metric:.6f})")
np.save(NODE_DIR / "oof.npy", oof_proba)

print("Retraining on full train …")
fm = XGBClassifier(
    objective="multi:softprob", num_class=3, n_estimators=800, learning_rate=0.06,
    max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    device="cuda", enable_categorical=True, random_state=42, verbosity=0)
fm.fit(X, y_enc, sample_weight=compute_sample_weight("balanced", y_enc.values))
tp = fm.predict_proba(X_test)                            # cols already LABEL_ORDER
np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in np.argmax(tp, axis=1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0011 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
n_features: {len(feat_cols)}
change: XGBoost (node_0004 config) on the FULL 28-feature set + GPU. Saves test_probs.npy.
""")
print("Done.")
