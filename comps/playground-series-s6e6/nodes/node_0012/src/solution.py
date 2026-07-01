"""node_0012 — CatBoost on the FULL 28-feature set (research features) — GPU.

One atomic change vs node_0003: use the full engineered feature set (same as node_0006 /
the NNs) instead of the base 15 features. Hyperparameters are byte-identical to node_0003
(iterations=800, lr=0.06, depth=7, auto_class_weights='Balanced'); only task_type='GPU'
(RTX 5090) for speed. node_0003 (base feats) was undertrained at 0.961294 — the research
features should help. Saves test_probs.npy for the blend.

Metric: Balanced Accuracy Score (macro-average per-class recall) — maximize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier, Pool

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
CAT_COLS = ["spectral_type", "galaxy_population"]


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
    # byte-identical to node_0003 except task_type='GPU'
    return CatBoostClassifier(
        iterations=800, learning_rate=0.06, depth=7,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=42,
        task_type="GPU", devices="0", od_type="Iter", od_wait=50, verbose=False,
    )


print("Loading + engineering (full feature set) …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

feat_cols = feature_columns(train)
print(f"  features ({len(feat_cols)}): {feat_cols}")
cat_idx = [i for i, c in enumerate(feat_cols) if c in CAT_COLS]
X = train[feat_cols].copy()
for c in CAT_COLS:
    X[c] = X[c].astype(str)
y = train[TARGET].copy()
X_test = test[feat_cols].copy()
for c in CAT_COLS:
    X_test[c] = X_test[c].astype(str)
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols) + "\n")

n = len(train)
oof_proba = np.zeros((n, 3))
oof_labels = np.empty(n, dtype=object)
per_fold = []
print("Running 5-fold OOF (CatBoost, GPU) …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    m = make_model()
    tr_pool = Pool(X.iloc[tr_idx], label=y.iloc[tr_idx], cat_features=cat_idx)
    va_pool = Pool(X.iloc[val_idx], label=y.iloc[val_idx], cat_features=cat_idx)
    m.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    proba = m.predict_proba(va_pool)
    co = list(m.classes_)
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
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}  (oof_metric={oof_metric:.6f})")
np.save(NODE_DIR / "oof.npy", oof_proba)

print("Retraining on full train …")
fm = make_model()
fm.fit(Pool(X, label=y, cat_features=cat_idx))
tp_pool = Pool(X_test, cat_features=cat_idx)
tp_raw = fm.predict_proba(tp_pool)
co = list(fm.classes_)
tp = np.zeros((len(X_test), 3))
for lbl in LABEL_ORDER:
    tp[:, LABEL2IDX[lbl]] = tp_raw[:, co.index(lbl)]
np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in np.argmax(tp, axis=1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0012 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
n_features: {len(feat_cols)}
change: CatBoost (node_0003 config) on the FULL 28-feature set + GPU. Saves test_probs.npy.
""")
print("Done.")
