"""node_0004 — XGBoost multiclass with balanced sample weights.

Built on: node_0001 (LightGBM multiclass baseline) structure, inherited byte-
identical except for the model family. The frozen 5-fold stratified CV (folds.json,
seed=42), feature pipeline (raw numeric + 2 categorical + 5 color indices), and
evaluation harness are unchanged.

Change: swap LightGBM for XGBoost (xgboost 3.2.0). XGBoost does not natively
handle pandas 'category' dtype in the sklearn API — we use enable_categorical=True
with tree_method='hist' which IS supported in xgboost 3.x when the input columns
have pandas category dtype. This avoids one-hot expansion entirely; XGBoost
internally maps each category to an integer at split time. The category dtype is
set by cast_categoricals() which takes the levels from each individual dataframe
— a fixed-vocabulary stateless mapping, not a fitted statistic. No transform is
fit on full data or across folds.

Balanced class weighting is implemented via sklearn compute_sample_weight('balanced',
y_train_fold) applied per fold (train-fold class frequencies only, never leaking
val-fold label distribution). XGBoost receives these as the sample_weight argument
to fit().

Metric: Balanced Accuracy Score (sklearn, macro-average per-class recall) — maximize.
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

# ---------------------------------------------------------------------------
# Repo root + comp root
# ---------------------------------------------------------------------------
NODE_SRC = Path(__file__).resolve().parent           # …/node_0004/src
NODE_DIR = NODE_SRC.parent                           # …/node_0004/
COMP_DIR = NODE_DIR.parent.parent                    # …/playground-series-s6e6/

_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMP_SRC = COMP_DIR / "src"
if str(COMP_SRC) not in sys.path:
    sys.path.insert(0, str(COMP_SRC))

import importlib.util as _ilu
assert _ilu.find_spec("clean") is not None, f"clean.py not found; COMP_SRC={COMP_SRC}"

from clean import cast_categoricals, add_color_features, feature_columns  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
RANDOM_BASELINE = 1.0 / 3.0   # balanced accuracy for a random predictor

LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
IDX2LABEL = {i: lbl for lbl, i in LABEL2IDX.items()}


def score_fn(y_true, y_pred):
    """Official metric: balanced accuracy (labels, not probabilities)."""
    return balanced_accuracy_score(y_true, y_pred)


def make_pipeline():
    """Return a fresh, unfitted XGBClassifier.

    enable_categorical=True + tree_method='hist' allows XGBoost 3.x to consume
    pandas category dtype columns natively — no one-hot expansion needed.
    n_jobs=6 as instructed (another model runs concurrently).
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=800,
        learning_rate=0.06,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=6,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())

print(f"  train shape: {train.shape}, test shape: {test.shape}")

# ---------------------------------------------------------------------------
# Feature engineering (stateless — no data leakage possible)
# ---------------------------------------------------------------------------
train = cast_categoricals(train)
train = add_color_features(train)
test = cast_categoricals(test)
test = add_color_features(test)

feat_cols = feature_columns(train)  # excludes id and class
print(f"  feature columns ({len(feat_cols)}): {feat_cols}")

X = train[feat_cols].copy()
y = train[TARGET].copy()
# XGBoost multi:softprob requires integer labels 0..num_class-1
y_enc = y.map(LABEL2IDX).astype(np.int32)
X_test = test[feat_cols].copy()

# ---------------------------------------------------------------------------
# Write features.txt (leakage scanner reads this)
# ---------------------------------------------------------------------------
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols) + "\n")
print("  wrote features.txt")

# ---------------------------------------------------------------------------
# 5-fold OOF CV
# ---------------------------------------------------------------------------
folds_list = folds_data["folds"]
n = len(train)
oof_proba = np.zeros((n, 3), dtype=np.float64)
oof_labels = np.empty(n, dtype=object)

per_fold_scores = []

print("Running 5-fold OOF ...")
for fold_info in folds_list:
    fold_idx = fold_info["fold"]
    val_idx = np.asarray(fold_info["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_va = y_enc.iloc[tr_idx], y_enc.iloc[val_idx]   # integer-encoded
    y_va_str = y.iloc[val_idx]                               # original strings for score_fn

    # Balanced sample weights — computed from train-fold labels ONLY (no leakage)
    sw_tr = compute_sample_weight("balanced", y_tr.values)

    model = make_pipeline()
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    proba = model.predict_proba(X_va)   # shape (|val|, 3); columns are 0,1,2 = GALAXY,QSO,STAR
    # Store in oof_proba directly (integer class indices match LABEL_ORDER by construction)
    oof_proba[val_idx] = proba

    pred_int = np.argmax(proba, axis=1)
    pred_labels = np.array([IDX2LABEL[i] for i in pred_int])
    oof_labels[val_idx] = pred_labels

    fold_score = score_fn(y_va_str.values, pred_labels)
    per_fold_scores.append(fold_score)
    print(f"  fold {fold_idx}: balanced_accuracy = {fold_score:.6f}  "
          f"(best_iteration={model.best_iteration})")

# OOF metric across all training rows
oof_metric = score_fn(y.values, oof_labels)   # y is string labels
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))

print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"oof_metric={oof_metric:.6f}")

# ---------------------------------------------------------------------------
# Save OOF probabilities
# ---------------------------------------------------------------------------
np.save(NODE_DIR / "oof.npy", oof_proba)
print("  saved oof.npy")

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Retrain on ALL train rows -> test predictions
# ---------------------------------------------------------------------------
print("Retraining on full train for submission ...")
final_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=800,
    learning_rate=0.06,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=6,
    random_state=42,
    verbosity=0,
)
sw_full = compute_sample_weight("balanced", y_enc.values)
final_model.fit(X, y_enc, sample_weight=sw_full)
test_pred_int = final_model.predict(X_test)
test_pred_labels = np.array([IDX2LABEL[i] for i in test_pred_int])

submission = pd.DataFrame({IDC: test[IDC].values, TARGET: test_pred_labels})
submission = submission[list(sample_sub.columns)]
submission.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(submission)} rows)")

# ---------------------------------------------------------------------------
# metrics.md
# ---------------------------------------------------------------------------
categorical_encoding = "XGBoost enable_categorical=True + pandas category dtype (native, no one-hot)"
metrics_text = f"""# node_0004 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold_scores)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
change: XGBoost multiclass classifier with balanced sample weights (compute_sample_weight per fold), 5-fold OOF, all features + color indices
categorical_encoding: {categorical_encoding}
"""
(NODE_DIR / "metrics.md").write_text(metrics_text)
print("  wrote metrics.md")
print("Done.")
