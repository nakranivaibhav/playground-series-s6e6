"""node_0003 — CatBoost multiclass with auto_class_weights='Balanced'.

Built on: node_0001 (LightGBM baseline). Inherits byte-identical fold loop,
OOF structure, feature engineering (cast_categoricals, add_color_features,
feature_columns from clean.py), shuffled-label control, and submission logic.

ONE atomic change: the model is swapped from LGBMClassifier to CatBoostClassifier.
CatBoost receives the two categorical columns (spectral_type, galaxy_population)
natively via cat_features= (no one-hot encoding needed). Class balancing uses
auto_class_weights='Balanced' computed per train-fold inside the model. Early
stopping is applied against the val fold to limit wall-clock time.

Metric = Balanced Accuracy Score (sklearn, macro-average of per-class recall),
direction = maximize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier, Pool

# ---------------------------------------------------------------------------
# Repo root + comp root paths
# ---------------------------------------------------------------------------
NODE_SRC = Path(__file__).resolve().parent           # …/node_0003/src
NODE_DIR = NODE_SRC.parent                           # …/node_0003/
COMP_DIR = NODE_DIR.parent.parent                    # …/playground-series-s6e6/

# Walk up to repo root (where tools/ lives)
_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add comp src for clean.py helpers
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

# Categorical column names (CatBoost handles them natively as strings)
CAT_COLS = ["spectral_type", "galaxy_population"]


def score_fn(y_true, y_pred):
    """Official metric: balanced accuracy (labels, not probabilities)."""
    return balanced_accuracy_score(y_true, y_pred)


def make_pipeline():
    """Return a fresh, unfitted CatBoostClassifier (so the shuffled control can rebuild)."""
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",   # fit inside-fold by CatBoost automatically
        random_seed=42,
        thread_count=6,
        od_type="Iter",
        od_wait=50,
        verbose=False,
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
# Feature engineering (stateless row-wise — no leakage possible)
# ---------------------------------------------------------------------------
train = cast_categoricals(train)
train = add_color_features(train)
test = cast_categoricals(test)
test = add_color_features(test)

feat_cols = feature_columns(train)   # excludes id and class
print(f"  feature columns ({len(feat_cols)}): {feat_cols}")

# Identify which feat_cols are categorical (for CatBoost cat_features arg)
cat_feature_indices = [i for i, c in enumerate(feat_cols) if c in CAT_COLS]
print(f"  cat_feature_indices (within feat_cols): {cat_feature_indices} -> {[feat_cols[i] for i in cat_feature_indices]}")

X = train[feat_cols].copy()
# CatBoost needs categoricals as strings (not pandas category dtype)
for c in CAT_COLS:
    if c in X.columns:
        X[c] = X[c].astype(str)

y = train[TARGET].copy()

X_test = test[feat_cols].copy()
for c in CAT_COLS:
    if c in X_test.columns:
        X_test[c] = X_test[c].astype(str)

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
    y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]

    model = make_pipeline()

    # Build CatBoost Pools (enables early stopping)
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_feature_indices)
    val_pool = Pool(X_va, label=y_va, cat_features=cat_feature_indices)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    proba = model.predict_proba(val_pool)   # shape (|val|, 3)
    class_order = model.classes_            # e.g. ['GALAXY', 'QSO', 'STAR']

    for lbl in LABEL_ORDER:
        dest_col = LABEL2IDX[lbl]
        src_col = list(class_order).index(lbl)
        oof_proba[val_idx, dest_col] = proba[:, src_col]

    pred_labels = np.array([class_order[i] for i in np.argmax(proba, axis=1)])
    oof_labels[val_idx] = pred_labels

    fold_score = score_fn(y_va.values, pred_labels)
    per_fold_scores.append(fold_score)
    print(f"  fold {fold_idx}: balanced_accuracy = {fold_score:.6f}")

# OOF metric on all training rows
oof_metric = score_fn(y.values, oof_labels)
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
final_model = make_pipeline()
full_pool = Pool(X, label=y, cat_features=cat_feature_indices)
test_pool = Pool(X_test, cat_features=cat_feature_indices)
final_model.fit(full_pool)

test_pred_labels = final_model.predict(test_pool).flatten()
test_proba = final_model.predict_proba(test_pool)
np.save(NODE_DIR / "test_probs.npy", test_proba)
print(f"  wrote test_probs.npy {test_proba.shape}")

submission = pd.DataFrame({IDC: test[IDC].values, TARGET: test_pred_labels})
submission = submission[list(sample_sub.columns)]   # byte-match column order
submission.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(submission)} rows)")

# ---------------------------------------------------------------------------
# metrics.md
# ---------------------------------------------------------------------------
metrics_text = f"""# node_0003 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold_scores)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
change: CatBoost multiclass classifier with auto_class_weights='Balanced', native categoricals via cat_features, 5-fold OOF, all features + color indices
"""
(NODE_DIR / "metrics.md").write_text(metrics_text)
print("  wrote metrics.md")
print("Done.")
