"""node_0001 — LightGBM multiclass with class_weight='balanced'.

One atomic change vs the dumb baseline: a real LightGBM model (5-fold OOF,
frozen folds, native categorical support, class_weight='balanced') is trained
instead of the majority-class constant.

Metric = Balanced Accuracy Score (sklearn, macro-average of per-class recall).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Repo root + comp root (so we can import tools.leakage_scan)
# ---------------------------------------------------------------------------
NODE_SRC = Path(__file__).resolve().parent           # …/node_0001/src
NODE_DIR = NODE_SRC.parent                           # …/node_0001/
# NODE_DIR.parent = nodes/;  .parent again = playground-series-s6e6/
COMP_DIR = NODE_DIR.parent.parent                    # …/playground-series-s6e6/

# Walk up to the repo root (where tools/ lives)
_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add comp src for clean.py helpers (lives in comps/<slug>/src/, not the node src)
COMP_SRC = COMP_DIR / "src"
if str(COMP_SRC) not in sys.path:
    sys.path.insert(0, str(COMP_SRC))
# Verify it can be found
import importlib.util as _ilu
assert _ilu.find_spec("clean") is not None, f"clean.py not found on sys.path; COMP_SRC={COMP_SRC}"

from clean import cast_categoricals, add_color_features, feature_columns  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
RANDOM_BASELINE = 1.0 / 3.0   # balanced accuracy for a random (or majority) predictor

# Label encoding: keep it deterministic
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
IDX2LABEL = {i: lbl for lbl, i in LABEL2IDX.items()}


def score_fn(y_true, y_pred):
    """Official metric: balanced accuracy (labels, not probabilities)."""
    return balanced_accuracy_score(y_true, y_pred)


def make_pipeline():
    """Return a fresh, unfitted LGBMClassifier (so the shuffled control can rebuild)."""
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
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
oof_proba = np.zeros((n, 3), dtype=np.float64)   # per-class probabilities
oof_labels = np.empty(n, dtype=object)            # decoded predicted labels

per_fold_scores = []

print("Running 5-fold OOF …")
for fold_info in folds_list:
    fold_idx = fold_info["fold"]
    val_idx = np.asarray(fold_info["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]

    model = make_pipeline()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[],
    )

    proba = model.predict_proba(X_va)   # shape (|val|, 3); columns in model.classes_ order
    # Map model.classes_ back to LABEL_ORDER indices so oof_proba is consistent
    class_order = list(model.classes_)  # e.g. ['GALAXY', 'QSO', 'STAR']
    for lbl in LABEL_ORDER:
        dest_col = LABEL2IDX[lbl]
        src_col = class_order.index(lbl)
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
# Retrain on ALL train rows → test predictions
# ---------------------------------------------------------------------------
print("Retraining on full train for submission …")
final_model = make_pipeline()
final_model.fit(X, y)
test_proba = final_model.predict_proba(X_test)
test_pred_labels = final_model.predict(X_test)

submission = pd.DataFrame({IDC: test[IDC].values, TARGET: test_pred_labels})
submission = submission[list(sample_sub.columns)]   # byte-match column order
submission.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(submission)} rows)")

# ---------------------------------------------------------------------------
# metrics.md
# ---------------------------------------------------------------------------
metrics_text = f"""# node_0001 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold_scores)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
change: LightGBM multiclass classifier with class_weight='balanced', 5-fold OOF, all features + color indices + native categoricals
"""
(NODE_DIR / "metrics.md").write_text(metrics_text)
print("  wrote metrics.md")
print("Done.")
