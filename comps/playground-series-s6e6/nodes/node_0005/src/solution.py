"""node_0005 — tuned/regularized LightGBM (improve on node_0001).

One atomic change vs node_0001: fix its under-regularization and undertraining.
node_0001 used LightGBM defaults (min_child_samples=20, n_estimators=500, no early
stopping) which is too low-regularized at 577k rows. Here:
  - min_child_samples=200  (research: 100-2000 at this scale; biggest reg knob)
  - learning_rate=0.03 + n_estimators=3000 + early_stopping(100)  (the real lever)
  - num_leaves=127, subsample=0.8 (bagging_freq=1), colsample_bytree=0.8, reg_lambda=1
Everything else identical (features, folds, class_weight='balanced', label encoding).
Also SAVES test_probs.npy (node_0001 computed but discarded test probabilities) so the
blend node can produce a submission cheaply without retraining.

Metric = Balanced Accuracy Score (macro-average per-class recall), maximize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMP_SRC = COMP_DIR / "src"
if str(COMP_SRC) not in sys.path:
    sys.path.insert(0, str(COMP_SRC))

from clean import cast_categoricals, add_color_features, feature_columns  # noqa: E402

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


def score_fn(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def make_model(n_estimators=3000):
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=200,        # the under-regularization fix
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


print("Loading data …")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())
print(f"  train {train.shape}, test {test.shape}")

train = add_color_features(cast_categoricals(train))
test = add_color_features(cast_categoricals(test))
feat_cols = feature_columns(train)
print(f"  features ({len(feat_cols)}): {feat_cols}")

X, y = train[feat_cols].copy(), train[TARGET].copy()
X_test = test[feat_cols].copy()
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols) + "\n")

folds_list = folds_data["folds"]
n = len(train)
oof_proba = np.zeros((n, 3), dtype=np.float64)
oof_labels = np.empty(n, dtype=object)
per_fold_scores, best_iters = [], []

print("Running 5-fold OOF (early stopping) …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]

    model = make_model()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    best_iters.append(model.best_iteration_ or model.n_estimators)

    proba = model.predict_proba(X_va)
    class_order = list(model.classes_)
    for lbl in LABEL_ORDER:
        oof_proba[val_idx, LABEL2IDX[lbl]] = proba[:, class_order.index(lbl)]
    pred_labels = np.array([class_order[i] for i in np.argmax(proba, axis=1)])
    oof_labels[val_idx] = pred_labels
    s = score_fn(y_va.values, pred_labels)
    per_fold_scores.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}  (best_iter={best_iters[-1]})")

oof_metric = score_fn(y.values, oof_labels)
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"oof_metric={oof_metric:.6f}")
np.save(NODE_DIR / "oof.npy", oof_proba)
print("  saved oof.npy")

print(f"Retraining on full train (n_estimators={final_n}) …")
final_model = make_model(n_estimators=final_n)
final_model.fit(X, y)
test_proba = final_model.predict_proba(X_test)
# reorder test_proba columns to LABEL_ORDER and save for blending
class_order = list(final_model.classes_)
test_proba_ord = np.zeros((len(X_test), 3))
for lbl in LABEL_ORDER:
    test_proba_ord[:, LABEL2IDX[lbl]] = test_proba[:, class_order.index(lbl)]
np.save(NODE_DIR / "test_probs.npy", test_proba_ord)
print("  saved test_probs.npy")

test_pred_labels = np.array([LABEL_ORDER[i] for i in np.argmax(test_proba_ord, axis=1)])
submission = pd.DataFrame({IDC: test[IDC].values, TARGET: test_pred_labels})
submission = submission[list(sample_sub.columns)]
submission.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(submission)} rows)")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0005 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold_scores)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
best_iters: {best_iters}  -> final n_estimators={final_n}
change: tuned/regularized LightGBM (min_child_samples=200, lr=0.03, ES100, num_leaves=127, subsample/colsample=0.8, reg_lambda=1) vs node_0001 defaults. Saves test_probs.npy.
""")
print("Done.")
