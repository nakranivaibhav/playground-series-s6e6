"""node_0022 — draft (nn): TabPFN-3 base, subsample-ensemble.

Built on: root — new NN base family. Data-loading/fold scaffolding matches
node_0009 (fs_research features, frozen folds.json), but the model is entirely
different: TabPFN (in-context Bayesian predictor) instead of TabM.

Change: Use TabPFN (`tabpfn` library) with a SUBSAMPLE-ENSEMBLE per fold to
handle the 577k-row scale that far exceeds TabPFN's native context window.
Per fold:
  - For OOF (val) predictions: draw K=8 class-stratified random subsamples of
    ≤10k rows drawn ONLY from that fold's train indices (fit_in_fold, never
    from val or test). Fit/condition TabPFN on each subsample; predict the val
    rows; AVERAGE the softmax probabilities across K subsamples.
  - For TEST predictions: same scheme using the FULL train as subsample pool
    (test rows never enter any context). Average K subsamples' softmax.
  - Subsample RNG seed is fixed per fold for reproducibility (seed = SEED*100 + fold_id).
  - Val rows predicted in batches to stay under GPU memory (≤4k rows per call).

Outputs: oof.npy (577347×3), test_probs.npy (247435×3), submission.csv (argmax→label).
Label order: [GALAXY, QSO, STAR].

Metric: Balanced Accuracy Score (macro per-class recall, maximize).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

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

SEED = 42
K_SUBSAMPLES = 8
SUBSAMPLE_SIZE = 10_000
VAL_BATCH = 4_000   # predict val in chunks to limit GPU memory

SMOKE = os.environ.get("TABPFN_SMOKE") == "1"


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


print("Loading + engineering ...")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

FEAT_COLS = feature_columns(train)

CATF = ["spectral_type", "galaxy_population"]
CONTF = [c for c in FEAT_COLS if c not in CATF]

# Encode categoricals as integer codes; categories fixed from full train (leak-safe: labels only)
for col in CATF:
    # categories already set by cast_categoricals (fixed vocab, not data-driven from counts)
    train[col] = train[col].cat.codes.astype(np.float32)
    test[col] = test[col].cat.codes.astype(np.float32)

# All features as float (categoricals already converted to int codes above)
X_all = train[FEAT_COLS].to_numpy(np.float32)
X_te = test[FEAT_COLS].to_numpy(np.float32)
y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)
print(f"  rows={n}  features={len(FEAT_COLS)}")

# Import TabPFN after confirming data shape
import torch
from tabpfn import TabPFNClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | tabpfn installed")


def stratified_subsample(pool_idx, pool_y, size, rng):
    """Draw a class-stratified subsample of `size` from pool_idx."""
    classes, counts = np.unique(pool_y, return_counts=True)
    fracs = counts / counts.sum()
    per_class = np.round(fracs * size).astype(int)
    # adjust rounding to hit target size exactly
    diff = size - per_class.sum()
    per_class[np.argmax(per_class)] += diff
    chosen = []
    for cls, cnt in zip(classes, per_class):
        cls_pool = pool_idx[pool_y == cls]
        cnt = min(cnt, len(cls_pool))
        chosen.append(rng.choice(cls_pool, cnt, replace=False))
    return np.concatenate(chosen)


def subsample_predict(pool_idx, pool_y, query_X, fold_seed):
    """Predict query_X by averaging K subsample-conditioned TabPFN predictions.

    Subsamples come ONLY from pool_idx (train fold indices).
    query_X rows are never in the context.
    """
    pool_X = X_all[pool_idx]  # features of the pool
    rng = np.random.default_rng(fold_seed)

    n_query = len(query_X)
    acc = np.zeros((n_query, 3), dtype=np.float64)

    for k in range(K_SUBSAMPLES):
        sub_idx_local = stratified_subsample(
            np.arange(len(pool_idx)), pool_y, SUBSAMPLE_SIZE, rng
        )
        ctx_X = pool_X[sub_idx_local]
        ctx_y = pool_y[sub_idx_local]

        clf = TabPFNClassifier(device=DEVICE, n_estimators=1)
        clf.fit(ctx_X, ctx_y)

        # Predict in batches to avoid OOM
        proba_k = np.zeros((n_query, 3), dtype=np.float64)
        for start in range(0, n_query, VAL_BATCH):
            end = min(start + VAL_BATCH, n_query)
            p = clf.predict_proba(query_X[start:end])
            # clf.classes_ are integer labels (LABEL2IDX values); columns already in that order
            # TabPFN returns columns indexed by clf.classes_ (which are 0,1,2 = GALAXY,QSO,STAR)
            for j, c in enumerate(clf.classes_):
                proba_k[start:end, int(c)] = p[:, j]

        acc += proba_k
        print(f"    subsample {k+1}/{K_SUBSAMPLES} done")

    return acc / K_SUBSAMPLES


if SMOKE:
    folds_list = [folds_list[0]]
    print("[smoke] running 1 fold only")

oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running OOF (TabPFN subsample-ensemble) ...")
for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    pool_y = y[tr_idx]

    print(f"Fold {fold_id}: tr={len(tr_idx)} val={len(val_idx)}")
    fold_seed = SEED * 100 + fold_id

    proba = subsample_predict(tr_idx, pool_y, X_all[val_idx], fold_seed)
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    print(f"  fold {fold_id}: balanced_accuracy = {s:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")

if SMOKE:
    print("[smoke] OK — pipeline ran. Exiting before full artifacts.")
    sys.exit(0)

np.save(NODE_DIR / "oof.npy", oof_proba)

# ---- Test predictions: subsample from full train ----
print("Predicting test set (subsamples from full train) ...")
# Use a deterministic seed for test (fold 99)
test_proba = subsample_predict(np.arange(n), y, X_te, SEED * 100 + 99)
np.save(NODE_DIR / "test_probs.npy", test_proba)

labels = np.array([LABEL_ORDER[i] for i in test_proba.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
print(f"Final OOF balanced_accuracy = {oof_metric:.6f}")
print("Done.")
