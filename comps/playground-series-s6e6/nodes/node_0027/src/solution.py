"""node_0027 — improve (nn): TabPFN-v3 multiclass checkpoint, 50k context, K=8 subsamples.

Built on: node_0025 (TabPFN-2.5 large-samples, the fast fixed subsample-ensemble).
The fs_research feature load, frozen folds.json loop, fold-honest OOF/test interface,
stratified_subsample helper, and the entire data-engineering pipeline are kept
byte-identical. Only the following changes:

Change: ONE atomic change — swap the checkpoint from the TabPFN-2.5 v2.5_large-samples
checkpoint to the TabPFN-v3 multiclass checkpoint:
  HF repo: Prior-Labs/tabpfn_3
  file:    tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
This v3 variant is specifically trained for multiclass classification (our task is
3-class: GALAXY, QSO, STAR). Configuration stays identical:
  1. K_SUBSAMPLES = 8 (bagging saturates fast; coverage is 8x50k = 400k).
  2. SUBSAMPLE_SIZE = 50_000 (5x richer context vs node_0022's 10k).
  3. model_path = the v3_20260417_multiclass .ckpt file in ~/.cache/tabpfn/.
  4. CRITICAL BATCHING FIX: predict the ENTIRE val/test query block in ONE call
     per subsample (not a small-batch loop that re-encodes context repeatedly).
     TabPFN's predict_proba re-runs the full context every call — batching the
     queries re-encodes the 50k context on each batch call, which is ~1000x slower.
     Encode the context ONCE per subsample by predicting all queries in one shot.
  5. On OOM: halve the query chunk from the full block size (never floor at 512).
  6. ignore_pretraining_limits=True to unlock the 50k context window.

Contexts drawn ONLY from each fold's TRAIN indices (fit_in_fold, label-free);
val rows are queries, never in context; test queries use full-train context,
never test rows. Subsample RNG seed fixed per fold for reproducibility.

Outputs: oof.npy (577347x3), test_probs.npy (247435x3), submission.csv (argmax->label).
Label order: [GALAXY, QSO, STAR].
Metric: Balanced Accuracy Score (macro per-class recall, maximize).
"""
from __future__ import annotations

import json
import os
import sys
import time
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
K_SUBSAMPLES = 8            # bagging saturates fast; 8 x 50k = 400k coverage
SUBSAMPLE_SIZE = 50_000     # 5x richer context than node_0022's 10k

# TabPFN-v3 multiclass checkpoint (Prior-Labs/tabpfn_3, file: tabpfn-v3-classifier-v3_20260417_multiclass.ckpt)
TABPFN_CKPT = Path.home() / ".cache" / "tabpfn" / "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"

SMOKE = os.environ.get("TABPFN_SMOKE") == "1"
# For timing one fold: set TABPFN_ONE_FOLD=1
ONE_FOLD = os.environ.get("TABPFN_ONE_FOLD") == "1"


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


print("Loading + engineering ...", flush=True)
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
    train[col] = train[col].cat.codes.astype(np.float32)
    test[col] = test[col].cat.codes.astype(np.float32)

# All features as float (categoricals already converted to int codes above)
X_all = train[FEAT_COLS].to_numpy(np.float32)
X_te = test[FEAT_COLS].to_numpy(np.float32)
y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)
print(f"  rows={n}  features={len(FEAT_COLS)}", flush=True)

# Import TabPFN after confirming data shape
import torch
from tabpfn import TabPFNClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | TabPFN-v3 multiclass checkpoint: {TABPFN_CKPT}", flush=True)
assert TABPFN_CKPT.exists(), (
    f"TabPFN-v3 multiclass checkpoint not found: {TABPFN_CKPT}\n"
    "Download it from HuggingFace: Prior-Labs/tabpfn_3 "
    "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt\n"
    "  from huggingface_hub import hf_hub_download\n"
    "  hf_hub_download(repo_id='Prior-Labs/tabpfn_3', "
    "filename='tabpfn-v3-classifier-v3_20260417_multiclass.ckpt', "
    "local_dir='~/.cache/tabpfn/', token=HF_TOKEN)"
)


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


def predict_full_query(clf, query_X, n_classes=3):
    """Predict ALL query rows, falling back to large chunks on OOM.

    CRITICAL: predict the entire block in one call so the context is encoded
    ONCE (not once per chunk). Fall back to large chunks only on OOM — never
    small batches that would trigger repeated context re-encoding.
    The OOM fallback halves the chunk from the full block, keeping chunks large.
    """
    n_query = len(query_X)
    proba = np.zeros((n_query, n_classes), dtype=np.float64)

    # Try the full block first; chunk only on OOM
    chunk_size = n_query
    start = 0
    while start < n_query:
        end = min(start + chunk_size, n_query)
        try:
            p = clf.predict_proba(query_X[start:end])
            # Map clf.classes_ columns to canonical order [GALAXY=0, QSO=1, STAR=2]
            for j, c in enumerate(clf.classes_):
                proba[start:end, int(c)] = p[:, j]
            start = end
            # After a successful chunk, try larger chunks for the remainder
            chunk_size = n_query - start  # try the remainder as one chunk
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                new_chunk = max(chunk_size // 2, 10_000)
                if new_chunk >= chunk_size:
                    raise  # can't reduce further
                chunk_size = new_chunk
                print(
                    f"    OOM: reducing chunk_size to {chunk_size} (start={start})",
                    flush=True,
                )
            else:
                raise

    return proba


def subsample_predict(pool_idx, pool_y, query_X, fold_seed):
    """Predict query_X by averaging K subsample-conditioned TabPFN predictions.

    Subsamples come ONLY from pool_idx (train fold indices).
    query_X rows are NEVER in the context.
    Each subsample uses 50k class-balanced rows.
    FULL query block predicted in ONE call per subsample (encode context once).
    """
    pool_X = X_all[pool_idx]
    rng = np.random.default_rng(fold_seed)

    n_query = len(query_X)
    acc = np.zeros((n_query, 3), dtype=np.float64)

    for k in range(K_SUBSAMPLES):
        t0 = time.perf_counter()
        sub_idx_local = stratified_subsample(
            np.arange(len(pool_idx)), pool_y, SUBSAMPLE_SIZE, rng
        )
        ctx_X = pool_X[sub_idx_local]
        ctx_y = pool_y[sub_idx_local]

        clf = TabPFNClassifier(
            device=DEVICE,
            n_estimators=1,
            model_path=str(TABPFN_CKPT),
            ignore_pretraining_limits=True,
            show_progress_bar=False,
        )
        clf.fit(ctx_X, ctx_y)

        # Predict the ENTIRE query block in ONE call — encodes context once
        proba_k = predict_full_query(clf, query_X, n_classes=3)
        acc += proba_k

        t1 = time.perf_counter()
        print(
            f"    subsample {k+1}/{K_SUBSAMPLES} done  ({t1-t0:.1f}s)",
            flush=True,
        )

    return acc / K_SUBSAMPLES


if SMOKE:
    folds_list = [folds_list[0]]
    print("[smoke] running 1 fold only", flush=True)
elif ONE_FOLD:
    folds_list = [folds_list[0]]
    print("[one_fold] timing single fold", flush=True)

oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running OOF (TabPFN-v3 multiclass, 50k context, K=8) ...", flush=True)
for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    pool_y = y[tr_idx]

    print(
        f"Fold {fold_id}: tr={len(tr_idx)} val={len(val_idx)}",
        flush=True,
    )
    fold_seed = SEED * 100 + fold_id
    t_fold_start = time.perf_counter()

    proba = subsample_predict(tr_idx, pool_y, X_all[val_idx], fold_seed)
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    t_fold_end = time.perf_counter()
    print(
        f"  fold {fold_id}: balanced_accuracy = {s:.6f}  "
        f"({t_fold_end - t_fold_start:.1f}s)",
        flush=True,
    )

mean_cv = float(np.mean(per_fold))
sem_cv = (
    float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
    if len(per_fold) > 1
    else 0.0
)
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")

if SMOKE or ONE_FOLD:
    if ONE_FOLD:
        print("[one_fold] timing complete — exiting before full artifacts", flush=True)
    else:
        print("[smoke] OK — pipeline ran. Exiting before full artifacts.")
    sys.exit(0)

np.save(NODE_DIR / "oof.npy", oof_proba)

# ---- Test predictions: subsample from full train ----
print("Predicting test set (subsamples from full train) ...", flush=True)
t_test_start = time.perf_counter()
# Use a deterministic seed for test (fold 99)
test_proba = subsample_predict(np.arange(n), y, X_te, SEED * 100 + 99)
np.save(NODE_DIR / "test_probs.npy", test_proba)
t_test_end = time.perf_counter()
print(f"  test done in {t_test_end - t_test_start:.1f}s", flush=True)

labels = np.array([LABEL_ORDER[i] for i in test_proba.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(
    f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy",
    flush=True,
)

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
print(f"Final OOF balanced_accuracy = {oof_metric:.6f}")
print(f"OOF shape: {oof_proba.shape}, test_probs shape: {test_proba.shape}")
print("Done.")
