"""node_0026 — draft (nn): TabICL foundation-model base.

TabICL is purpose-built for large tabular data via column-then-row attention +
in-context learning, with CPU/disk offloading. We use the sklearn interface
(TabICLClassifier) which handles the ensemble of dataset views internally.

ONE ATOMIC CHANGE (vs node_0009): the model family is TabICL (foundation model
with in-context learning) instead of TabM. Feature construction (fs_research) is
byte-identical to node_0009/src via the shared clean.py.

Leakage discipline:
  - standardization: fit on the train fold ONLY (same as node_0009).
  - TabICL context: subsampled class-balanced from train-fold indices ONLY.
    Val rows are never in the context. fit() sees train-fold rows only.
  - test_probs: re-fit on FULL train, never test rows in context.

Context strategy:
  - We subsample up to CONTEXT_SIZE rows from each train fold, class-balanced,
    to stay within VRAM limits and keep per-fold time under ~30s.
  - CONTEXT_SIZE=100000 gives a 5-fold projection of ~3 min total (probed).
  - kv_cache="repr": context KV projections are cached after fit(); predict_proba
    only processes the query block -> context encoded ONCE per fold.
  - All val rows (up to 115k) predicted in a single predict_proba call.

Set TABICL_SMOKE=1 for a fast shape/sanity run (subsample, 1 fold).

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
Outputs: oof.npy (577347x3), test_probs.npy (247435x3), submission.csv, features.txt.
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
from tabicl import TabICLClassifier

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

# fs_research features (byte-identical to node_0009 -- one atomic change only)
CONT = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
        "u_g", "g_r", "r_i", "i_z", "u_z", "u_r", "u_i", "g_i", "r_z",
        "c_ug_gr", "c_gr_ri", "log1p_redshift", "gal_l", "gal_b"]   # 22, standardized
FLAGS = ["is_star_z", "is_highz", "qso_box", "uv_excess"]           # 4, numeric 0/1
NUMF = CONT + FLAGS                                                  # 26 -> numeric features
CATF = ["spectral_type", "galaxy_population"]                        # 2 categorical

# TabICL context subsampling: class-balanced from train-fold rows only.
# 100k context (~33.3k per class) keeps VRAM in range, fold ~28s (probed).
CONTEXT_SIZE = 100_000
N_ESTIMATORS = 8       # default; probed at ~27s per fold
SMOKE = os.environ.get("TABICL_SMOKE") == "1"
SEED = 42
N_CONT = len(CONT)


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


def standardize_fit(Xnum, rows):
    """Return (mu, sd) computed on the train-fold rows only (fit-inside-fold)."""
    mu = Xnum[rows, :N_CONT].mean(0)
    sd = Xnum[rows, :N_CONT].std(0) + 1e-8
    return mu, sd


def apply_std(Xnum, mu, sd):
    out = Xnum.copy()
    out[:, :N_CONT] = (out[:, :N_CONT] - mu) / sd
    return out


def class_balanced_subsample(local_indices, y_local, n, fold_rng):
    """Subsample `n` rows from `local_indices` (indices into y_local), class-balanced."""
    classes = np.unique(y_local[local_indices])
    n_per_class = max(1, n // len(classes))
    selected = []
    for c in classes:
        c_idx = local_indices[y_local[local_indices] == c]
        k = min(n_per_class, len(c_idx))
        selected.append(fold_rng.choice(c_idx, size=k, replace=False))
    sel = np.concatenate(selected)
    # If under n, top-up from remainder uniformly
    if len(sel) < n:
        remaining = np.setdiff1d(local_indices, sel)
        extra = min(n - len(sel), len(remaining))
        if extra > 0:
            sel = np.concatenate([sel, fold_rng.choice(remaining, size=extra, replace=False)])
    fold_rng.shuffle(sel)
    return sel


print("Loading + engineering data ...")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

# Write features.txt: all columns except id/target
(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)

# Build numeric feature matrix (standardized inside fold)
Xnum_all = train[NUMF].to_numpy(np.float32)
Xnum_te = test[NUMF].to_numpy(np.float32)

# One-hot encode categorical features (stateless, fixed category vocab from cast_categoricals)
# spectral_type: 4 categories, galaxy_population: 2 categories -> 6 OHE columns
cat_ohe_parts_tr = []
cat_ohe_parts_te = []
for c in CATF:
    n_cats = int(train[c].cat.categories.size)
    ohe_tr = np.eye(n_cats, dtype=np.float32)[train[c].cat.codes.to_numpy()]
    ohe_te = np.eye(n_cats, dtype=np.float32)[test[c].cat.codes.to_numpy()]
    cat_ohe_parts_tr.append(ohe_tr)
    cat_ohe_parts_te.append(ohe_te)

cat_ohe_all = np.concatenate(cat_ohe_parts_tr, axis=1)  # (n, 6)
cat_ohe_te = np.concatenate(cat_ohe_parts_te, axis=1)   # (m, 6)
n_features = Xnum_all.shape[1] + cat_ohe_all.shape[1]
print(f"  n_rows={n}, n_test={len(test)}, n_features={n_features} (NUMF={len(NUMF)}, cat_ohe={cat_ohe_all.shape[1]})")

rng = np.random.default_rng(SEED)

if SMOKE:
    folds_list = [folds_list[0]]
    print("[smoke] 1 fold only")

oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print(f"Running OOF (TabICL n_estimators={N_ESTIMATORS}, context_size={CONTEXT_SIZE}, kv_cache=repr) ...")

for fi in folds_list:
    fold_t0 = time.time()
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)

    if SMOKE:
        # Small subsample for smoke test
        keep_tr = rng.choice(tr_idx, size=min(5000, len(tr_idx)), replace=False)
        keep_va = rng.choice(val_idx, size=min(2000, len(val_idx)), replace=False)
        tr_idx = keep_tr
        val_idx = keep_va

    # Standardize: fit on train-fold ONLY (fit-inside-fold)
    mu, sd = standardize_fit(Xnum_all, tr_idx)
    Xn_tr_all = apply_std(Xnum_all[tr_idx], mu, sd)
    Xn_va = apply_std(Xnum_all[val_idx], mu, sd)

    # Assemble full feature matrices
    X_tr_full = np.concatenate([Xn_tr_all, cat_ohe_all[tr_idx]], axis=1)
    X_va = np.concatenate([Xn_va, cat_ohe_all[val_idx]], axis=1)
    y_tr = y[tr_idx]

    # Context subsampling: class-balanced from TRAIN FOLD ONLY
    # local_indices are positions into tr_idx (0..len(tr_idx)-1)
    ctx_size = min(CONTEXT_SIZE, len(tr_idx))
    local_all = np.arange(len(tr_idx))
    ctx_local = class_balanced_subsample(local_all, y_tr, ctx_size, rng)
    X_ctx = X_tr_full[ctx_local]
    y_ctx = y_tr[ctx_local]
    print(f"  fold {fi['fold']}: tr={len(tr_idx)}, val={len(val_idx)}, ctx={len(X_ctx)} (class dist: {np.bincount(y_ctx, minlength=3).tolist()})")

    # TabICL: fit on context (train-fold rows only), then predict all val rows at once
    # kv_cache="repr" caches context after fit; predict_proba only processes queries
    clf = TabICLClassifier(
        n_estimators=N_ESTIMATORS,
        device="cuda",
        verbose=False,
        kv_cache="repr",
        random_state=SEED,
    )
    clf.fit(X_ctx, y_ctx)

    # Predict all val rows in one call (context already cached in repr mode)
    raw_proba = clf.predict_proba(X_va)  # shape (n_val, n_classes_seen)
    # y is already int-encoded (0=GALAXY, 1=QSO, 2=STAR) so clf.classes_ is [0,1,2];
    # if for any reason the order differs, remap via clf.classes_ indices.
    proba_ordered = np.zeros((len(val_idx), 3), dtype=np.float64)
    for i, cls_idx in enumerate(clf.classes_):
        proba_ordered[:, int(cls_idx)] = raw_proba[:, i]

    oof_proba[val_idx] = proba_ordered
    s = balanced_accuracy_score(y[val_idx], proba_ordered.argmax(1))
    per_fold.append(s)
    elapsed = time.time() - fold_t0
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}  [{elapsed:.1f}s]")

    del clf
    import torch as _torch
    _torch.cuda.empty_cache()

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")

if SMOKE:
    print("[smoke] OK -- pipeline runs end-to-end. Exiting before full artifacts.")
    sys.exit(0)

np.save(NODE_DIR / "oof.npy", oof_proba)

# ---- full-train fit -> test probs + submission ----
print("Retraining on full train for the test set ...")
t_final = time.time()
mu_full, sd_full = standardize_fit(Xnum_all, np.arange(n))
Xn_full = apply_std(Xnum_all, mu_full, sd_full)
X_full = np.concatenate([Xn_full, cat_ohe_all], axis=1)

# Context: class-balanced subsample of ALL train rows (no test rows ever)
ctx_size_final = min(CONTEXT_SIZE, n)
local_all_final = np.arange(n)
ctx_final = class_balanced_subsample(local_all_final, y, ctx_size_final, rng)
X_ctx_final = X_full[ctx_final]
y_ctx_final = y[ctx_final]
print(f"  final fit ctx={len(X_ctx_final)} (class dist: {np.bincount(y_ctx_final, minlength=3).tolist()})")

import torch as _torch
_torch.cuda.empty_cache()

clf_final = TabICLClassifier(
    n_estimators=N_ESTIMATORS,
    device="cuda",
    verbose=False,
    kv_cache="repr",
    random_state=SEED,
)
clf_final.fit(X_ctx_final, y_ctx_final)

# Predict test in chunks of TEST_CHUNK rows to avoid OOM on the col-embedding
# output buffer (n_estimators * n_rows * n_features * embed_dim).
# With 247k test rows in one shot that buffer is ~17GB; 100k chunks stay safe.
TEST_CHUNK = 100_000
Xn_te_std = apply_std(Xnum_te, mu_full, sd_full)
X_te_feat = np.concatenate([Xn_te_std, cat_ohe_te], axis=1)
raw_tp_chunks = []
n_te = len(X_te_feat)
for start in range(0, n_te, TEST_CHUNK):
    end = min(start + TEST_CHUNK, n_te)
    chunk = X_te_feat[start:end]
    raw_tp_chunks.append(clf_final.predict_proba(chunk))
    print(f"  test chunk [{start}:{end}] done")
raw_tp = np.concatenate(raw_tp_chunks, axis=0)

# Remap to canonical label order (y is int-encoded, clf.classes_ is [0,1,2])
tp = np.zeros((len(test), 3), dtype=np.float64)
for i, cls_idx in enumerate(clf_final.classes_):
    tp[:, int(cls_idx)] = raw_tp[:, i]

np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in tp.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  final fit+predict: {time.time()-t_final:.1f}s")
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
(NODE_DIR / "metrics.md").write_text(
    f"""# node_0026 metrics
metric: Balanced Accuracy Score (maximize)
model: TabICL v2 (tabicl, n_estimators={N_ESTIMATORS}, context_size={CONTEXT_SIZE}, kv_cache=repr), CUDA
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} +/- {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
change: TabICL foundation model; class-balanced context subsample from train-fold only; kv_cache=repr for efficient query prediction.
""")
print("Done.")
