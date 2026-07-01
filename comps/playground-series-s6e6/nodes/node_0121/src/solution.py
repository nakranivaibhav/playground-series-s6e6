"""node_0121 — SDR sharpened-manifold kNN base.

FALL BACK REASON: pySDR on PyPI is a software-defined RADIO library, NOT the
astronomy SDR paper's code. No installable pySDR/SHARC for the A&A 2024 paper
exists. Falling back to: mean-shift density sharpening + UMAP + kNN (sklearn).

LEAK CLASS = fit_in_fold:
  - StandardScaler, UMAP, KNeighborsClassifier ALL fit on TRAIN FOLD ONLY.
  - val/test rows transformed via umap.transform() (no re-fit).
  - Mean-shift sharpening fit on train fold, then applied to val/test.
  - Final refit on full train AFTER OOF loop is expected and correct.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


NODE_DIR = Path(__file__).resolve().parent.parent
COMP_DIR = NODE_DIR.parent.parent

TARGET = "class"
IDC = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
SEED = 42

# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")

with open(COMP_DIR / "folds.json") as f:
    folds_data = json.load(f)
folds = folds_data["folds"]
n_train_expected = folds_data["n_rows"]

# Precheck: target and id absent from features (leak check 1 & 2)
assert TARGET not in ["u","g","r","i","z","redshift","u_g","g_r","r_i","i_z","u_r","g_i","r_z"]
assert IDC not in ["u","g","r","i","z","redshift","u_g","g_r","r_i","i_z","u_r","g_i","r_z"]
log("Leak checks 1&2 passed: target/id not in features")


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Stateless row-wise feature engineering (base photometric + colors)."""
    u = df["u"].values
    g = df["g"].values
    r = df["r"].values
    i = df["i"].values
    z = df["z"].values
    red = df["redshift"].values

    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    u_r = u - r
    g_i = g - i
    r_z = r - z

    X = np.column_stack([u, g, r, i, z, red, u_g, g_r, r_i, i_z, u_r, g_i, r_z])
    return X.astype(np.float32)


# Leak check 3: single-feature corr sweep on ≤50k sample
log("Leak check 3: single-feature↔target sweep...")
sample = train_raw.sample(min(50_000, len(train_raw)), random_state=0)
ys = np.array([LABEL_MAP[c] for c in sample[TARGET]])
Xs = build_features(sample)
feat_names = ["u","g","r","i","z","redshift","u_g","g_r","r_i","i_z","u_r","g_i","r_z"]
for k, name in enumerate(feat_names):
    col = Xs[:, k]
    if np.std(col) > 0:
        corr = abs(np.corrcoef(col, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {name} corr={corr:.4f} with target")
log("Leak check 3 passed: no near-perfect single-feature correlation")

# ─── SDR parameters ─────────────────────────────────────────────────────────
# Mean-shift sharpening: iteratively shift each point toward the local mean
# of its k nearest neighbors (density-gradient ascent in feature space).
MS_ITER = 2      # sharpening iterations (5 too slow: 116s/iter on 460k rows)
MS_K = 15        # neighbors for density estimate
UMAP_N_COMPONENTS = 8
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
KNN_K = 15


def mean_shift_sharpen(X: np.ndarray, k: int = MS_K, n_iter: int = MS_ITER) -> np.ndarray:
    """Shift each point toward the mean of its k nearest neighbors (train-fold only)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    X_sharp = X.copy()
    for _ in range(n_iter):
        _, indices = nn.kneighbors(X_sharp)
        # exclude self (first neighbor)
        neighbors = X[indices[:, 1:]]  # shape (n, k, d)
        X_sharp = neighbors.mean(axis=1)
    return X_sharp


def sharpen_new_points(X_train_orig: np.ndarray, X_new: np.ndarray,
                       k: int = MS_K, n_iter: int = MS_ITER) -> np.ndarray:
    """Apply mean-shift to new points using the TRAIN-FOLD density reference."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(X_train_orig)  # train-fold is the density reference
    X_sharp = X_new.copy()
    for _ in range(n_iter):
        _, indices = nn.kneighbors(X_sharp)
        # average over train-fold neighbors
        neighbors = X_train_orig[indices]  # shape (n, k, d)
        X_sharp = neighbors.mean(axis=1)
    return X_sharp


# ─── Full data arrays ────────────────────────────────────────────────────────
y_all = np.array([LABEL_MAP[c] for c in train_raw[TARGET]])
X_all = build_features(train_raw)
X_test = build_features(test_raw)
n_train = len(train_raw)
n_test = len(test_raw)
n_folds = len(folds)
log(f"n_train={n_train}  n_test={n_test}  n_folds={n_folds}")

# Leak check 5: folds from frozen folds.json
all_val_idx = []
for fold in folds:
    all_val_idx.extend(fold["val_idx"])
assert sorted(all_val_idx) == list(range(n_train)), "Fold indices don't cover train exactly once"
log("Leak check 5 passed: folds loaded from frozen folds.json, cover train exactly once")

# OOF storage
oof_probs = np.zeros((n_train, 3), dtype=np.float32)
fold_scores = []

KILL_FOLD0_BA = 0.94

for fold_idx, fold in enumerate(folds):
    va_idx = np.array(fold["val_idx"])
    mask = np.ones(n_train, dtype=bool)
    mask[va_idx] = False
    tr_idx = np.where(mask)[0]

    X_tr = X_all[tr_idx]
    y_tr = y_all[tr_idx]
    X_va = X_all[va_idx]
    y_va = y_all[va_idx]

    log(f"Fold {fold_idx}: train={len(tr_idx)} val={len(va_idx)}")

    # Step 1: Scale (fit on train fold only)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    # Step 2: Mean-shift sharpening (fit density reference on train fold only)
    log(f"  Fold {fold_idx}: mean-shift sharpening (train)...")
    X_tr_sharp = mean_shift_sharpen(X_tr_sc, k=MS_K, n_iter=MS_ITER)
    log(f"  Fold {fold_idx}: mean-shift sharpening (val)...")
    X_va_sharp = sharpen_new_points(X_tr_sc, X_va_sc, k=MS_K, n_iter=MS_ITER)

    # Step 3: UMAP (fit on sharpened train fold only)
    log(f"  Fold {fold_idx}: UMAP fit...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=SEED,
        n_jobs=-1,
    )
    E_tr = reducer.fit_transform(X_tr_sharp)
    log(f"  Fold {fold_idx}: UMAP transform val...")
    E_va = reducer.transform(X_va_sharp)

    # Step 4: kNN in sharpened embedding (fit on train fold only)
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance", n_jobs=-1)
    knn.fit(E_tr, y_tr)
    va_probs = knn.predict_proba(E_va)  # (n_val, 3)

    oof_probs[va_idx] = va_probs
    ba = balanced_accuracy_score(y_va, va_probs.argmax(axis=1))
    fold_scores.append(ba)
    log(f"Fold {fold_idx}: BA={ba:.6f}")

    if fold_idx == 0 and ba < KILL_FOLD0_BA:
        log(f"KILL: fold-0 BA={ba:.4f} < {KILL_FOLD0_BA} — stopping early")
        # Save partial oof for diagnostics
        np.save(NODE_DIR / "oof.npy", oof_probs)
        print(f"cv=KILLED_fold0_BA={ba:.4f}")
        sys.exit(0)

# Report
cv = float(np.mean(fold_scores))
sem = float(np.std(fold_scores, ddof=1) / np.sqrt(n_folds))
log(f"\nAll fold BAs: {[f'{s:.6f}' for s in fold_scores]}")
log(f"cv={cv:.6f}  sem={sem:.6f}")
print(f"cv={cv:.6f}")
for i, s in enumerate(fold_scores):
    print(f"fold{i}={s:.6f}")

# ─── Error correlation vs node_0070 ─────────────────────────────────────────
n070_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
if n070_oof_path.exists():
    n070_oof = np.load(n070_oof_path)
    n070_pred = n070_oof.argmax(axis=1)
    our_pred = oof_probs.argmax(axis=1)
    n070_err = (n070_pred != y_all).astype(float)
    our_err = (our_pred != y_all).astype(float)
    if our_err.std() > 0 and n070_err.std() > 0:
        err_corr = float(np.corrcoef(n070_err, our_err)[0, 1])
    else:
        err_corr = float("nan")
    log(f"Error correlation vs node_0070: {err_corr:.4f}")
    print(f"err_corr_vs_n070={err_corr:.4f}")
else:
    log("node_0070/oof.npy not found — skipping err-corr")

# ─── Refit on full train (AFTER OOF loop — expected and correct) ─────────────
log("Refitting on full train...")
scaler_full = StandardScaler()
X_all_sc = scaler_full.fit_transform(X_all)
X_test_sc = scaler_full.transform(X_test)

log("Full train mean-shift sharpening...")
X_all_sharp = mean_shift_sharpen(X_all_sc, k=MS_K, n_iter=MS_ITER)
log("Test mean-shift sharpening (using full train density)...")
X_test_sharp = sharpen_new_points(X_all_sc, X_test_sc, k=MS_K, n_iter=MS_ITER)

log("Full train UMAP fit...")
reducer_full = umap.UMAP(
    n_components=UMAP_N_COMPONENTS,
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    random_state=SEED,
    n_jobs=-1,
)
E_all = reducer_full.fit_transform(X_all_sharp)
log("Test UMAP transform...")
E_test = reducer_full.transform(X_test_sharp)

knn_full = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance", n_jobs=-1)
knn_full.fit(E_all, y_all)
test_probs = knn_full.predict_proba(E_test).astype(np.float32)

# ─── Save artifacts ──────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_probs)
np.save(NODE_DIR / "test_probs.npy", test_probs)
log("Saved oof.npy and test_probs.npy")

# ─── Submission ──────────────────────────────────────────────────────────────
test_pred_labels = [CLASSES[i] for i in test_probs.argmax(axis=1)]
sub = pd.DataFrame({"id": test_raw[IDC], "class": test_pred_labels})
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log("Saved submission.csv")

# ─── Post-train leak checks ───────────────────────────────────────────────────
# Check 7: OOF covers every train row exactly once, no NaN
assert oof_probs.shape == (n_train, 3), f"OOF shape wrong: {oof_probs.shape}"
assert not np.any(np.isnan(oof_probs)), "OOF has NaN"
log("Post-train checks: OOF complete and no NaN")

# Check 8: distribution sane
assert np.allclose(oof_probs.sum(axis=1), 1.0, atol=1e-4), "OOF probs don't sum to 1"
assert oof_probs.min() >= 0 and oof_probs.max() <= 1, "OOF probs out of [0,1]"
log("Post-train checks: distribution sane")

log(f"\nDONE — cv={cv:.6f} sem={sem:.6f}")
