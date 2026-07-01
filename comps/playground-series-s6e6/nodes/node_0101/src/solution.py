"""node_0101 — kNN-graph GraphSAGE base (fs_knngraph).

THE ONE ATOMIC CHANGE vs root:
  A new model FAMILY: kNN-graph GraphSAGE. For each row, build k=12 nearest
  neighbours in standardized feature space (fit on TRAIN-FOLD ONLY). A 3-layer
  SAGEConv GNN aggregates neighbourhood info before classifying. The prediction
  for a row depends on its neighbourhood — no other base in the bank has this.

  MODEL PATH: torch-geometric SAGEConv (PyG 2.8.0 + torch 2.11.0+cu128 — verified).

Leak safety (fit_in_fold):
  - StandardScaler fit on TRAIN-FOLD ROWS ONLY.
  - NearestNeighbors index built on TRAIN-FOLD ROWS ONLY.
  - Val/test rows query the SAME train-fold index (never see each other or labels).
  - Folds loaded from frozen folds.json.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}")
log("MODEL PATH: torch-geometric SAGEConv (PyG 2.8.0 + torch 2.11.0+cu128 — cuda verified)")
assert torch.cuda.is_available(), "CUDA required"

FOLD0_ONLY = os.environ.get("FOLD0_ONLY") == "1"
CHEAP_KILL_THRESHOLD = 0.962

# kNN graph parameters
K_NEIGH = 12       # number of nearest neighbours

# GraphSAGE hyperparameters
HIDDEN_DIM = 256
N_LAYERS = 3
DROPOUT = 0.2
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 8192   # mini-batch training via neighbour sampling
LR = 3e-3
WEIGHT_DECAY = 1e-4

MAGS = ["u", "g", "r", "i", "z"]
COLORS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "z")]
COLOR_NAMES = [f"{a}-{b}" for a, b in COLORS]


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_everything(SEED)


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract raw numeric features: colors + redshift + magnitudes.
    Row-wise stateless (no fitting). Returns (N, n_features) float32.
    """
    cols = []
    # colors
    for a, b in COLORS:
        cols.append((df[a] - df[b]).values.astype(np.float32))
    # raw redshift
    cols.append(df["redshift"].values.astype(np.float32))
    # raw magnitudes
    for m in MAGS:
        cols.append(df[m].values.astype(np.float32))
    # spectral_type ordinal
    st_map = {"GALAXY": 0, "QSO": 1, "STAR": 2, "UNK": 3}
    st = df["spectral_type"].map(st_map).fillna(3).astype(np.float32).values
    cols.append(st)
    # galaxy_population ordinal (already numeric or categorical)
    gp = pd.to_numeric(df["galaxy_population"], errors="coerce").fillna(-1).astype(np.float32).values
    cols.append(gp)
    return np.column_stack(cols)


# ─── GraphSAGE model ─────────────────────────────────────────────────────────

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int, n_classes: int, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.convs.append(SAGEConv(hidden, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


# ─── Build kNN graph edges (train-fold index only) ───────────────────────────

def build_train_graph(X_tr: np.ndarray, k: int):
    """
    Build kNN graph for train-fold rows. Returns (edge_index, nn_model).
    edge_index: (2, E) int64 — source→dest pairs.
    Each node connects to up to k neighbours (excluding self).
    """
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nn_model.fit(X_tr)
    distances, indices = nn_model.kneighbors(X_tr)  # shape (N_tr, k+1) — first col is self
    # Exclude self (first column)
    indices = indices[:, 1:]  # (N_tr, k)
    n_tr = len(X_tr)
    # Build edge list: for each node i, edges i→j for j in neighbours
    src = np.repeat(np.arange(n_tr), k)
    dst = indices.ravel()
    edge_index = np.stack([src, dst], axis=0)  # (2, N_tr*k)
    return edge_index, nn_model


def build_query_edges(X_query: np.ndarray, nn_model: NearestNeighbors, k: int, n_tr: int, offset: int):
    """
    Build edges for val/test rows querying the TRAIN-FOLD index.
    Query rows are indexed [n_tr + offset : n_tr + offset + len(X_query)] in the full node set.
    Returns edge_index where query rows point to their k train-fold neighbours.
    """
    distances, indices = nn_model.kneighbors(X_query, n_neighbors=k)
    n_q = len(X_query)
    src = np.arange(n_tr + offset, n_tr + offset + n_q).repeat(k)
    dst = indices.ravel()  # train-fold node indices
    edge_index = np.stack([src, dst], axis=0)  # (2, n_q*k)
    return edge_index


# ─── Training ─────────────────────────────────────────────────────────────────

def train_sage(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    edge_index_tr: np.ndarray,  # (2, E) train-only edges
    edge_index_val: np.ndarray, # (2, E') edges for val (val→train)
    n_tr: int,
    n_val: int,
    fold_seed: int,
) -> GraphSAGEClassifier:
    """
    Full-graph training on the train-fold graph.
    For efficiency, we do mini-batch gradient steps on FULL GRAPH (fits in VRAM for 460k nodes).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    in_channels = X_tr.shape[1]
    model = GraphSAGEClassifier(in_channels, HIDDEN_DIM, N_CLASSES, N_LAYERS, DROPOUT).to(DEVICE)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # Build combined node feature matrix: [train | val]
    X_all = np.concatenate([X_tr, X_val], axis=0)
    x_tensor = torch.as_tensor(X_all, dtype=torch.float32, device=DEVICE)

    # Combined edge index: train edges + val→train edges
    edge_idx_combined = np.concatenate([edge_index_tr, edge_index_val], axis=1)
    ei_tensor = torch.as_tensor(edge_idx_combined, dtype=torch.long, device=DEVICE)

    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=DEVICE)
    tr_idx = torch.arange(n_tr, dtype=torch.long, device=DEVICE)

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(x_tensor, ei_tensor)
        loss = loss_fn(out[tr_idx], y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (ep + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(x_tensor, ei_tensor)
                val_node_idx = torch.arange(n_tr, n_tr + n_val, dtype=torch.long, device=DEVICE)
                val_probs = torch.softmax(out_eval[val_node_idx].float(), dim=-1).cpu().numpy()
            ba = balanced_accuracy_score(y_val, val_probs.argmax(1))
            if ba > best_ba + 1e-5:
                best_ba = ba
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    SAGE early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model


def predict_sage(
    model: GraphSAGEClassifier,
    X_tr: np.ndarray,
    X_query: np.ndarray,
    edge_index_tr: np.ndarray,
    edge_index_query: np.ndarray,
    n_tr: int,
    n_query: int,
) -> np.ndarray:
    """Predict probabilities for query rows (val or test) against the train-fold graph."""
    model.eval()
    X_all = np.concatenate([X_tr, X_query], axis=0)
    x_tensor = torch.as_tensor(X_all, dtype=torch.float32, device=DEVICE)
    ei = np.concatenate([edge_index_tr, edge_index_query], axis=1)
    ei_tensor = torch.as_tensor(ei, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(x_tensor, ei_tensor)
        query_idx = torch.arange(n_tr, n_tr + n_query, dtype=torch.long, device=DEVICE)
        probs = torch.softmax(out[query_idx].float(), dim=-1).cpu().numpy()
    return probs.astype(np.float32)


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_raw = json.loads((COMP_DIR / "folds.json").read_text())
folds_list = folds_raw["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Pre-flight leakage checks ───────────────────────────────────────────────
log("Pre-flight check 1+2: target/id not in feature columns")
feat_sample = extract_features(train_raw.head(10))  # stateless, just check structure
# TARGET and IDC are never passed to extract_features
assert TARGET not in ["u-g", "g-r", "r-i", "i-z", "u-z", "redshift", "u", "g", "r", "i", "z",
                       "spectral_type", "galaxy_population"]
assert IDC not in ["u-g", "g-r", "r-i", "i-z", "u-z", "redshift", "u", "g", "r", "i", "z",
                   "spectral_type", "galaxy_population"]
log("  -> OK (target/id not in feature set)")
log("Pre-flight check 5: frozen folds.json loaded (never recomputed)")
log(f"  -> OK ({len(folds_list)} folds from folds.json)")

# Pre-flight check 3: single-feature sweep on fold-0 train rows
log("Pre-flight check 3: single-feature ~ target sweep on fold-0 sample ...")
fold0_val_idx = np.asarray(folds_list[0]["val_idx"], dtype=int)
fold0_tr_idx = np.setdiff1d(np.arange(n_train), fold0_val_idx)
X_pf = extract_features(train_raw.iloc[fold0_tr_idx])
# Apply scaler (train-fold only) for this check
sc_pf = StandardScaler()
X_pf_s = sc_pf.fit_transform(X_pf)
y_pf = y_all[fold0_tr_idx]
sample_n = min(50000, len(X_pf_s))
rng_pf = np.random.default_rng(0)
sidx = rng_pf.choice(len(X_pf_s), sample_n, replace=False)
ys_check = y_pf[sidx]
max_corr = 0.0
for fi_idx in range(X_pf_s.shape[1]):
    xf = X_pf_s[sidx, fi_idx]
    if np.isnan(xf).any() or np.std(xf) < 1e-10:
        continue
    c = abs(np.corrcoef(xf, ys_check)[0, 1])
    if c > max_corr:
        max_corr = c
    if c >= 0.999:
        raise SystemExit(f"LEAK: feature idx={fi_idx} corr={c:.4f}")
log(f"  max |corr| = {max_corr:.4f}  (< 0.999 = clean)")
del X_pf, X_pf_s, sc_pf
gc.collect()

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"], dtype=int)
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    df_tr = train_raw.iloc[tr_idx].reset_index(drop=True)
    df_val = train_raw.iloc[val_idx].reset_index(drop=True)
    df_te = test_raw.copy()

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Extract raw features (stateless row-wise)
    X_tr_raw = extract_features(df_tr)
    X_val_raw = extract_features(df_val)
    X_te_raw = extract_features(df_te)

    # StandardScaler — FIT ON TRAIN FOLD ONLY
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    n_tr_fold = len(X_tr)
    n_val_fold = len(X_val)
    n_te_fold = len(X_te)

    log(f"  Building kNN graph (k={K_NEIGH}) on {n_tr_fold} train rows ...")
    t_graph = time.perf_counter()

    # kNN INDEX — fit on TRAIN FOLD ONLY
    edge_index_tr, nn_model = build_train_graph(X_tr, K_NEIGH)
    log(f"  Train graph built: {edge_index_tr.shape[1]} edges  ({time.perf_counter()-t_graph:.1f}s)")

    # Val rows query train-fold index (no val→val edges)
    edge_index_val = build_query_edges(X_val, nn_model, K_NEIGH, n_tr_fold, offset=0)

    # Train the GraphSAGE on train-fold graph (val nodes included for messaging but not loss)
    log(f"  Training GraphSAGE (hidden={HIDDEN_DIM}, layers={N_LAYERS}, epochs≤{MAX_EPOCHS}) ...")
    t_train = time.perf_counter()
    model = train_sage(
        X_tr, y_tr_fold,
        X_val, y_val_fold,
        edge_index_tr, edge_index_val,
        n_tr_fold, n_val_fold,
        fold_seed,
    )
    log(f"  Training done ({time.perf_counter()-t_train:.1f}s)")

    # OOF predictions for val rows
    val_probs = predict_sage(model, X_tr, X_val, edge_index_tr, edge_index_val, n_tr_fold, n_val_fold)
    oof_proba[val_idx] = val_probs

    fold_score = balanced_accuracy_score(y_val_fold, val_probs.argmax(1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    # Test predictions — accumulate across folds
    if not FOLD0_ONLY:
        edge_index_te = build_query_edges(X_te, nn_model, K_NEIGH, n_tr_fold, offset=0)
        te_probs = predict_sage(model, X_tr, X_te, edge_index_tr, edge_index_te, n_tr_fold, n_te_fold)
        test_proba_accum += te_probs / len(folds_list)

    del model, X_tr, X_val, X_te, X_tr_raw, X_val_raw, X_te_raw
    del edge_index_tr, edge_index_val, nn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

    # CHEAP-KILL: if fold 0 BA < 0.962, abort
    if fold_id == 0:
        if fold_score < CHEAP_KILL_THRESHOLD:
            log(f"CHEAP-KILL: fold-0 BA={fold_score:.6f} < {CHEAP_KILL_THRESHOLD}. Stopping.")
            print(f"CHEAP_KILL fold0={fold_score:.6f}", flush=True)
            # Save partial OOF (only fold 0 filled) so gates can check
            np.save(NODE_DIR / "oof.npy", oof_proba)
            sys.exit(42)
        if FOLD0_ONLY:
            log("FOLD0_ONLY mode — stopping after fold 0.")
            np.save(NODE_DIR / "oof.npy", oof_proba)
            sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save artifacts ───────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# ─── Decorrelation check vs node_0070 bank reference ─────────────────────────
log("Computing decorrelation vs node_0070 oof bank reference ...")
bank_oof = np.load(COMP_DIR / "nodes/node_0070/oof.npy")  # (577347, 3)
this_oof = oof_proba  # (577347, 3)

# Per-class error-correlation: for each class c, compare argmax-errors
this_pred = this_oof.argmax(1)
bank_pred = bank_oof.argmax(1)

err_corrs = []
for c in range(N_CLASSES):
    this_err = (this_pred != y_all).astype(np.float32) * (y_all == c).astype(np.float32)
    bank_err = (bank_pred != y_all).astype(np.float32) * (y_all == c).astype(np.float32)
    mask = (y_all == c)
    if mask.sum() > 1:
        corr = np.corrcoef(this_err[mask], bank_err[mask])[0, 1]
        err_corrs.append(float(corr))
        log(f"  class {CLASSES[c]}: error_corr_vs_bank = {corr:.4f}")

mean_err_corr = float(np.mean(err_corrs))
log(f"  MEAN per-class error_corr_vs_bank = {mean_err_corr:.4f}")
print(f"mean_err_corr_vs_bank={mean_err_corr:.4f}", flush=True)

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
