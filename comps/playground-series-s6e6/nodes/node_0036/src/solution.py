"""node_0036 — draft (nn): plain deep MLP on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0028:
    Replace the hand-rolled RealMLP (n_ens=8 internal ensemble + PBLD periodic
    numerical embeddings) with a PLAIN deep MLP:
    - Architecture: [512, 512, 256] hidden layers (3 layers)
    - BatchNorm1d after each linear (before activation)
    - Dropout (p=0.3) after each activation
    - GELU activation throughout
    - AdamW optimizer (lr=1e-3, weight_decay=1e-4)
    - Class-balanced cross-entropy (compute_class_weight("balanced"))
    - StandardScaler fit-in-fold (on train fold rows only)
    - Cosine annealing LR schedule with warm restarts
    - Early stopping on val balanced accuracy (patience=5 epochs)
    - Max 40 epochs, batch size 2048 (GPU)
    - NO PBLD periodic embeddings, NO internal n_ens ensemble
    - Categoricals: one-hot the 2 low-card base cats (spectral_type 4 levels,
      galaxy_population 2 levels); route all other categorical codes (integer)
      as plain numerics through the StandardScaler

Built on: root (new draft, NOT an improve on node_0028). Template FE pipeline
copied from node_0028/src exactly — only the model class and its fit/predict
wrappers change. The fs_realmlp_fe feature engineering (stateless + fit_in_fold)
is byte-identical to node_0028.

Leakage discipline:
  - Stateless FE (color pairs, mag stats, redshift ratio, log1p_redshift) is
    computed once on the full X/X_test dataframes — NO target, NO cross-row stats,
    NO fitting. Safe (stateless, row-wise).
  - KBinsDiscretizer (delta bins), category factorize maps: fit on train-fold only,
    applied to val and test.
  - TargetEncoder: fit on train-fold only (using sklearn's internal CV=5 strategy),
    applied to val and test.
  - StandardScaler: fit on train-fold rows only, applied to val and test.
  - frozen folds.json used throughout; no refitting of folds.

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
while not (REPO_ROOT / "tools" / "leakage_scan.py").exists():
    REPO_ROOT = REPO_ROOT.parent

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


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)

# ─── MLP CONFIG ──────────────────────────────────────────────────────────────
MLP_CONFIG = {
    "hidden_dims": [512, 512, 256],
    "dropout": 0.30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 40,
    "train_bs": 2048,
    "eval_bs": 16384,
    "patience": 5,           # early stop patience (in epochs)
    "t_max": 10,             # CosineAnnealingWarmRestarts T_0
}

# ─── Categorical handling for plain MLP ─────────────────────────────────────
# Low-cardinality base cats → one-hot (spectral_type: 4 levels, galaxy_population: 2)
# All other categoricals (high-card floor-int codes, bin codes, combo codes) →
# treated as plain numerics (integer values) and scaled by StandardScaler.
LOW_CARD_BASE_CATS = ["spectral_type", "galaxy_population"]

# ─── Feature engineering globals ─────────────────────────────────────────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless feature engineering — safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    """
    df = df.copy()

    # Redshift ratios
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")

    # Color pairs
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")

    # Magnitude aggregates
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")

    # Log1p of shifted redshift
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")

    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.
    """
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    # Work on copies
    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    # Original categorical columns (spectral_type, galaxy_population)
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    # Integer-floor categorical views of every base numeric
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_tr in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    # Delta quantile bins (100 and 500) — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

    # Interaction cross-combos
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_names.append(combo_name)
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        local_map[combo_name] = uniques
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    new_cat_cols = sorted([c for c in tr.columns if str(tr[c].dtype) == "category"])
    return tr, va, te, new_cat_cols, combo_names, local_map


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names: list, fold_seed: int):
    """
    TargetEncoder fit on train fold only (fit_in_fold), transform val and test.
    Returns modified copies and the list of new TE column names.
    """
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    try:
        encoder = TargetEncoder(
            target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed
        )
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)

    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])

    te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(N_CLASSES)]
    X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
    X_val[te_names] = np.asarray(val_enc, dtype="float32")
    X_te[te_names] = np.asarray(tst_enc, dtype="float32")

    return X_tr, X_val, X_te, te_names


# ─── Plain deep MLP model ─────────────────────────────────────────────────────

class PlainMLP(nn.Module):
    """
    Plain deep MLP: Linear -> BatchNorm -> GELU -> Dropout, repeated.
    No periodic embeddings, no internal ensemble, no parameter tricks.
    Different inductive bias than RealMLP/TabM for stack diversity.
    """

    def __init__(self, in_features: int, hidden_dims: list, out_features: int,
                 dropout: float = 0.30):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainMLPClassifier:
    """
    Sklearn-style wrapper for PlainMLP with:
    - StandardScaler fit on train fold only
    - One-hot encoding for low-card categoricals
    - Integer codes as numerics for high-card categoricals
    - Class-balanced cross-entropy
    - CosineAnnealingWarmRestarts LR schedule
    - Early stopping on val balanced accuracy
    """

    def __init__(self, cfg: dict, device: torch.device, seed: int = 42):
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.scaler_ = None
        self.model_ = None
        self.low_card_cats_ = None
        self.low_card_dims_ = None
        self.high_card_cats_ = None
        self.feature_names_ = None
        self.best_val_probs_ = None
        self.best_score_ = -np.inf
        self.onehot_width_ = 0

    def _prepare_features(self, X: pd.DataFrame,
                          cat_cols: list,
                          low_card_cats: list,
                          low_card_dims: dict) -> np.ndarray:
        """
        Build the input matrix:
          - low-card cat cols → one-hot (using stored dims)
          - all other cols (nums + high-card cat codes) → pass as float
        Concatenate into one float32 array.
        """
        parts = []
        # One-hot for low-card base cats
        for col in low_card_cats:
            n_cats = low_card_dims[col]
            codes = X[col].values.astype(np.int64)
            codes = np.clip(codes, 0, n_cats - 1)
            onehot = np.zeros((len(X), n_cats), dtype=np.float32)
            onehot[np.arange(len(X)), codes] = 1.0
            parts.append(onehot)

        # All remaining cols as float
        other_cols = [c for c in X.columns if c not in low_card_cats]
        parts.append(X[other_cols].values.astype(np.float32))

        return np.concatenate(parts, axis=1)

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray,
            cat_cols: list) -> "PlainMLPClassifier":
        cfg = self.cfg
        dev = self.device

        # Identify low-card and high-card categoricals
        low_card_cats = [c for c in cat_cols if c in LOW_CARD_BASE_CATS]
        high_card_cats = [c for c in cat_cols if c not in LOW_CARD_BASE_CATS]

        # Compute one-hot dims from train fold only
        low_card_dims = {}
        for col in low_card_cats:
            low_card_dims[col] = int(X_tr[col].astype(int).max()) + 1

        self.low_card_cats_ = low_card_cats
        self.low_card_dims_ = low_card_dims
        self.high_card_cats_ = high_card_cats

        # Build feature matrices (before scaling)
        X_tr_mat = self._prepare_features(X_tr, cat_cols, low_card_cats, low_card_dims)
        X_val_mat = self._prepare_features(X_val, cat_cols, low_card_cats, low_card_dims)

        # Compute one-hot width to know which columns to scale
        onehot_width = sum(low_card_dims[c] for c in low_card_cats)
        self.onehot_width_ = onehot_width

        # StandardScaler — fit on train fold only, on non-one-hot portion
        # (one-hot is already 0/1, no need to scale it)
        self.scaler_ = StandardScaler()
        # Fit scaler on the non-onehot part (after onehot_width)
        self.scaler_.fit(X_tr_mat[:, onehot_width:])

        def scale(mat):
            scaled_tail = self.scaler_.transform(mat[:, onehot_width:])
            return np.concatenate([mat[:, :onehot_width], scaled_tail], axis=1)

        X_tr_scaled = scale(X_tr_mat)
        X_val_scaled = scale(X_val_mat)

        in_features = X_tr_scaled.shape[1]
        log(f"  in_features={in_features}  onehot_width={onehot_width}")

        # Class weights — balanced
        classes = np.array([0, 1, 2])
        weights_np = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        # Build model
        self.model_ = PlainMLP(
            in_features=in_features,
            hidden_dims=cfg["hidden_dims"],
            out_features=N_CLASSES,
            dropout=cfg["dropout"],
        ).to(dev)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg["t_max"], T_mult=1
        )

        # Move data to tensors on GPU
        Xtn = torch.as_tensor(X_tr_scaled, dtype=torch.float32, device=dev)
        ytn = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_scaled, dtype=torch.float32, device=dev)

        n_tr = len(y_tr)
        train_bs = cfg["train_bs"]
        eval_bs = cfg["eval_bs"]
        epochs = cfg["epochs"]
        patience = cfg["patience"]

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            # Shuffle indices
            perm = torch.randperm(n_tr, device=dev)
            Xtn_s = Xtn[perm]
            ytn_s = ytn[perm]

            self.model_.train()
            for start in range(0, n_tr, train_bs):
                xb = Xtn_s[start: start + train_bs]
                yb = ytn_s[start: start + train_bs]
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = F.cross_entropy(logits, yb, weight=class_weights)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            self.model_.eval()
            val_probs_list = []
            with torch.no_grad():
                for s in range(0, len(y_val), eval_bs):
                    logits_v = self.model_(Xvn[s: s + eval_bs])
                    val_probs_list.append(F.softmax(logits_v, dim=1).cpu().numpy())
            val_probs = np.concatenate(val_probs_list, axis=0)
            epoch_score = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))

            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            current_lr = scheduler.get_last_lr()[0]
            log(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                f"best={best_score:.5f}  lr={current_lr:.2e}"
                + ("  *" if improved else f"  (no_improve={no_improve})"))

            if no_improve >= patience:
                log(f"  Early stopping at epoch {epoch + 1} (best={best_epoch})")
                break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        log(f"  Best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        cat_cols_all = self.low_card_cats_ + self.high_card_cats_
        X_mat = self._prepare_features(X, cat_cols_all, self.low_card_cats_, self.low_card_dims_)
        scaled_tail = self.scaler_.transform(X_mat[:, self.onehot_width_:])
        X_scaled = np.concatenate([X_mat[:, :self.onehot_width_], scaled_tail], axis=1)

        Xn = torch.as_tensor(X_scaled, dtype=torch.float32, device=self.device)
        eval_bs = cfg["eval_bs"]
        self.model_.eval()
        probs_list = []
        with torch.no_grad():
            for s in range(0, len(X_scaled), eval_bs):
                logits = self.model_(Xn[s: s + eval_bs])
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs_list, axis=0)


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None
num_cols_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # Target encoding — fit_in_fold
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]
        log(f"  n_features={X_tr_fold.shape[1]}  n_cat={len(cat_cols_sorted)}  n_num={len(num_cols_final)}")

    # Train plain MLP — fit_in_fold (scaler fit inside PlainMLPClassifier.fit)
    model = PlainMLPClassifier(cfg=MLP_CONFIG, device=DEVICE, seed=fold_seed)
    model.fit(
        X_tr_fold, y_tr_fold,
        X_val_fold, y_val_fold,
        cat_cols=cat_cols_sorted,
    )

    # OOF probabilities (indexed into original row positions via val_idx)
    oof_proba[val_idx] = model.best_val_probs_.astype("float32")

    # Test predictions — average across folds
    test_proba_accum += model.predict_proba(X_te_fold).astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── VRAM usage (if GPU) ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    log(f"Peak VRAM allocated: {peak_mb:.0f} MB")

# ─── Save OOF ────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ─────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
# All feature columns the model receives (after FE, before scaler / one-hot)
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
