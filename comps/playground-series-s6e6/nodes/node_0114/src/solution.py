"""node_0113 — TabM on fs_realmlp_fe + Negative Correlation Learning (NCL) penalty.

THE ONE ATOMIC CHANGE vs node_0033:
  Add an NCL penalty term to the training loss:
    total_loss = balanced-CE(class) - gamma * ncl_ambiguity

  The NCL ambiguity term (Liu & Yao 1999) explicitly pushes this learner's
  predictions away from the bank consensus (node_0070 OOF) while CE keeps it accurate.

  ncl_ambiguity = mean[ (f_i - p_bank_i) · (p_bank_mean_class - p_bank_i) ]
  where:
    f_i = softmax(logits)_i  (this model's prediction for sample i)
    p_bank_i = bank consensus OOF for sample i (train-fold rows only, fit_in_fold)
    p_bank_mean_class = mean(p_bank) over the batch (the ensemble mean reference)

  Subtracting this from CE minimizes loss when f_i disagrees with bank consensus.
  At inference: NO penalty — class head only.

  Gamma sweep at fold-0: {0.1, 0.3}; best gamma used for all 5 folds.

Gate criteria (fold-0, for EACH gamma):
  1. solo BA >= 0.965
  2. err-corr vs node_0070 bank OOF < 0.65
  If best gamma fails either: STOP, record.

FE pipeline (byte-identical to node_0033):
  - Stateless FE: redshift ratios, 7 color pairs, mag_mean, mag_range, log1p_redshift,
    integer-floor categorical views of every base numeric — computed once on full df (safe).
  - fit_in_fold KBinsDiscretizer (delta 100/500 quantile bins) on train fold only.
  - fit_in_fold TargetEncoder (on combo cats) on train fold only.
  - fit_in_fold standardization (mean/std from train fold numerical features only).
  - fit_in_fold PiecewiseLinearEmbeddings bins (compute_bins on train fold's Xnum+y).
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

import tabm
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
log(f"Device: {DEVICE}  tabm={tabm.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"

# TabM hyperparameters (byte-identical to node_0033)
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# NCL hyperparameters
GAMMA_SWEEP = [1.0, 5.0, 20.0]   # aggressive sweep — n113's {0.1,0.3} never engaged
GAMMA_BEST = None            # set after fold-0 sweep


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

# ─── Feature engineering globals (byte-identical to node_0033) ───────────────
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
    df = df.copy()
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr, df_val, df_te):
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

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

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

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


# ─── NCL penalty ─────────────────────────────────────────────────────────────

def ncl_ambiguity_penalty(logits: torch.Tensor, p_bank_batch: torch.Tensor) -> torch.Tensor:
    """
    Standard NCL ambiguity term (Liu & Yao 1999):
      ambiguity = mean_over_samples[ (f_i - p_bank_i) · (p_bank_mean - p_bank_i) ]
    where:
      f_i        = softmax(logits[i])                     shape (B, 3)
      p_bank_i   = bank OOF probabilities for sample i     shape (B, 3)
      p_bank_mean = mean(p_bank_i) over batch              shape (3,)

    Subtracting this from CE (total = CE - gamma * ambiguity) rewards disagreement
    with the bank consensus. Numerically stable: clamp p_bank to [1e-6, 1-1e-6].

    logits: (B, k, C)  — TabM ensemble logits
    p_bank_batch: (B, C)  — bank OOF probs for this batch (train-fold rows)
    Returns scalar penalty term.
    """
    # Average over ensemble heads k → (B, C)
    f_i = torch.softmax(logits.float(), dim=-1).mean(dim=1)   # (B, C)

    p_b = p_bank_batch.clamp(1e-6, 1 - 1e-6)                 # (B, C)
    p_mean = p_b.mean(dim=0, keepdim=True)                     # (1, C)

    # Element-wise product summed over classes, averaged over batch
    diff_fi   = f_i - p_b           # (B, C)  how this model deviates from bank
    diff_pmean = p_mean - p_b        # (B, C)  how bank deviates from its own mean

    # NCL ambiguity: E[(f_i - p_bank_i) · (p_mean - p_bank_i)]
    # We want to MAXIMIZE this (i.e., minimize its negative) so model's deviation
    # aligns with opposite sign of bank's deviation from mean.
    ambiguity = (diff_fi * diff_pmean).sum(dim=1).mean()       # scalar

    return ambiguity


# ─── TabM training ────────────────────────────────────────────────────────────

def build_tabm_model(n_num: int, cat_cards: list[int], bins: list) -> tabm.TabM:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    model = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=N_CLASSES,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    return model.to(DEVICE)


def predict_proba_batch(model: tabm.TabM, Xn: np.ndarray, Xc: np.ndarray | None,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            logits = model(xn, xc)          # (B, k, 3)
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)  # (B, 3)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm_ncl(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    p_bank_tr: np.ndarray,       # (N_tr, 3) bank OOF for train-fold rows (fit_in_fold)
    cat_cards: list[int],
    fold_seed: int,
    gamma: float,
) -> tuple[tabm.TabM, np.ndarray]:
    """
    Train TabM with PLR embeddings + NCL penalty.
    p_bank_tr: bank consensus OOF for the current train-fold rows (train-fold only, no leakage).
    gamma: NCL penalty coefficient.
    Returns (best_model, bins).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    # PLR bins — fit on TRAIN portion only (fit-in-fold)
    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

    # Class weights (balanced)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # Move train data to GPU
    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    # Bank OOF for train portion only
    p_bank_t = torch.as_tensor(p_bank_tr[ti], dtype=torch.float32, device=DEVICE)
    nt = len(ti)

    yv = y_tr[vi]
    Xn_vi = Xn_tr[vi]
    Xc_vi = Xc_tr[vi] if Xc_tr is not None else None

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_t[idx] if Xc_t is not None else None
            y_b = y_t[idx]
            p_bank_b = p_bank_t[idx]   # (B, 3) bank OOF for this batch

            opt.zero_grad()
            logits = model(xn_b, xc_b)     # (B, k, 3)
            b, k, c = logits.shape

            # CE loss (balanced)
            ce_loss = loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))

            # NCL penalty: subtract ambiguity to push away from bank consensus
            ncl_pen = ncl_ambiguity_penalty(logits, p_bank_b)

            # Total loss: minimize CE, maximize ambiguity (decorrelate from bank)
            total_loss = ce_loss - gamma * ncl_pen

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Early-stop on internal val (BA, no penalty at inference)
        val_probs = predict_proba_batch(model, Xn_vi, Xc_vi)
        ba = balanced_accuracy_score(yv, val_probs.argmax(1))
        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    TabM NCL early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}  gamma={gamma}")
    return model, bins


def compute_err_corr(oof_pred: np.ndarray, bank_oof: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute mean per-class error correlation between this model's OOF and the bank OOF.
    err_corr = mean over classes of Pearson corr between (1 - p_model[class]) and (1 - p_bank[class])
    """
    corrs = []
    for c in range(N_CLASSES):
        err_model = 1 - oof_pred[:, c]
        err_bank  = 1 - bank_oof[:, c]
        if np.std(err_model) < 1e-8 or np.std(err_bank) < 1e-8:
            corrs.append(1.0)
        else:
            corrs.append(float(np.corrcoef(err_model, err_bank)[0, 1]))
    return float(np.mean(corrs))


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_raw = json.loads((COMP_DIR / "folds.json").read_text())
folds_list = folds_raw["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

# ─── Load bank OOF (node_0070, 577347×3) ──────────────────────────────────────
log("Loading bank OOF (node_0070) ...")
p_bank_all = np.load(COMP_DIR / "nodes/node_0070/oof.npy").astype(np.float32)
assert p_bank_all.shape == (len(train_raw), N_CLASSES), \
    f"Bank OOF shape mismatch: {p_bank_all.shape}"
log(f"  p_bank shape={p_bank_all.shape}  min={p_bank_all.min():.4f}  max={p_bank_all.max():.4f}")

# ─── PRE-FLIGHT LEAKAGE CHECKS ────────────────────────────────────────────────
log("PRE-FLIGHT LEAKAGE CHECKS ...")

# Check 1: target/id not in features (stateless_fe adds only new columns)
sample_fe = stateless_fe(train_raw.drop(columns=[IDC, TARGET]).head(5))
assert TARGET not in sample_fe.columns, f"LEAK: target in features!"
assert IDC not in sample_fe.columns, f"LEAK: id in features!"
log("  [OK] target/id not in stateless FE columns")

# Check 2: Bank OOF — only train-fold rows used for NCL penalty (verified in fold loop)
log("  [OK] bank OOF will be sliced to tr_idx inside fold loop (no val/test leakage)")

# Check 3: single-feature↔target sweep on ≤50k sample
log("  Running single-feature corr sweep ...")
sample_size = min(50_000, len(train_raw))
rng_check = np.random.default_rng(0)
check_idx = rng_check.choice(len(train_raw), sample_size, replace=False)
s_check = stateless_fe(train_raw.drop(columns=[IDC, TARGET])).iloc[check_idx]
ys_check = train_raw[TARGET].map(LABEL_MAP).values[check_idx]
leak_found = False
for col in s_check.columns:
    try:
        x_c = pd.to_numeric(s_check[col], errors="coerce")
        if x_c.nunique() > 1:
            corr_val = abs(np.corrcoef(x_c.fillna(x_c.mean()), ys_check)[0, 1])
            if corr_val >= 0.999:
                log(f"  LEAK SMELL: {col} corr={corr_val:.5f}")
                leak_found = True
    except Exception:
        pass
if not leak_found:
    log("  [OK] no single-feature leak smell (corr < 0.999)")

# Check 4: frozen folds (loaded above)
log(f"  [OK] folds loaded from folds.json ({len(folds_list)} folds)")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── FOLD-0 GAMMA SWEEP ───────────────────────────────────────────────────────
log("=" * 60)
log("FOLD-0 GAMMA SWEEP (NCL gate)")
log("=" * 60)

fi0 = folds_list[0]
assert fi0["fold"] == 0, "Expected fold 0 first"
val_idx_0 = np.asarray(fi0["val_idx"], dtype=int)
tr_idx_0 = np.setdiff1d(np.arange(n_train), val_idx_0)
fold_seed_0 = SEED + 1 * 100

sweep_results = {}  # gamma -> (ba, err_corr)

for gamma in GAMMA_SWEEP:
    log(f"\n--- Gamma={gamma} ---")
    seed_everything(fold_seed_0)

    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx_0].reset_index(drop=True),
        X_stateless.iloc[val_idx_0].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx_0]
    y_val_fold = y_all[val_idx_0]

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed_0
    )

    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
    all_cols_sorted = sorted(X_tr_fold.columns)
    num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

    Xn_tr = X_tr_fold[num_for_tabm].values.astype(np.float32)
    Xn_va = X_val_fold[num_for_tabm].values.astype(np.float32)

    if TABM_CAT_COLS:
        Xc_tr = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_va = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_te = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
        cat_cards = (Xc_tr.max(axis=0) + 2).tolist()
        card_arr = np.array(cat_cards) - 1
        Xc_tr = np.clip(Xc_tr, 0, card_arr)
        Xc_va = np.clip(Xc_va, 0, card_arr)
        Xc_te = np.clip(Xc_te, 0, card_arr)
    else:
        Xc_tr = Xc_va = Xc_te = None
        cat_cards = []

    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr_s = (Xn_tr - mu) / sd
    Xn_va_s = (Xn_va - mu) / sd

    # Bank OOF — train-fold rows only (NCL penalty is fit_in_fold)
    p_bank_tr_fold = p_bank_all[tr_idx_0]   # (N_tr, 3)

    model, bins = train_tabm_ncl(
        Xn_tr_s, Xc_tr, y_tr_fold,
        p_bank_tr_fold,
        cat_cards, fold_seed_0, gamma
    )

    val_probs = predict_proba_batch(model, Xn_va_s, Xc_va)
    fold0_ba = balanced_accuracy_score(y_val_fold, val_probs.argmax(1))

    # Error correlation vs bank OOF on val fold
    bank_val = p_bank_all[val_idx_0]
    err_corr_val = compute_err_corr(val_probs, bank_val, y_val_fold)

    sweep_results[gamma] = (fold0_ba, err_corr_val)
    log(f"  gamma={gamma}: fold0_BA={fold0_ba:.6f}  err_corr={err_corr_val:.4f}")
    print(f"SWEEP gamma={gamma} fold0_BA={fold0_ba:.6f} err_corr={err_corr_val:.4f}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Report sweep frontier
log("\n=== NCL FRONTIER (fold-0) ===")
for g, (ba, ec) in sweep_results.items():
    log(f"  gamma={g}: BA={ba:.6f}  err_corr={ec:.4f}")
    print(f"FRONTIER gamma={g} BA={ba:.6f} err_corr={ec:.4f}", flush=True)

# Select: first gamma that achieves err-corr < 0.65 (plan: sweep in order {1,5,20})
# If none reaches err-corr < 0.65, record frontier and stop.
GAMMA_BEST = None
best_ba_f0, best_ec_f0 = None, None

log("\n=== GATE CHECKS ===")
for g in GAMMA_SWEEP:
    ba, ec = sweep_results[g]
    log(f"  gamma={g}: BA={ba:.6f}  err_corr={ec:.4f}")
    if ec < 0.65:
        GAMMA_BEST = g
        best_ba_f0, best_ec_f0 = ba, ec
        log(f"  -> gamma={g} FIRST achieves err_corr < 0.65 (ec={ec:.4f})")
        break

if GAMMA_BEST is None:
    # No gamma reached err-corr < 0.65
    log("NO gamma achieved err_corr < 0.65. Wall confirmed. STOPPING.")
    print("GATE_FAIL gate=err_corr no_gamma_reached_threshold", flush=True)
    sys.exit(1)

gate_ba_pass = best_ba_f0 >= 0.965
log(f"  BA gate (>= 0.965): {gate_ba_pass}  BA={best_ba_f0:.6f}")

if not gate_ba_pass:
    log(f"GATE FAILED: err_corr<0.65 gamma={GAMMA_BEST} but BA={best_ba_f0:.6f} < 0.965. Wall confirmed. STOPPING.")
    print(f"GATE_FAIL gate=BA gamma={GAMMA_BEST} BA={best_ba_f0:.6f} err_corr={best_ec_f0:.4f}", flush=True)
    sys.exit(1)

log(f"Both gates passed at gamma={GAMMA_BEST}: BA={best_ba_f0:.6f} err_corr={best_ec_f0:.4f}. Proceeding to full 5-fold run.")

# ─── FULL 5-FOLD RUN ──────────────────────────────────────────────────────────
log(f"\n=== FULL 5-FOLD RUN (gamma={GAMMA_BEST}) ===")

oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None
num_cols_final = None
TABM_CAT_COLS_FINAL = None

fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"], dtype=int)
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
    all_cols_sorted = sorted(X_tr_fold.columns)
    num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = num_for_tabm
        TABM_CAT_COLS_FINAL = TABM_CAT_COLS
        log(f"  n_features={X_tr_fold.shape[1]}  tabm_cat={len(TABM_CAT_COLS)}  tabm_num={len(num_for_tabm)}")

    Xn_tr = X_tr_fold[num_for_tabm].values.astype(np.float32)
    Xn_va = X_val_fold[num_for_tabm].values.astype(np.float32)
    Xn_te = X_te_fold[num_for_tabm].values.astype(np.float32)

    if TABM_CAT_COLS:
        Xc_tr = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_va = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_te = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
        cat_cards = (Xc_tr.max(axis=0) + 2).tolist()
        card_arr = np.array(cat_cards) - 1
        Xc_tr = np.clip(Xc_tr, 0, card_arr)
        Xc_va = np.clip(Xc_va, 0, card_arr)
        Xc_te = np.clip(Xc_te, 0, card_arr)
    else:
        Xc_tr = Xc_va = Xc_te = None
        cat_cards = []

    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Bank OOF — train-fold rows only (fit_in_fold, no leakage)
    p_bank_tr_fold = p_bank_all[tr_idx]   # (N_tr, 3)

    model, bins = train_tabm_ncl(
        Xn_tr, Xc_tr, y_tr_fold,
        p_bank_tr_fold,
        cat_cards, fold_seed, GAMMA_BEST
    )

    val_probs = predict_proba_batch(model, Xn_va, Xc_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# Full OOF err-corr
err_corr_full = compute_err_corr(oof_proba, p_bank_all, y_all)
log(f"Full OOF mean err-corr vs node_0070: {err_corr_full:.4f}")
print(f"err_corr_full={err_corr_full:.4f}", flush=True)

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
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
tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
