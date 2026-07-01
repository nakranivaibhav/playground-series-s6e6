"""node_0106 — STAR-gate hierarchical specialist base.

THE ATOMIC CHANGE:
  Two-stage hierarchical framing (vs monolithic softmax in prior nodes):
  Stage-1: STAR-vs-nonSTAR TabM head (binary; redshift≈0 makes STAR near-trivial).
  Stage-2: GALAXY-vs-QSO TabM specialist trained ONLY on non-STAR rows.
  Combine: P(STAR)=p1_star; P(GALAXY)=(1-p1_star)*p2_galaxy; P(QSO)=(1-p1_star)*p2_qso.

CRITICAL LEAK SELF-CHECK #1: Stage-2's training rows are defined by per-fold
stage-1 OUT-OF-FOLD predictions (inner 10% split of train fold produces held-out
stage-1 predictions), NOT in-fold predictions. Assert per fold.

CHEAP-KILL:
  fold-0 combined BA < 0.965 → STOP.
  fold-0 err-corr vs node_0070 bank >= 0.65 → STOP.

FE pipeline: byte-identical to node_0033/TabM-on-fs_realmlp_fe.
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
# STAR idx=2, GALAXY=0, QSO=1
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}
STAR_IDX = LABEL_MAP["STAR"]   # 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  tabm={tabm.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"

# TabM hyperparameters
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096


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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
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


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names: list, fold_seed: int, n_classes: int):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    target_type = "multiclass" if n_classes > 2 else "binary"
    try:
        encoder = TargetEncoder(
            target_type=target_type, cv=5, smooth="auto", shuffle=True, random_state=fold_seed
        )
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)

    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])

    tr_enc_arr = np.asarray(tr_enc, dtype="float32")
    val_enc_arr = np.asarray(val_enc, dtype="float32")
    tst_enc_arr = np.asarray(tst_enc, dtype="float32")
    # For binary TE, sklearn returns shape (N, n_features) with 1 col per feature.
    # For multiclass, returns (N, n_features * n_classes).
    n_out_cols = tr_enc_arr.shape[1]
    if n_classes > 2:
        te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(n_classes)]
    else:
        te_names = [f"_{col}TE_binary" for col in combo_names]
    assert len(te_names) == n_out_cols, f"TE col count mismatch: {len(te_names)} vs {n_out_cols}"
    X_tr[te_names] = tr_enc_arr
    X_val[te_names] = val_enc_arr
    X_te[te_names] = tst_enc_arr

    return X_tr, X_val, X_te, te_names


# ─── TabM building/training ───────────────────────────────────────────────────

def build_tabm_model(n_num: int, cat_cards: list[int], bins: list, d_out: int) -> tabm.TabM:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    model = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=d_out,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    return model.to(DEVICE)


def predict_proba_batch(model: tabm.TabM, Xn: np.ndarray, Xc: np.ndarray | None,
                        d_out: int, batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            logits = model(xn, xc)          # (B, k, d_out)
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)  # (B, d_out)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
    d_out: int = 3,
) -> tuple[tabm.TabM, list]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins, d_out)

    counts = np.bincount(y_tr, minlength=d_out).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (d_out * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
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
            opt.zero_grad()
            logits = model(xn_b, xc_b)
            b, k, c = logits.shape
            loss = loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        val_probs = predict_proba_batch(model, Xn_vi, Xc_vi, d_out)
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
    log(f"    TabM early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}  d_out={d_out}")
    return model, bins


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

# ─── Pre-flight leakage checks ────────────────────────────────────────────────
log("Pre-flight leakage checks ...")
X_all_cols = [c for c in train_raw.columns if c not in [IDC, TARGET]]
assert TARGET not in X_all_cols, "TARGET in features!"
assert IDC not in X_all_cols, "ID in features!"

# single-feature↔target sweep on ≤50k sample
_sample = train_raw.sample(min(50_000, len(train_raw)), random_state=0)
_ys = pd.factorize(_sample[TARGET])[0]
for _c in X_all_cols:
    _x = pd.to_numeric(_sample[_c], errors="coerce")
    if _x.nunique() > 1:
        _corr = abs(np.corrcoef(_x.fillna(_x.mean()), _ys)[0, 1])
        if _corr >= 0.999:
            raise SystemExit(f"leak smell: {_c} ~ target (|corr|={_corr:.4f})")
log("  leak checks 1-3: CLEAN")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# y_s1: binary STAR=1 vs nonSTAR=0
y_s1_all = (y_all == STAR_IDX).astype(int)
# y_s2: among nonSTAR rows: GALAXY=0, QSO=1
# GALAXY=0, QSO=1 in 3-class. We map: GALAXY→0, QSO→1 (binary)
GALAXY_IDX = LABEL_MAP["GALAXY"]  # 0
QSO_IDX = LABEL_MAP["QSO"]        # 1

if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Load bank OOF for err-corr check ────────────────────────────────────────
BANK_OOF_PATH = COMP_DIR / "nodes/node_0070/oof.npy"
bank_oof = np.load(BANK_OOF_PATH)  # (577347, 3)
log(f"  Loaded bank OOF from node_0070: {bank_oof.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
FOLD0_ONLY = True  # start with fold-0 only for cheap-kill gate

log("Starting OOF loop (fold-0 cheap-kill first) ...")
fold_t0 = time.perf_counter()

killed = None
fold0_errcorr = None

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

    # ── Categorical/numerical FE for this fold (fit_in_fold) ──────────────────
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # ── Sort columns consistently ─────────────────────────────────────────────
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
    all_cols_sorted = sorted(X_tr_fold.columns)
    num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

    # ── Extract arrays ────────────────────────────────────────────────────────
    Xn_tr_raw = X_tr_fold[num_for_tabm].values.astype(np.float32)
    Xn_va_raw = X_val_fold[num_for_tabm].values.astype(np.float32)
    Xn_te_raw = X_te_fold[num_for_tabm].values.astype(np.float32)

    if TABM_CAT_COLS:
        Xc_tr_raw = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_va_raw = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_te_raw = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
        cat_cards = (Xc_tr_raw.max(axis=0) + 2).tolist()
        card_arr = np.array(cat_cards) - 1
        Xc_tr_raw = np.clip(Xc_tr_raw, 0, card_arr)
        Xc_va_raw = np.clip(Xc_va_raw, 0, card_arr)
        Xc_te_raw = np.clip(Xc_te_raw, 0, card_arr)
    else:
        Xc_tr_raw = Xc_va_raw = Xc_te_raw = None
        cat_cards = []

    # ── Standardize numerical — fit on train fold only ────────────────────────
    mu = Xn_tr_raw.mean(0)
    sd = Xn_tr_raw.std(0) + 1e-8
    Xn_tr = (Xn_tr_raw - mu) / sd
    Xn_va = (Xn_va_raw - mu) / sd
    Xn_te = (Xn_te_raw - mu) / sd

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE-1: STAR-vs-nonSTAR (binary)
    # ═══════════════════════════════════════════════════════════════════════════
    log(f"  [Fold {fold_id}] Stage-1: STAR-vs-nonSTAR binary TabM ...")
    y_s1_tr = y_s1_all[tr_idx]
    y_s1_val = y_s1_all[val_idx]

    # Add binary target encoding for stage-1 (binary: STAR vs nonSTAR)
    # Use combo_names from the categorical FE
    X_tr_s1 = X_tr_fold.copy()
    X_va_s1 = X_val_fold.copy()
    X_te_s1 = X_te_fold.copy()

    X_tr_s1, X_va_s1, X_te_s1, te_names_s1 = add_target_encoding(
        X_tr_s1, y_s1_tr, X_va_s1, X_te_s1, combo_names, fold_seed, n_classes=2
    )

    X_tr_s1 = X_tr_s1.reindex(sorted(X_tr_s1.columns), axis=1)
    X_va_s1 = X_va_s1.reindex(sorted(X_va_s1.columns), axis=1)
    X_te_s1 = X_te_s1.reindex(sorted(X_te_s1.columns), axis=1)

    num_s1 = [c for c in sorted(X_tr_s1.columns) if c not in TABM_CAT_COLS]

    Xn_tr_s1 = X_tr_s1[num_s1].values.astype(np.float32)
    Xn_va_s1 = X_va_s1[num_s1].values.astype(np.float32)
    Xn_te_s1 = X_te_s1[num_s1].values.astype(np.float32)

    mu_s1 = Xn_tr_s1.mean(0)
    sd_s1 = Xn_tr_s1.std(0) + 1e-8
    Xn_tr_s1 = (Xn_tr_s1 - mu_s1) / sd_s1
    Xn_va_s1 = (Xn_va_s1 - mu_s1) / sd_s1
    Xn_te_s1 = (Xn_te_s1 - mu_s1) / sd_s1

    # train stage-1 on FULL train fold
    model_s1, _ = train_tabm(Xn_tr_s1, Xc_tr_raw, y_s1_tr, cat_cards, fold_seed, d_out=2)

    # OOF predictions of stage-1 on val fold
    s1_val_probs = predict_proba_batch(model_s1, Xn_va_s1, Xc_va_raw, d_out=2)
    # s1_val_probs shape: (n_val, 2); col 0=nonSTAR, col 1=STAR
    p1_star_val = s1_val_probs[:, 1]

    # For test: stage-1 predictions on test set
    s1_te_probs = predict_proba_batch(model_s1, Xn_te_s1, Xc_te_raw, d_out=2)
    p1_star_te = s1_te_probs[:, 1]

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE-1 → inner OOF for stage-2 row selection (LEAK SELF-CHECK #1)
    # Stage-2 train rows must be from held-out stage-1 predictions.
    # We do an inner split of the train fold: 80% train stage-1 inner,
    # 20% held-out to get OOF stage-1 predictions for stage-2 training.
    # This ensures stage-2 never sees a stage-1 prediction that was trained
    # on the same row.
    # ═══════════════════════════════════════════════════════════════════════════
    log(f"  [Fold {fold_id}] Stage-1 inner OOF for stage-2 row selection ...")
    n_tr = len(tr_idx)
    rng_inner = np.random.default_rng(fold_seed + 7)
    inner_perm = rng_inner.permutation(n_tr)
    n_inner_val = max(1, int(0.20 * n_tr))
    inner_vi = inner_perm[:n_inner_val]
    inner_ti = inner_perm[n_inner_val:]

    # Train stage-1 on inner_ti, predict inner_vi
    model_s1_inner, _ = train_tabm(
        Xn_tr_s1[inner_ti], Xc_tr_raw[inner_ti] if Xc_tr_raw is not None else None,
        y_s1_tr[inner_ti], cat_cards, fold_seed + 13, d_out=2
    )
    s1_inner_probs = predict_proba_batch(
        model_s1_inner, Xn_tr_s1[inner_vi],
        Xc_tr_raw[inner_vi] if Xc_tr_raw is not None else None,
        d_out=2
    )
    # p(STAR) for the inner-held-out rows
    p1_star_inner_heldout = s1_inner_probs[:, 1]

    # LEAK SELF-CHECK #1 ASSERT: inner_vi was NOT used to train model_s1_inner
    # (inner_ti and inner_vi are disjoint by construction)
    assert len(set(inner_ti.tolist()) & set(inner_vi.tolist())) == 0, \
        "LEAK: inner_ti and inner_vi overlap!"
    log(f"  [Fold {fold_id}] LEAK CHECK #1: PASS — inner_vi disjoint from inner_ti "
        f"({len(inner_vi)} heldout / {len(inner_ti)} train)")

    del model_s1_inner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage-2 training rows = inner_vi where inner OOF stage-1 says NOT STAR
    # threshold: p(STAR) < 0.5 → treat as nonSTAR for stage-2
    STAR_THRESH = 0.5
    nonstar_mask_inner = p1_star_inner_heldout < STAR_THRESH
    s2_local_indices = inner_vi[nonstar_mask_inner]  # local indices in tr fold

    log(f"  [Fold {fold_id}] Stage-2 rows (nonSTAR by inner OOF): "
        f"{nonstar_mask_inner.sum()}/{len(inner_vi)} ({nonstar_mask_inner.mean()*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE-2: GALAXY-vs-QSO specialist (binary, trained on nonSTAR rows only)
    # Uses inner-heldout OOF selection → no leak from stage-1 in-fold predictions
    # ═══════════════════════════════════════════════════════════════════════════
    log(f"  [Fold {fold_id}] Stage-2: GALAXY-vs-QSO binary TabM ...")

    # Filter stage-2 training rows: nonSTAR by inner OOF + actual nonSTAR in ground truth
    # (use GT to avoid training on mislabeled STAR; this is pure train-fold supervision)
    y_tr_fold_local = y_s1_all[tr_idx]  # 1=STAR, 0=nonSTAR
    gt_nonstar_mask = y_tr_fold_local[s2_local_indices] == 0  # keep only true nonSTAR

    s2_train_local = s2_local_indices[gt_nonstar_mask]
    log(f"  [Fold {fold_id}] After GT filter: stage-2 rows = {len(s2_train_local)}")

    if len(s2_train_local) < 100:
        log(f"  [Fold {fold_id}] WARNING: too few stage-2 training rows, skipping stage-2")
        # fallback: use 50/50 GALAXY/QSO for nonSTAR rows
        s2_val_probs_fallback = np.full((len(val_idx), 2), 0.5, dtype=np.float32)
        s2_val_probs = s2_val_probs_fallback
        s2_te_probs = np.full((n_test, 2), 0.5, dtype=np.float32)
    else:
        # y_s2: among nonSTAR: GALAXY→0, QSO→1
        y_s2_tr_sub = y_tr_fold[s2_train_local]  # original 3-class labels
        # remap: GALAXY=0→0, QSO=1→1 (STAR should not appear)
        assert (y_s2_tr_sub == STAR_IDX).sum() == 0, \
            f"STAR rows leaked into stage-2 training! count={( y_s2_tr_sub == STAR_IDX).sum()}"
        y_s2_binary = (y_s2_tr_sub == QSO_IDX).astype(int)  # 0=GALAXY, 1=QSO

        # Add target encoding for stage-2 (binary)
        X_tr_s2 = X_tr_fold.iloc[s2_train_local].reset_index(drop=True).copy()
        # Need val and test for TE fit_transform
        X_va_s2 = X_val_fold.copy()
        X_te_s2 = X_te_fold.copy()

        X_tr_s2, X_va_s2, X_te_s2, te_names_s2 = add_target_encoding(
            X_tr_s2, y_s2_binary, X_va_s2, X_te_s2, combo_names, fold_seed + 1, n_classes=2
        )

        X_tr_s2 = X_tr_s2.reindex(sorted(X_tr_s2.columns), axis=1)
        X_va_s2 = X_va_s2.reindex(sorted(X_va_s2.columns), axis=1)
        X_te_s2 = X_te_s2.reindex(sorted(X_te_s2.columns), axis=1)

        num_s2 = [c for c in sorted(X_tr_s2.columns) if c not in TABM_CAT_COLS]

        Xn_tr_s2 = X_tr_s2[num_s2].values.astype(np.float32)
        Xn_va_s2 = X_va_s2[num_s2].values.astype(np.float32)
        Xn_te_s2 = X_te_s2[num_s2].values.astype(np.float32)

        # cat for s2 (subset of tr rows)
        if Xc_tr_raw is not None:
            Xc_tr_s2 = Xc_tr_raw[s2_train_local]
        else:
            Xc_tr_s2 = None

        mu_s2 = Xn_tr_s2.mean(0)
        sd_s2 = Xn_tr_s2.std(0) + 1e-8
        Xn_tr_s2 = (Xn_tr_s2 - mu_s2) / sd_s2
        Xn_va_s2 = (Xn_va_s2 - mu_s2) / sd_s2
        Xn_te_s2 = (Xn_te_s2 - mu_s2) / sd_s2

        model_s2, _ = train_tabm(Xn_tr_s2, Xc_tr_s2, y_s2_binary, cat_cards, fold_seed + 1, d_out=2)

        s2_val_probs = predict_proba_batch(model_s2, Xn_va_s2, Xc_va_raw, d_out=2)
        s2_te_probs = predict_proba_batch(model_s2, Xn_te_s2, Xc_te_raw, d_out=2)

        del model_s2, X_tr_s2, X_va_s2, X_te_s2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINE: P(STAR)=p1_star; P(GALAXY)=(1-p1_star)*p2_galaxy; P(QSO)=(1-p1_star)*p2_qso
    # ═══════════════════════════════════════════════════════════════════════════
    # s2_val_probs[:, 0] = P(GALAXY | nonSTAR)
    # s2_val_probs[:, 1] = P(QSO | nonSTAR)
    combined_val = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
    combined_val[:, STAR_IDX] = p1_star_val
    combined_val[:, GALAXY_IDX] = (1.0 - p1_star_val) * s2_val_probs[:, 0]
    combined_val[:, QSO_IDX] = (1.0 - p1_star_val) * s2_val_probs[:, 1]
    # Renormalize (should sum to 1 already but just in case)
    row_sums = combined_val.sum(axis=1, keepdims=True)
    combined_val = combined_val / np.clip(row_sums, 1e-8, None)

    oof_proba[val_idx] = combined_val

    combined_te = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    combined_te[:, STAR_IDX] = p1_star_te
    combined_te[:, GALAXY_IDX] = (1.0 - p1_star_te) * s2_te_probs[:, 0]
    combined_te[:, QSO_IDX] = (1.0 - p1_star_te) * s2_te_probs[:, 1]
    row_sums_te = combined_te.sum(axis=1, keepdims=True)
    combined_te = combined_te / np.clip(row_sums_te, 1e-8, None)
    test_proba_accum += combined_te / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(combined_val, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    # ─── CHEAP-KILL GATES (fold-0 only) ─────────────────────────────────────
    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

        # Gate 1: fold-0 BA < 0.965
        if fold_score < 0.965:
            log(f"CHEAP-KILL: fold-0 BA={fold_score:.6f} < 0.965 — STOPPING")
            killed = f"fold0_BA={fold_score:.6f}_below_0.965"
            break

        # Gate 2: err-corr vs node_0070
        bank_val_oof = bank_oof[val_idx]
        bank_err = (bank_val_oof.argmax(1) != y_val_fold).astype(float)
        this_err = (combined_val.argmax(1) != y_val_fold).astype(float)
        err_corr = float(np.corrcoef(bank_err, this_err)[0, 1])
        fold0_errcorr = err_corr
        log(f"  fold-0 err-corr vs bank node_0070: {err_corr:.4f}")
        print(f"fold0_errcorr={err_corr:.4f}", flush=True)

        if err_corr >= 0.65:
            log(f"CHEAP-KILL: fold-0 err-corr={err_corr:.4f} >= 0.65 — STOPPING (no decorrelation)")
            killed = f"fold0_errcorr={err_corr:.4f}_above_0.65"
            break

        log(f"  fold-0 PASS: BA={fold_score:.6f} ≥ 0.965 AND err-corr={err_corr:.4f} < 0.65 — running all folds")

    del X_tr_fold, X_val_fold, X_te_fold, model_s1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores))) if len(per_fold_scores) > 1 else 0.0
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

if killed:
    log(f"KILLED: {killed} — stopping before saving full artifacts")
    # Still save partial OOF (fold-0 only) for diagnostic but mark incomplete
    np.save(NODE_DIR / "oof.npy", oof_proba)
    log("Saved partial oof.npy (fold-0 only)")
    sys.exit(0)

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

# ─── OOF leakage checks (post-train) ─────────────────────────────────────────
log("Post-train OOF checks ...")
assert not np.any(np.isnan(oof_proba)), "NaN in OOF!"
assert oof_proba.shape == (n_train, N_CLASSES), f"OOF shape mismatch: {oof_proba.shape}"
# All rows filled
covered = np.abs(oof_proba.sum(1) - 1.0) < 0.01
assert covered.all(), f"{(~covered).sum()} rows have prob sum != 1"
log("  OOF: shape OK, no NaN, probs sum to 1")

# Final err-corr (full OOF)
full_bank_err = (bank_oof.argmax(1) != y_all).astype(float)
full_this_err = (oof_proba.argmax(1) != y_all).astype(float)
full_errcorr = float(np.corrcoef(full_bank_err, full_this_err)[0, 1])
log(f"Full OOF err-corr vs node_0070: {full_errcorr:.4f}")
print(f"full_errcorr={full_errcorr:.4f}", flush=True)

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
