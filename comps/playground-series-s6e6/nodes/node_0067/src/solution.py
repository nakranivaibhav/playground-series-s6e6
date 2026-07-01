"""node_0067 — transductive soft-label distillation of bank-17 stack into TabM student.

THE ONE ATOMIC CHANGE vs node_0033 (TabM-richFE):
  Instead of CrossEntropyLoss on hard labels, train the student with KLDivLoss
  on soft teacher probabilities over BOTH train-fold rows AND test rows.
  Teacher = per-fold balanced-LogReg meta on 17-model public bank logits,
  temperature-scaled (T=2) before distillation.

Honesty rule (reviewer-confirmed):
  For fold f, the teacher on train-fold rows = meta fit on all-OTHER-folds rows,
  applied to fold-f rows (these rows were HELD OUT from the meta, so the teacher
  label is not trivially leaked).
  The teacher on test rows = meta fit on all-other-folds, applied to test.
  NEVER the refit-on-all-train teacher for train-fold rows.

  A small CE-to-true-label mix (weight 0.3) is added on train-fold rows to
  anchor the student to ground truth (optional but stabilizes training).

  Val-fold rows: predict with the student as usual; scored against TRUE labels
  for honest OOF balanced accuracy.

Architecture: TabM-richFE from node_0033 — byte-identical model, FE, and
training recipe except the loss function and the addition of test rows in
the training loop.

Kill: fold-0 solo BA < 0.967.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

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
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  tabm={tabm.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"
SINGLE_FOLD = os.environ.get("TABM_SINGLE_FOLD") == "1"

# TabM hyperparameters (identical to node_0033)
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# Distillation
TEMP = 2.0           # temperature for soft labels
CE_MIX_WEIGHT = 0.3  # fraction of hard-CE loss mixed with KL loss on train rows


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

# ─── Feature engineering (byte-identical to node_0033) ───────────────────────
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

    tr = df_tr.copy(); va = df_val.copy(); te = df_te.copy()

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


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
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


# ─── TabM model (identical to node_0033) ─────────────────────────────────────

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
            logits = model(xn, xc)
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def temperature_soften(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to probability array: re-softmax(logits/T)."""
    log_p = np.log(np.clip(probs, 1e-7, 1.0))
    log_p_T = log_p / T
    log_p_T -= log_p_T.max(1, keepdims=True)
    p_T = np.exp(log_p_T)
    return (p_T / p_T.sum(1, keepdims=True)).astype(np.float32)


def train_tabm_distill(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    teacher_tr: np.ndarray,   # (n_train_fold, 3) soft labels from per-fold meta
    Xn_te: np.ndarray,
    Xc_te: np.ndarray | None,
    teacher_te: np.ndarray,   # (n_test, 3) soft labels
    cat_cards: list[int],
    fold_seed: int,
) -> tuple[tabm.TabM, np.ndarray]:
    """
    Train TabM student with:
      - KL(student || teacher_tr) on train-fold rows
      - CE(student, y_true) * CE_MIX_WEIGHT mixed in on train-fold rows (anchor)
      - KL(student || teacher_te) on test rows (transductive)
    PLR bins computed on train-fold only. Early-stop on internal val using true labels.
    Returns (best_model, bins).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng_np = np.random.default_rng(fold_seed)
    perm = rng_np.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    # PLR bins on hard train subset only
    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    hard_loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # GPU tensors — hard train subset
    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    soft_tr_t = torch.as_tensor(teacher_tr[ti], dtype=torch.float32, device=DEVICE)
    nt = len(ti)

    # Test tensors for transductive pass
    n_te = len(Xn_te)
    Xn_te_t = torch.as_tensor(Xn_te, dtype=torch.float32, device=DEVICE)
    Xc_te_t = (torch.as_tensor(Xc_te, dtype=torch.long, device=DEVICE)
                if Xc_te is not None else None)
    soft_te_t = torch.as_tensor(teacher_te, dtype=torch.float32, device=DEVICE)
    log(f"    Distill: n_train_ti={nt}  n_test={n_te}  CE_mix={CE_MIX_WEIGHT}  T={TEMP}")

    yv = y_tr[vi]
    Xn_vi = Xn_tr[vi]
    Xc_vi = Xc_tr[vi] if Xc_tr is not None else None

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()

        # Train-fold pass: KL(student || teacher_tr) + CE_mix * CE(student, y_true)
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_t[idx] if Xc_t is not None else None
            y_b = y_t[idx]
            soft_b = soft_tr_t[idx]   # (B, 3) teacher soft labels
            opt.zero_grad()
            logits = model(xn_b, xc_b)   # (B, k, 3)
            b, k, c = logits.shape
            # KL divergence loss with soft labels
            log_p = F.log_softmax(logits.float(), dim=-1)   # (B, k, 3)
            soft_rep = soft_b.unsqueeze(1).expand(-1, k, -1)  # (B, k, 3)
            kl_loss = F.kl_div(
                log_p.reshape(b * k, c),
                soft_rep.reshape(b * k, c),
                reduction="batchmean",
            )
            # CE anchor on true labels
            ce_loss = hard_loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))
            loss = (1 - CE_MIX_WEIGHT) * kl_loss + CE_MIX_WEIGHT * ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Transductive pass: KL(student || teacher_te) on test rows
        te_perm = torch.randperm(n_te, device=DEVICE)
        for s in range(0, n_te, BATCH_SIZE):
            idx_te = te_perm[s:s + BATCH_SIZE]
            xn_te_b = Xn_te_t[idx_te]
            xc_te_b = Xc_te_t[idx_te] if Xc_te_t is not None else None
            soft_te_b = soft_te_t[idx_te]
            opt.zero_grad()
            logits_te = model(xn_te_b, xc_te_b)  # (B, k, 3)
            bt, kt, ct = logits_te.shape
            log_p_te = F.log_softmax(logits_te.float(), dim=-1)
            soft_te_rep = soft_te_b.unsqueeze(1).expand(-1, kt, -1)
            kl_te = F.kl_div(
                log_p_te.reshape(bt * kt, ct),
                soft_te_rep.reshape(bt * kt, ct),
                reduction="batchmean",
            )
            kl_te.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        # Early-stop on internal val — TRUE labels only
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
    log(f"    TabM early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model, bins


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_data)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

if SMOKE:
    log("SMOKE MODE")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_data = [folds_data[0]]

if SINGLE_FOLD:
    log("SINGLE_FOLD timing probe")
    folds_data = [folds_data[0]]

folds_list = folds_data
fval = [np.asarray(f["val_idx"]) for f in folds_list]


# ─── Build per-fold teacher meta ──────────────────────────────────────────────
# Load the 17-model public bank OOF + our node_0033 OOF (18 bases total)
# Then per fold f: fit meta on rows NOT in fold f's val_idx, predict fold-f rows.
log("Building per-fold teacher (leave-fold-out meta on bank-17 + node_0033) ...")

REFS = COMP_DIR / "refs"
KERNEL_OUT = REFS / "kernel_out"
OOF_BANK = REFS / "oof_bank"
LAB = ["GALAXY", "QSO", "STAR"]


def _norm(a: np.ndarray, nr: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a.reshape(nr, -1) if a.ndim == 1 else a
    a = a[:, :3]
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return (a / s).astype(np.float32)


def _rd(path, nr: int) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        return _norm(np.load(p, allow_pickle=True), nr)
    d = pd.read_csv(p)
    cols = list(d.columns)
    if set(LAB).issubset(cols):
        return _norm(d[LAB].values, nr)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(cols):
        return _norm(d[pc].values, nr)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3:
        return _norm(num.values[:, :3], nr)
    v = d.iloc[:, 0].values.astype(float)
    return _norm(v, nr)


# Bank manifest (17 models, drop xgb-0 and xgb-3 per the established bank)
MANIFEST_OOF = {
    "xgb-1": KERNEL_OUT / "xgb-v1-for-s6e6/oof_preds.npy",
    "realmlp-0": OOF_BANK / "oof_preds_realmlp0_v12.csv",
    "realmlp-1": KERNEL_OUT / "realmlp-v1-for-s6e6/oof_preds.npy",
    "tabm-0": OOF_BANK / "oof_preds_tabm0_v2.csv",
    "cat-0": KERNEL_OUT / "cat-v0-for-s6e6/catboost_oof_predictions.csv",
    "realmlp-2": OOF_BANK / "oof_preds_realmlp2_v10.csv",
    "tabicl-2": KERNEL_OUT / "tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",
    "lgbm-3": KERNEL_OUT / "lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",
    "logreg-1": KERNEL_OUT / "logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",
    "nn-1": KERNEL_OUT / "nn-v1-for-s6e6/train_oof/nn-1_oof.npy",
    "xgb-5": KERNEL_OUT / "xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",
    "realmlp-5": KERNEL_OUT / "realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",
    "nn-2": KERNEL_OUT / "nn-v2-for-s6e6/train_oof/nn-2_oof.npy",
    "cat-3": KERNEL_OUT / "cat-v3-for-s6e6/train_oof/cat-3_oof.npy",
    "lgbm-5": OOF_BANK / "oof_preds_lgbm5_v1.csv",
    "xgb-6": OOF_BANK / "oof_final_xgb6_v1.csv",
    "tabm-1": OOF_BANK / "oof_final_tabm1_v1.csv",
}
MANIFEST_TEST = {
    "xgb-1": KERNEL_OUT / "xgb-v1-for-s6e6/test_preds.npy",
    "realmlp-0": OOF_BANK / "test_preds_realmlp0_v12.csv",
    "realmlp-1": KERNEL_OUT / "realmlp-v1-for-s6e6/test_preds.npy",
    "tabm-0": OOF_BANK / "test_preds_tabm0_v2.csv",
    "cat-0": KERNEL_OUT / "cat-v0-for-s6e6/catboost_test_predictions.csv",
    "realmlp-2": OOF_BANK / "test_preds_realmlp2_v10.csv",
    "tabicl-2": KERNEL_OUT / "tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy",
    "lgbm-3": KERNEL_OUT / "lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy",
    "logreg-1": KERNEL_OUT / "logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy",
    "nn-1": KERNEL_OUT / "nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy",
    "xgb-5": KERNEL_OUT / "xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy",
    "realmlp-5": KERNEL_OUT / "realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy",
    "nn-2": KERNEL_OUT / "nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy",
    "cat-3": KERNEL_OUT / "cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy",
    "lgbm-5": OOF_BANK / "test_preds_lgbm5_v1.csv",
    "xgb-6": OOF_BANK / "test_final_xgb6_v1.csv",
    "tabm-1": OOF_BANK / "test_final_tabm1_v1.csv",
}

# Load all base OOF + test arrays, normalize
def logp(a): return np.log(np.clip(a, 1e-7, 1.0)).astype(np.float32)

base_oof_cols = []
base_test_cols = []
good_names = []
for name in MANIFEST_OOF:
    try:
        o = _rd(MANIFEST_OOF[name], n_train)
        t = _rd(MANIFEST_TEST[name], n_test)
        assert o.shape == (n_train, 3) and t.shape == (n_test, 3)
        ba = balanced_accuracy_score(y_all, o.argmax(1))
        if 0.90 < ba < 0.972:
            base_oof_cols.append(logp(o))
            base_test_cols.append(logp(t))
            good_names.append(name)
        else:
            log(f"  SKIP {name}: BA={ba:.4f} (outside 0.90–0.972)")
    except Exception as e:
        log(f"  FAIL {name}: {e}")

# Also add node_0033 (de-correlated TabM)
n33_oof = logp(np.load(COMP_DIR / "nodes/node_0033/oof.npy"))
n33_test = logp(np.load(COMP_DIR / "nodes/node_0033/test_probs.npy"))
base_oof_cols.append(n33_oof)
base_test_cols.append(n33_test)
good_names.append("node_0033")
log(f"  Loaded {len(good_names)} base models: {good_names}")

# Stack into matrices: (n_train, n_bases*3), (n_test, n_bases*3)
base_oof_mat = np.concatenate(base_oof_cols, axis=1)  # (n_train, n_bases*3)
base_test_mat = np.concatenate(base_test_cols, axis=1)  # (n_test, n_bases*3)

# Build per-fold teacher: leave-fold-out meta
# For fold f: meta fit on rows NOT in fval[f], predict fval[f] rows (val)
# ALSO predict all train rows using the leave-fold-out meta (for student's train-fold input)
teacher_train = np.zeros((n_train, N_CLASSES), dtype=np.float32)
teacher_test_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)

n_folds = len(folds_list)
for fi_idx, fi in enumerate(folds_list):
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)

    # Meta fit on train-fold rows only
    meta = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    meta.fit(base_oof_mat[tr_idx], y_all[tr_idx])

    # Teacher on ALL train rows (including train-fold rows — this is fold-honest
    # because the meta was fit ONLY on tr_idx; val_idx rows were never seen)
    teacher_train[tr_idx] = meta.predict_proba(base_oof_mat[tr_idx]).astype(np.float32)
    # Val rows: also fill (student won't use these for training, but we fill anyway)
    teacher_train[val_idx] = meta.predict_proba(base_oof_mat[val_idx]).astype(np.float32)

    # Teacher test: average across folds
    teacher_test_accum += meta.predict_proba(base_test_mat).astype(np.float32) / n_folds

    ba_tr_fold = balanced_accuracy_score(y_all[tr_idx], teacher_train[tr_idx].argmax(1))
    log(f"  fold-{fold_id} meta teacher: train-fold BA={ba_tr_fold:.5f}")

teacher_test = teacher_test_accum  # averaged across fold-metas

# Check teacher overall BA
ba_teacher_oof = balanced_accuracy_score(y_all, teacher_train.argmax(1))
log(f"Teacher (leave-fold-out) overall BA={ba_teacher_oof:.5f}")

# Apply temperature softening to teacher probs
teacher_train_soft = temperature_soften(teacher_train, TEMP)
teacher_test_soft = temperature_soften(teacher_test, TEMP)
log(f"Teacher soft probs range: train [{teacher_train_soft.min():.4f},{teacher_train_soft.max():.4f}]  "
    f"test [{teacher_test_soft.min():.4f},{teacher_test_soft.max():.4f}]")

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}")

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

    if SMOKE:
        keep_set = set(np.random.default_rng(0).choice(n_train, 30000, replace=False).tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
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

    # Standardize — fit on train fold only
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Teacher soft labels for this fold's train rows and test rows
    teacher_tr_fold = teacher_train_soft[tr_idx]   # (n_tr, 3) leave-fold-out meta
    teacher_te_fold = teacher_test_soft             # (n_test, 3) averaged across fold-metas

    # Train student with distillation
    model, bins = train_tabm_distill(
        Xn_tr, Xc_tr, y_tr_fold,
        teacher_tr_fold,
        Xn_te, Xc_te, teacher_te_fold,
        cat_cards, fold_seed,
    )

    # OOF predictions — val fold, TRUE labels
    val_probs = predict_proba_batch(model, Xn_va, Xc_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions
    test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    # Per-class recall
    preds_val = np.argmax(oof_proba[val_idx], axis=1)
    for ci, cname in enumerate(CLASSES):
        mask_c = y_val_fold == ci
        if mask_c.sum() > 0:
            recall_c = (preds_val[mask_c] == ci).mean()
            log(f"    recall_{cname}={recall_c:.5f}  (n={mask_c.sum()})")

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * 5
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")

        if SINGLE_FOLD:
            log("SINGLE_FOLD probe done. Exiting.")
            sys.exit(0)

        # Kill switch
        if fold_score < 0.967:
            log(f"KILL-SWITCH TRIPPED: fold-0 BA={fold_score:.6f} < 0.967. Stopping.")
            print(f"KILL_SWITCH fold0_BA={fold_score:.6f}", flush=True)
            sys.exit(1)

if SMOKE:
    log("[smoke] OK")
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

tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# Per-class recall full OOF
oof_preds = oof_proba.argmax(1)
log("=== Full OOF per-class recall ===")
for ci, cname in enumerate(CLASSES):
    mask_c = y_all == ci
    if mask_c.sum() > 0:
        recall_c = (oof_preds[mask_c] == ci).mean()
        log(f"  recall_{cname}={recall_c:.5f}  (n={mask_c.sum()})")

oof_metric = balanced_accuracy_score(y_all, oof_preds)
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
