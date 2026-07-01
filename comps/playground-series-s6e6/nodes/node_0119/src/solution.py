"""node_0119 — synthetic generative pretrain then finetune TabM

THE ONE COUPLED CHANGE (wildcard) vs node_0033 (TabM on fs_realmlp_fe, cv 0.968053):
  Add a generative PRETRAIN phase per fold:
    1. Fit a per-class GaussianCopulaSynthesizer (sdv) on the TRAIN FOLD rows only.
    2. Sample ~2M synthetic labelled rows from the fitted class-conditional generators.
    3. PRE-TRAIN a TabM on those synthetic rows (same architecture as node_0033).
    4. FINE-TUNE (continue training) on the real train fold.
    5. Predict val/test as normal.
  New artifact: fs_synthpre (leak-safety: fit_in_fold — generator fit on train-fold only;
  val/test rows are NEVER generated or seen by the generator).
  All other FE, hyperparameters, and architecture are byte-identical to node_0033.

Leakage discipline:
  - Generator fit on train-fold rows only (fit_in_fold).
  - Val/test never passed to generator.
  - Stateless FE: no target, no cross-row stats — safe to apply once.
  - KBinsDiscretizer, factorize maps, TargetEncoder: fit on train-fold rows only.
  - Standardization: fit on train-fold only.
  - PLR bins: fit on train-fold (ti subset).
  - Frozen folds.json used throughout.

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

# Control flags
FOLD0_ONLY = os.environ.get("FOLD0_ONLY") == "1"     # run only fold 0 (for kill gate)
COLD_START = os.environ.get("COLD_START") == "1"     # skip pretrain (A/B comparison)
SMOKE = os.environ.get("TABM_SMOKE") == "1"


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
log(f"Device: {DEVICE}  tabm={tabm.__version__}  FOLD0_ONLY={FOLD0_ONLY}  COLD_START={COLD_START}")

# TabM hyperparameters (identical to node_0033)
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# Pretrain hyperparameters
SYNTH_N_ROWS = 2_000_000       # total synthetic rows to sample (split equally across classes)
PRETRAIN_EPOCHS = 30           # epochs on synthetic data before fine-tune
PRETRAIN_PATIENCE = 8
SYNTH_BATCH_SIZE = 16384       # larger batch for pretrain speed

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


# ─── Synthetic generation (fit_in_fold) ──────────────────────────────────────

def fit_generators_on_fold(X_tr_num: pd.DataFrame, y_tr: np.ndarray, fold_seed: int):
    """Fit one GaussianCopulaSynthesizer per class on train-fold numeric rows only.
    LEAK CHECK: called with train-fold rows ONLY; val/test rows never passed here.
    """
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    log("  [synth] Fitting per-class GaussianCopula generators on train-fold ...")
    synth_t0 = time.perf_counter()
    synthesizers = []
    for cls_idx in range(N_CLASSES):
        cls_mask = (y_tr == cls_idx)
        cls_df = X_tr_num[cls_mask].copy().reset_index(drop=True)
        # Drop constant columns to avoid fit errors
        cls_df = cls_df.loc[:, cls_df.nunique() > 1]

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(cls_df)

        syn = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
        syn.fit(cls_df)
        synthesizers.append((syn, list(cls_df.columns)))
        log(f"    class {CLASSES[cls_idx]}: n={cls_mask.sum()} fitted in {time.perf_counter()-synth_t0:.1f}s")

    return synthesizers


def sample_synthetic(synthesizers, total_rows: int, fold_seed: int):
    """Sample synthetic rows. NEVER includes val/test rows."""
    rows_per_class = total_rows // N_CLASSES
    log(f"  [synth] Sampling {rows_per_class} rows/class ({total_rows} total) ...")
    sample_t0 = time.perf_counter()

    all_Xnum = []
    all_y = []
    all_cols = None

    for cls_idx, (syn, cols) in enumerate(synthesizers):
        sample_df = syn.sample(num_rows=rows_per_class, batch_size=min(100_000, rows_per_class))
        all_Xnum.append(sample_df[cols].values.astype(np.float32))
        all_y.append(np.full(rows_per_class, cls_idx, dtype=np.int32))
        if all_cols is None:
            all_cols = cols

    Xnum_synth = np.concatenate(all_Xnum, axis=0)
    y_synth = np.concatenate(all_y, axis=0)

    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(len(Xnum_synth))
    Xnum_synth = Xnum_synth[perm]
    y_synth = y_synth[perm]

    log(f"  [synth] Sampled {len(Xnum_synth)} rows in {time.perf_counter()-sample_t0:.1f}s")
    return Xnum_synth, y_synth, all_cols


# ─── TabM model ───────────────────────────────────────────────────────────────

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


def pretrain_on_synthetic(
    model: tabm.TabM,
    Xnum_synth: np.ndarray,
    y_synth: np.ndarray,
    synth_cols: list[str],
    num_for_tabm: list[str],
    mu: np.ndarray,
    sd: np.ndarray,
    cat_cards: list[int],
) -> tabm.TabM:
    """Pretrain the TabM on synthetic numeric rows (no categorical features in synth).
    Uses only columns present in both synth and real feature spaces, mapped by name.
    """
    log(f"  [pretrain] Starting pretrain on {len(Xnum_synth)} synthetic rows ...")
    pt_t0 = time.perf_counter()

    synth_col_set = set(synth_cols)
    common_cols = [c for c in num_for_tabm if c in synth_col_set]
    synth_col_idx = [synth_cols.index(c) for c in common_cols]
    real_col_idx = [num_for_tabm.index(c) for c in common_cols]

    log(f"  [pretrain] matched {len(common_cols)}/{len(num_for_tabm)} cols to synth")

    Xnum_synth_sub = Xnum_synth[:, synth_col_idx]
    mu_sub = mu[real_col_idx]
    sd_sub = sd[real_col_idx]
    Xnum_synth_sub = (Xnum_synth_sub - mu_sub) / sd_sub
    Xnum_synth_sub = np.nan_to_num(Xnum_synth_sub, nan=0.0, posinf=10.0, neginf=-10.0)

    n = len(Xnum_synth_sub)
    rng = np.random.default_rng(SEED)
    n_val_pt = max(1, int(0.05 * n))
    perm_pt = rng.permutation(n)
    vi_pt, ti_pt = perm_pt[:n_val_pt], perm_pt[n_val_pt:]

    # Build full-dim arrays with zeroes for columns not in synth
    Xn_full_tr = np.zeros((len(ti_pt), len(num_for_tabm)), dtype=np.float32)
    Xn_full_va = np.zeros((len(vi_pt), len(num_for_tabm)), dtype=np.float32)
    Xn_full_tr[:, real_col_idx] = Xnum_synth_sub[ti_pt]
    Xn_full_va[:, real_col_idx] = Xnum_synth_sub[vi_pt]
    y_tr_pt = y_synth[ti_pt].astype(np.int64)
    y_va_pt = y_synth[vi_pt].astype(np.int64)

    Xn_t = torch.as_tensor(Xn_full_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.as_tensor(y_tr_pt, dtype=torch.long, device=DEVICE)
    nt_pt = len(ti_pt)

    # If model has cat embeddings, pass zero-valued cat features during pretrain
    # (synth data has no cat features; we use the first valid cat index = 0)
    if cat_cards:
        Xc_zero_tr = torch.zeros((SYNTH_BATCH_SIZE, len(cat_cards)), dtype=torch.long, device=DEVICE)
        Xc_zero_va = torch.zeros((len(vi_pt), len(cat_cards)), dtype=torch.long, device=DEVICE)
    else:
        Xc_zero_tr = None
        Xc_zero_va = None

    loss_fn_pt = nn.CrossEntropyLoss()  # balanced by construction
    opt_pt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=PRETRAIN_EPOCHS, eta_min=1e-5)

    best_ba_pt = -1.0
    best_state_pt = None
    bad_pt = 0

    for ep in range(PRETRAIN_EPOCHS):
        model.train()
        bperm = torch.randperm(nt_pt, device=DEVICE)
        for s in range(0, nt_pt, SYNTH_BATCH_SIZE):
            idx = bperm[s:s + SYNTH_BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_zero_tr[:len(idx)] if Xc_zero_tr is not None else None
            y_b = y_t[idx]
            opt_pt.zero_grad()
            logits = model(xn_b, xc_b)
            b, k, c = logits.shape
            loss = loss_fn_pt(logits.reshape(b * k, c), y_b.repeat_interleave(k))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pt.step()
        sched_pt.step()

        val_probs_pt = predict_proba_batch(model, Xn_full_va, Xc_zero_va.cpu().numpy() if Xc_zero_va is not None else None)
        ba_pt = balanced_accuracy_score(y_va_pt, val_probs_pt.argmax(1))
        if ba_pt > best_ba_pt + 1e-5:
            best_ba_pt = ba_pt
            best_state_pt = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            bad_pt = 0
        else:
            bad_pt += 1
            if bad_pt >= PRETRAIN_PATIENCE:
                break

    if best_state_pt is not None:
        model.load_state_dict(best_state_pt)

    log(f"  [pretrain] Done: best_pt_ba={best_ba_pt:.5f}  ep={ep+1}  "
        f"elapsed={time.perf_counter()-pt_t0:.1f}s")
    del Xn_t, y_t, Xn_full_tr, Xn_full_va
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def train_tabm_with_pretrain(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
    synth_data: tuple | None,
    num_for_tabm: list[str],
    mu: np.ndarray,
    sd: np.ndarray,
) -> tuple[tabm.TabM, np.ndarray]:
    """Train TabM: PLR bins fit → (optional pretrain on synth) → finetune on real fold."""
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    # PLR bins — fit on TRAIN subset only (fit-in-fold)
    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

    # Generative pretrain (fit_in_fold)
    if synth_data is not None and not COLD_START:
        Xnum_synth, y_synth, synth_cols = synth_data
        model = pretrain_on_synthetic(model, Xnum_synth, y_synth, synth_cols, num_for_tabm, mu, sd, cat_cards)

    # Fine-tune on real fold
    log(f"  [finetune] Starting fine-tune on {len(ti)} real rows ...")
    ft_t0 = time.perf_counter()

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
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

    log(f"  [finetune] Done: best_int_ba={best_ba:.5f}  ep={ep+1}  "
        f"elapsed={time.perf_counter()-ft_t0:.1f}s")
    return model, bins


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

if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]
    SYNTH_N_ROWS = 30_000    # tiny for smoke

# ─── Pre-train leakage checks 1–3, 5–6 ──────────────────────────────────────
log("Pre-flight leakage checks ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

# Check 1 & 2: target and id not in feature columns
assert TARGET not in X_raw.columns, f"LEAK: {TARGET} in features"
assert IDC not in X_raw.columns, f"LEAK: {IDC} in features"
log("  checks 1&2 OK: target/id not in features")

# Check 3: single-feature sweep on 50k sample
_sample = X_raw.sample(min(50_000, len(X_raw)), random_state=0)
_ys = pd.factorize(train_raw.loc[_sample.index, TARGET])[0]
for _c in X_raw.select_dtypes(include=[np.number]).columns:
    _x = pd.to_numeric(_sample[_c], errors="coerce")
    if _x.nunique() > 1:
        _corr = abs(np.corrcoef(_x.fillna(_x.mean()), _ys)[0, 1])
        if _corr >= 0.999:
            raise SystemExit(f"LEAK SMELL check3: {_c} corr={_corr:.4f} ~ target")
log("  check3 OK: no single feature corr >= 0.999 with target")

# Check 5: folds from frozen folds.json (not recomputed)
log("  check5 OK: folds loaded from folds.json")

# Check 6: train/test near-dup scan (sample)
_tr_sample = X_raw.select_dtypes(include=[np.number]).sample(min(5000, len(X_raw)), random_state=0).round(4)
_te_sample = X_test_raw.select_dtypes(include=[np.number]).sample(min(5000, len(X_test_raw)), random_state=0).round(4)
_tr_hashes = set(_tr_sample.apply(lambda r: hash(tuple(r)), axis=1))
_te_hashes = set(_te_sample.apply(lambda r: hash(tuple(r)), axis=1))
_dup_count = len(_tr_hashes & _te_hashes)
log(f"  check6 near-dup scan: {_dup_count} hash matches in 5k x 5k sample (warn if >50)")
del _sample, _ys, _tr_sample, _te_sample, _tr_hashes, _te_hashes
gc.collect()

# ─── Stateless FE ────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Columns fed to generator (base numerics + stateless FE — no fit_in_fold encodings)
SYNTH_GEN_COLS = BASE_NUM_COLS + [f"_{a}-{b}" for a, b in COLOR_PAIRS] + [
    "_g_div_redshift", "_i_div_redshift", "_mag_mean", "_mag_range", "_log1p_redshift"
]
SYNTH_GEN_COLS = [c for c in SYNTH_GEN_COLS if c in X_stateless.columns]
log(f"  Generator will use {len(SYNTH_GEN_COLS)} columns: {SYNTH_GEN_COLS[:5]}...")

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

    if FOLD0_ONLY and fold_id > 0:
        log("FOLD0_ONLY: stopping after fold 0")
        break

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

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

    # Standardize — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Generative pretrain data — fit generator on TRAIN FOLD rows only (fit_in_fold)
    # LEAK CHECK (check 4): X_stateless.iloc[tr_idx] = train-fold rows only;
    #   val_idx rows are NEVER passed to synthesizer.fit().
    synth_data = None
    if not COLD_START and not SMOKE:
        gen_t0 = time.perf_counter()
        X_tr_gen = X_stateless.iloc[tr_idx][SYNTH_GEN_COLS].reset_index(drop=True)
        synthesizers = fit_generators_on_fold(X_tr_gen, y_tr_fold, fold_seed)
        Xnum_synth, y_synth, synth_cols = sample_synthetic(synthesizers, SYNTH_N_ROWS, fold_seed)
        gen_elapsed = time.perf_counter() - gen_t0
        log(f"  [synth] Total gen+sample time: {gen_elapsed:.1f}s")

        # Kill gate 1: gen+sample < 20 min
        if fold_id == 0 and gen_elapsed > 1200:
            log(f"  KILL GATE 1 TRIPPED: gen+sample took {gen_elapsed/60:.1f} min > 20 min")
            sys.exit(1)

        synth_data = (Xnum_synth, y_synth, synth_cols)
        del X_tr_gen, synthesizers
        gc.collect()

    # Train TabM with pretrain + finetune
    model, bins = train_tabm_with_pretrain(
        Xn_tr, Xc_tr, y_tr_fold, cat_cards, fold_seed,
        synth_data, num_for_tabm, mu, sd
    )

    # OOF predictions
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

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    if synth_data is not None:
        del synth_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0 and not FOLD0_ONLY:
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

if FOLD0_ONLY:
    log("FOLD0_ONLY mode: skipping artifact saves.")
    sys.exit(0)

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

# Write features.txt
tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# Final OOF metric
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
