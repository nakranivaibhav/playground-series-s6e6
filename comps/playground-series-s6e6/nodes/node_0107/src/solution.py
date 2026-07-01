"""node_0107 — TabM on RICH flux-space FE (fs_flux_rich).

THE ONE ATOMIC CHANGE vs node_0033:
  Replace the feature matrix with fs_flux_rich (stateless, row-wise deterministic).
  The training loop (TabM + PLR, k=32) is byte-identical to node_0033.

fs_flux_rich features (stateless):
  - f_b = 10^(−0.4·(mag_b − mag_mean))  for b in {u,g,r,i,z}  (5 features)
  - ALL pairwise flux ratios f_b/f_b'  (10 pairs)
  - unit-sum SED simplex  (5 features)
  - flux aggregates: flux mean, flux range/spread  (2 features)
  - brightest-band one-hot (5 features)
  - faintest-band one-hot  (5 features)
  - flux × redshift interactions  (5 features)
  - flux-ratio × redshift interactions  (10 features)
  - raw redshift + log1p(redshift)  (2 features)
  Total numeric: 5+10+5+2+5+5+5+10+2 = 49 features
  + 2 engineered categoricals (spectral_type, galaxy_population) fed natively to TabM.
NO raw magnitudes, NO log colors.

Kill gates (in node.md):
  - fold-0 solo BA >= 0.965
  - fold-0 err-corr vs node_0070 < 0.65
  If either trips, STOP after fold-0.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv
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

# Categorical columns to pass natively to TabM (low-cardinality)
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]


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

# ─── fs_flux_rich feature engineering (STATELESS — the ONE change) ───────────
MAG_BANDS = ["u", "g", "r", "i", "z"]
BAND_PAIRS = [(a, b) for i, a in enumerate(MAG_BANDS) for b in MAG_BANDS[i + 1:]]  # 10 pairs


def add_flux_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise stateless flux-space features (fs_flux_rich).
    No fit, no target, no cross-row stats. No raw magnitudes, no log colors.

    Numeric features (49 total):
      - 5  linear fluxes  f_b = 10^(-0.4*(mag_b - mag_mean))
      - 10 pairwise flux ratios f_b/f_b'
      - 5  unit-sum SED simplex
      - 2  flux aggregates: flux mean, flux range
      - 5  brightest-band one-hot
      - 5  faintest-band one-hot
      - 5  flux x redshift interactions
      - 10 flux-ratio x redshift interactions
      - 2  raw redshift + log1p(redshift)
    Categorical features (kept from raw df, passed through as-is):
      - spectral_type, galaxy_population
    """
    df = df.copy()

    mags = df[MAG_BANDS].astype(np.float32).values  # (N, 5)
    mag_mean = mags.mean(axis=1, keepdims=True)  # (N, 1)
    rel_mags = mags - mag_mean  # (N, 5)

    # Linear fluxes
    fluxes = np.power(10.0, -0.4 * rel_mags).astype(np.float32)  # (N, 5)

    result = {}

    # 5 fluxes
    for j, b in enumerate(MAG_BANDS):
        result[f"_flux_{b}"] = fluxes[:, j]

    # 10 pairwise flux ratios
    for a, b in BAND_PAIRS:
        fa = fluxes[:, MAG_BANDS.index(a)]
        fb = fluxes[:, MAG_BANDS.index(b)]
        ratio = np.clip(fa / (fb + 1e-9), -1e6, 1e6).astype(np.float32)
        result[f"_fratio_{a}_{b}"] = ratio

    # 5 SED simplex (unit-sum)
    flux_sum = fluxes.sum(axis=1, keepdims=True) + 1e-9
    simplex = (fluxes / flux_sum).astype(np.float32)
    for j, b in enumerate(MAG_BANDS):
        result[f"_fsimplex_{b}"] = simplex[:, j]

    # 2 flux aggregates: mean and range
    result["_flux_mean"] = fluxes.mean(axis=1).astype(np.float32)
    result["_flux_range"] = (fluxes.max(axis=1) - fluxes.min(axis=1)).astype(np.float32)

    # 5 brightest-band one-hot
    brightest = np.argmax(fluxes, axis=1)  # (N,)
    for j, b in enumerate(MAG_BANDS):
        result[f"_bright_{b}"] = (brightest == j).astype(np.float32)

    # 5 faintest-band one-hot
    faintest = np.argmin(fluxes, axis=1)  # (N,)
    for j, b in enumerate(MAG_BANDS):
        result[f"_faint_{b}"] = (faintest == j).astype(np.float32)

    # redshift
    redshift = df["redshift"].astype(np.float32).values  # (N,)
    result["_redshift"] = redshift
    result["_log1p_redshift"] = np.log1p(np.clip(redshift, -1 + 1e-6, None)).astype(np.float32)

    # 5 flux x redshift interactions
    for j, b in enumerate(MAG_BANDS):
        result[f"_flux_z_{b}"] = (fluxes[:, j] * redshift).astype(np.float32)

    # 10 flux-ratio x redshift interactions
    for a, b in BAND_PAIRS:
        ratio_vals = result[f"_fratio_{a}_{b}"]
        result[f"_fratioZ_{a}_{b}"] = (ratio_vals * redshift).astype(np.float32)

    # Build result df (numeric only — no mags)
    num_df = pd.DataFrame(result, index=df.index)

    # Keep the two engineered categoricals from original df
    cat_df = df[BASE_CAT_COLS].copy() if all(c in df.columns for c in BASE_CAT_COLS) else pd.DataFrame(index=df.index)

    return pd.concat([num_df, cat_df], axis=1)


# ─── Categorical encoding (fit_in_fold for factorize maps) ───────────────────

def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Factorize the two engineered categoricals on train-fold only.
    Returns (df_tr, df_val, df_te, cat_cards).
    Called INSIDE the fold loop — fit_in_fold.
    """
    local_map = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    cat_cards = []
    for col in BASE_CAT_COLS:
        if col not in tr.columns:
            continue
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        n_unique = len(uniques)
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")
        cat_cards.append(n_unique + 1)  # +1 for unseen safety

    return tr, va, te, cat_cards


# ─── TabM training (byte-identical to node_0033) ─────────────────────────────

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


def train_tabm(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
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

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

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
    log(f"    TabM early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
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

# ─── fs_flux_rich stateless FE (computed once, safe) ──────────────────────────
log("Applying fs_flux_rich stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_flux = add_flux_rich_features(X_raw)
X_test_flux = add_flux_rich_features(X_test_raw)

num_feature_cols = [c for c in X_flux.columns if c not in BASE_CAT_COLS]
log(f"  X_flux={X_flux.shape}  X_test_flux={X_test_flux.shape}")
log(f"  numeric_features={len(num_feature_cols)}  cat_features={len(BASE_CAT_COLS)}")
log(f"  numeric cols: {num_feature_cols}")

# ─── Pre-flight leakage checks (BEFORE training) ──────────────────────────────
# Check 1+2: target/id not in features
assert TARGET not in X_flux.columns, f"LEAK: {TARGET} in features"
assert IDC not in X_flux.columns, f"LEAK: {IDC} in features"
log("Pre-flight: target/id not in features — OK")

# Check 3: single-feature sweep on sample (numeric only)
sample_size = min(50_000, n_train)
rng_check = np.random.default_rng(0)
sample_idx = rng_check.choice(n_train, sample_size, replace=False)
ys = y_all[sample_idx]
for c in num_feature_cols:
    x = X_flux[c].values[sample_idx].astype(np.float32)
    if np.isfinite(x).sum() > 1 and x[np.isfinite(x)].std() > 0:
        x_filled = np.where(np.isfinite(x), x, np.nanmean(x))
        corr = abs(np.corrcoef(x_filled, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {c} corr={corr:.5f} with target")
log("Pre-flight: single-feature sweep — no leak smell")

# Check 4: add_flux_rich_features is stateless (no .fit calls — verified by inspection)
log("Pre-flight: fs_flux_rich is stateless — no fit transforms — OK")
# The factorize maps for categoricals are fit INSIDE the fold loop (fit_in_fold).

# Check 5: folds from frozen folds.json (verified — loaded above from COMP_DIR/folds.json)
log("Pre-flight: folds from frozen folds.json — OK")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

gate_tripped = False  # will be set if a kill gate trips

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

    # Categorical encoding — fit_in_fold (factorize maps fit on train fold only)
    X_tr_fold, X_val_fold, X_te_fold, cat_cards = fit_fold_categoricals(
        X_flux.iloc[tr_idx].reset_index(drop=True),
        X_flux.iloc[val_idx].reset_index(drop=True),
        X_test_flux.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Identify cols: numeric vs cat
    all_cols = sorted(X_tr_fold.columns)
    TABM_CAT_COLS = [c for c in all_cols if c in BASE_CAT_COLS]
    num_for_tabm = [c for c in all_cols if c not in TABM_CAT_COLS]

    if fold_id == 0:
        log(f"  n_features={X_tr_fold.shape[1]}  tabm_cat={len(TABM_CAT_COLS)}  tabm_num={len(num_for_tabm)}")

    # Extract arrays
    Xn_tr = X_tr_fold[num_for_tabm].values.astype(np.float32)
    Xn_va = X_val_fold[num_for_tabm].values.astype(np.float32)
    Xn_te = X_te_fold[num_for_tabm].values.astype(np.float32)

    if TABM_CAT_COLS:
        Xc_tr = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_va = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_te = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
        card_arr = np.array(cat_cards) - 1
        Xc_tr = np.clip(Xc_tr, 0, card_arr)
        Xc_va = np.clip(Xc_va, 0, card_arr)
        Xc_te = np.clip(Xc_te, 0, card_arr)
    else:
        Xc_tr = Xc_va = Xc_te = None
        cat_cards = []

    # Standardize numerical features — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Train TabM
    model, bins = train_tabm(Xn_tr, Xc_tr, y_tr_fold, cat_cards, fold_seed)

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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

        # ── Decorrelation check vs node_0070 OOF (GATE 2) ────────────────────
        bank_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
        mean_err_corr = float("nan")
        if bank_oof_path.exists():
            bank_oof = np.load(str(bank_oof_path))  # (577347, 3)
            this_err = (np.argmax(oof_proba[val_idx], axis=1) != y_val_fold).astype(float)
            bank_preds_val = np.argmax(bank_oof[val_idx], axis=1)
            bank_err = (bank_preds_val != y_val_fold).astype(float)
            per_class_corrs = []
            for cls in range(N_CLASSES):
                mask = (y_val_fold == cls)
                if mask.sum() > 10:
                    this_c = this_err[mask]
                    bank_c = bank_err[mask]
                    if this_c.std() > 0 and bank_c.std() > 0:
                        c = np.corrcoef(this_c, bank_c)[0, 1]
                        per_class_corrs.append(c)
            mean_err_corr = float(np.mean(per_class_corrs)) if per_class_corrs else float("nan")
            log(f"  DECORRELATION: fold0 err-corr vs node_0070 = {mean_err_corr:.4f}  (GATE 2: kill if >=0.65)")
            print(f"fold0_err_corr_vs_n0070={mean_err_corr:.6f}", flush=True)
        else:
            log(f"  WARNING: bank OOF not found at {bank_oof_path}")

        # ── GATE 1: fold-0 BA >= 0.965 ────────────────────────────────────────
        if fold_score < 0.965:
            log(f"  GATE 1 TRIPPED: fold0 BA={fold_score:.6f} < 0.965 — STOPPING")
            print(f"KILL_BA: fold0_ba={fold_score:.6f}", flush=True)
            gate_tripped = True
            break

        log(f"  GATE 1 OK: fold0 BA={fold_score:.6f} >= 0.965")

        # ── GATE 2: err-corr < 0.65 ───────────────────────────────────────────
        if not np.isnan(mean_err_corr) and mean_err_corr >= 0.65:
            log(f"  GATE 2 TRIPPED: fold0 err-corr={mean_err_corr:.4f} >= 0.65 — STOPPING")
            print(f"KILL_DECORR: err_corr={mean_err_corr:.4f}", flush=True)
            gate_tripped = True
            break

        if not np.isnan(mean_err_corr):
            log(f"  GATE 2 OK: fold0 err-corr={mean_err_corr:.4f} < 0.65")
        else:
            log("  GATE 2: bank OOF not found, skipping decorrelation gate")

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

if gate_tripped:
    log("Gate tripped after fold-0. Not saving full artifacts.")
    sys.exit(0)

# ─── Full run stats ────────────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# Final err-corr check across all folds vs node_0070
bank_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
if bank_oof_path.exists():
    bank_oof = np.load(str(bank_oof_path))
    folds_all = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
    all_val_idx = np.concatenate([np.asarray(f["val_idx"], dtype=int) for f in folds_all])
    this_err_all = (np.argmax(oof_proba[all_val_idx], axis=1) != y_all[all_val_idx]).astype(float)
    bank_err_all = (np.argmax(bank_oof[all_val_idx], axis=1) != y_all[all_val_idx]).astype(float)
    per_class_corrs_full = []
    for cls in range(N_CLASSES):
        mask = (y_all[all_val_idx] == cls)
        if mask.sum() > 10:
            this_c = this_err_all[mask]
            bank_c = bank_err_all[mask]
            if this_c.std() > 0 and bank_c.std() > 0:
                c = np.corrcoef(this_c, bank_c)[0, 1]
                per_class_corrs_full.append(c)
    final_err_corr = float(np.mean(per_class_corrs_full)) if per_class_corrs_full else float("nan")
    log(f"  FINAL err-corr vs node_0070 = {final_err_corr:.4f}  (stack-add bar: <0.65)")
    print(f"final_err_corr_vs_n0070={final_err_corr:.6f}", flush=True)

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

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
