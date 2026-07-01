"""node_0103 — TabM on linear flux-space features (fs_flux).

THE ONE ATOMIC CHANGE vs node_0033:
  Replace the feature matrix with fs_flux (stateless, row-wise deterministic).
  The training loop (TabM + PLR, k=32) is byte-identical to node_0033.

fs_flux features (stateless):
  - f_b = 10^(−0.4·(mag_b − mag_mean)) for b in {u,g,r,i,z}  (mag_mean = per-row mean of 5 mags)
  - pairwise flux RATIOS f_b/f_b' (all 10 pairs)
  - the 5-flux vector normalized to unit sum (SED-shape simplex)
  - + raw redshift
  NO magnitudes, NO log colors.

Kill gates (see plan):
  - FOLD0_ONLY=1 env var runs only fold 0 for the cheap-kill check.
  - After fold-0, err-corr vs node_0070 OOF is printed; caller decides.

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

FOLD0_ONLY = os.environ.get("FOLD0_ONLY") == "1"
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

# ─── fs_flux feature engineering (STATELESS — the ONE change) ────────────────
MAG_BANDS = ["u", "g", "r", "i", "z"]
BAND_PAIRS = [(a, b) for i, a in enumerate(MAG_BANDS) for b in MAG_BANDS[i + 1:]]  # 10 pairs


def add_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise stateless flux-space features. No fit, no target, no cross-row stats.

    Features:
      - f_b = 10^(−0.4·(mag_b − mag_mean))  for b in {u,g,r,i,z}
      - pairwise flux ratios f_b/f_b'  (10 pairs)
      - simplex-normalized fluxes (unit-sum SED shape)  (5 features)
      - raw redshift
    Total: 5 + 10 + 5 + 1 = 21 features
    """
    df = df.copy()

    mags = df[MAG_BANDS].astype(np.float32)
    mag_mean = mags.mean(axis=1).values  # (N,)

    # Relative mags: mag_b - mag_mean
    rel_mags = mags.values - mag_mean[:, None]  # (N, 5)

    # Linear fluxes: f_b = 10^(−0.4 * (mag_b − mag_mean))
    fluxes = np.power(10.0, -0.4 * rel_mags).astype(np.float32)  # (N, 5)

    flux_df = pd.DataFrame(
        fluxes, columns=[f"_flux_{b}" for b in MAG_BANDS], index=df.index
    )
    for col in flux_df.columns:
        df[col] = flux_df[col]

    # Pairwise flux ratios (all 10 pairs)
    for a, b in BAND_PAIRS:
        fa = fluxes[:, MAG_BANDS.index(a)]
        fb = fluxes[:, MAG_BANDS.index(b)]
        ratio = (fa / (fb + 1e-9)).astype(np.float32)
        ratio = np.clip(ratio, -1e6, 1e6)
        df[f"_fratio_{a}_{b}"] = ratio

    # Unit-sum simplex normalization
    flux_sum = fluxes.sum(axis=1, keepdims=True) + 1e-9
    simplex = (fluxes / flux_sum).astype(np.float32)
    for j, b in enumerate(MAG_BANDS):
        df[f"_fsimplex_{b}"] = simplex[:, j]

    # Raw redshift (only numeric, no transform)
    df["_redshift"] = df["redshift"].astype(np.float32)

    # Drop all original columns — we only keep the 21 flux features
    keep_cols = (
        [f"_flux_{b}" for b in MAG_BANDS]
        + [f"_fratio_{a}_{b}" for a, b in BAND_PAIRS]
        + [f"_fsimplex_{b}" for b in MAG_BANDS]
        + ["_redshift"]
    )
    return df[keep_cols]


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

# ─── fs_flux stateless FE (computed once, safe) ───────────────────────────────
log("Applying fs_flux stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_flux = add_flux_features(X_raw)
X_test_flux = add_flux_features(X_test_raw)
log(f"  X_flux={X_flux.shape}  X_test_flux={X_test_flux.shape}  features={list(X_flux.columns)}")

# Pre-flight leakage checks (BEFORE training)
# 1+2: target/id not in features
assert TARGET not in X_flux.columns, f"LEAK: {TARGET} in features"
assert IDC not in X_flux.columns, f"LEAK: {IDC} in features"
log("Pre-flight: target/id not in features — OK")

# 3: single-feature sweep on sample
sample_size = min(50_000, n_train)
rng_check = np.random.default_rng(0)
sample_idx = rng_check.choice(n_train, sample_size, replace=False)
ys = y_all[sample_idx]
for c in X_flux.columns:
    x = X_flux[c].values[sample_idx].astype(np.float32)
    if np.isfinite(x).sum() > 1 and x[np.isfinite(x)].std() > 0:
        x_filled = np.where(np.isfinite(x), x, np.nanmean(x))
        corr = abs(np.corrcoef(x_filled, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {c} corr={corr:.5f} with target")
log("Pre-flight: single-feature sweep — no leak smell")

# 4: all FE is stateless — no .fit — verified by inspection (add_flux_features has no sklearn fit calls)
log("Pre-flight: fs_flux is stateless — no fit transforms — OK")

# ─── Restrict to fold-0 only if FOLD0_ONLY ────────────────────────────────────
if FOLD0_ONLY:
    folds_list = [folds_list[0]]
    log("FOLD0_ONLY mode: running fold 0 only")

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

    # fs_flux: all numerical, no categoricals
    Xn_tr = X_flux.values[tr_idx].astype(np.float32)
    Xn_va = X_flux.values[val_idx].astype(np.float32)
    Xn_te = X_test_flux.values.astype(np.float32)
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Standardize numerical features — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    if fold_id == 0:
        log(f"  n_features={Xn_tr.shape[1]}  tabm_num={Xn_tr.shape[1]}  no cats (all numeric)")

    # Train TabM (PLR bins fit inside train_model on ti subset of train fold)
    model, bins = train_tabm(Xn_tr, None, y_tr_fold, [], fold_seed)

    # OOF predictions
    val_probs = predict_proba_batch(model, Xn_va, None)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions — average across folds
    test_probs_fold = predict_proba_batch(model, Xn_te, None)
    # For FOLD0_ONLY, accumulate as if 5 folds (divide by 5); final artifact not saved in this mode
    n_folds_total = json.loads((COMP_DIR / "folds.json").read_text())
    n_folds_total = len(n_folds_total["folds"])
    test_proba_accum += test_probs_fold.astype(np.float32) / n_folds_total

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * 5
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

        # ── Decorrelation check vs node_0070 OOF ──────────────────────────────
        bank_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
        if bank_oof_path.exists():
            bank_oof = np.load(str(bank_oof_path))  # (577347, 3)
            # Errors on val_idx of fold 0
            this_err = (np.argmax(oof_proba[val_idx], axis=1) != y_val_fold).astype(float)
            bank_preds_val = np.argmax(bank_oof[val_idx], axis=1)
            bank_err = (bank_preds_val != y_val_fold).astype(float)
            # Per-class error correlation
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
            log(f"  DECORRELATION: fold0 err-corr vs node_0070 = {mean_err_corr:.4f}  (kill if >=0.75)")
            print(f"fold0_err_corr_vs_n0070={mean_err_corr:.6f}", flush=True)

            if mean_err_corr >= 0.75:
                log(f"  DECORRELATION KILL TRIPPED: err-corr={mean_err_corr:.4f} >= 0.75")
                log(f"  Flux ratios are re-encoded color space. STOPPING.")
                print(f"KILL_DECORR: err_corr={mean_err_corr:.4f}", flush=True)
                sys.exit(0)
            else:
                log(f"  Decorrelation OK: err-corr={mean_err_corr:.4f} < 0.75")
        else:
            log(f"  WARNING: bank OOF not found at {bank_oof_path}")
            mean_err_corr = float("nan")

        # ── Cheap-kill gate ────────────────────────────────────────────────────
        if fold_score < 0.965:
            log(f"  CHEAP-KILL TRIPPED: fold0 BA={fold_score:.6f} < 0.965")
            print(f"KILL_BA: fold0_ba={fold_score:.6f}", flush=True)
            sys.exit(0)
        else:
            log(f"  Cheap-kill OK: fold0 BA={fold_score:.6f} >= 0.965")

        if FOLD0_ONLY:
            log("FOLD0_ONLY complete — exiting without full artifacts.")
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
    all_val_mask = np.zeros(n_train, dtype=bool)
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
