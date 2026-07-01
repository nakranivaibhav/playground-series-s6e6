"""node_0087 — LightGBM base trained on fs_zresid (z-conditional color/mag residuals).

fs_zresid feature set (fit_in_fold):
  - ~40 redshift quantile bins (edges fit on train-fold only)
  - per-bin MEAN/STD of each color (u-g, g-r, r-i, i-z, u-z) and each magnitude
    fit on train-fold only; z-conditional z-score = (value - mean_zbin) / std_zbin
  - global mean/std fallback for sparse bins (<5 samples)
  - raw redshift KEPT (STAR z≈0 is itself discriminative)
  - raw colors DROPPED (residual-dominated, not additive)

Leakage: bin edges + per-bin stats fit on train fold only — applied to val+test.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

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
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

COLORS = ["u-g", "g-r", "r-i", "i-z", "u-z"]
MAGS = ["u", "g", "r", "i", "z"]
N_ZBINS = 40
MIN_BIN_SAMPLES = 5


def compute_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw colors for internal use in fs_zresid (not kept in final features)."""
    d = df.copy()
    d["u-g"] = (d["u"] - d["g"]).astype("float32")
    d["g-r"] = (d["g"] - d["r"]).astype("float32")
    d["r-i"] = (d["r"] - d["i"]).astype("float32")
    d["i-z"] = (d["i"] - d["z"]).astype("float32")
    d["u-z"] = (d["u"] - d["z"]).astype("float32")
    return d


def fit_zresid(df_tr: pd.DataFrame):
    """
    Fit fs_zresid encoder on train-fold rows ONLY.
    Returns (bin_edges, per_bin_stats, global_stats).
    bin_edges: 1D array of N_ZBINS+1 edges based on train-fold redshift quantiles.
    per_bin_stats: dict {feature: [(mean, std), ...]} for each of N_ZBINS bins.
    global_stats: dict {feature: (mean, std)} fallback.
    """
    z = df_tr["redshift"].values
    # Compute quantile edges from training fold only
    quantiles = np.linspace(0, 100, N_ZBINS + 1)
    bin_edges = np.percentile(z, quantiles)
    # Ensure strictly increasing (collapse duplicate edges)
    bin_edges = np.unique(bin_edges)

    # Feature values to z-score
    tr_with_colors = compute_colors(df_tr)
    features = COLORS + MAGS

    # Assign each train row to a bin
    # np.searchsorted on bin_edges (left), then clip to valid bin range
    bin_ids = np.searchsorted(bin_edges[1:-1], z, side="right")  # 0..len(bin_edges)-2
    n_actual_bins = len(bin_edges) - 1

    per_bin_stats = {}
    global_stats = {}
    for feat in features:
        vals = tr_with_colors[feat].values.astype(float)
        global_mean = np.nanmean(vals)
        global_std = np.nanstd(vals)
        if global_std < 1e-10:
            global_std = 1.0
        global_stats[feat] = (global_mean, global_std)

        bin_means = []
        bin_stds = []
        for b in range(n_actual_bins):
            mask = bin_ids == b
            bvals = vals[mask]
            if mask.sum() >= MIN_BIN_SAMPLES:
                bm = np.nanmean(bvals)
                bs = np.nanstd(bvals)
                if bs < 1e-10:
                    bs = global_std
            else:
                # sparse — use global
                bm = global_mean
                bs = global_std
            bin_means.append(bm)
            bin_stds.append(bs)
        per_bin_stats[feat] = (np.array(bin_means), np.array(bin_stds))

    return bin_edges, per_bin_stats, global_stats


def apply_zresid(df: pd.DataFrame, bin_edges, per_bin_stats, global_stats) -> pd.DataFrame:
    """
    Apply fs_zresid transform to any dataframe (train fold, val, test).
    Returns a new DataFrame with:
      - z-conditional residuals for COLORS + MAGS (replace raw)
      - raw redshift kept
      - raw individual magnitudes DROPPED (residual replaces them)
      - raw colors DROPPED (never were columns, we just compute them here)
      - categorical columns kept as-is
    """
    df_c = compute_colors(df)
    z = df["redshift"].values

    # Assign bins
    bin_ids = np.searchsorted(bin_edges[1:-1], z, side="right")
    n_actual_bins = len(bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_actual_bins - 1)

    features = COLORS + MAGS
    residuals = {}
    for feat in features:
        vals = df_c[feat].values.astype(float)
        gm, gs = global_stats[feat]
        bm_arr, bs_arr = per_bin_stats[feat]

        means = np.where(bin_ids < len(bm_arr), bm_arr[bin_ids], gm)
        stds  = np.where(bin_ids < len(bs_arr), bs_arr[bin_ids], gs)
        stds  = np.where(stds < 1e-10, gs, stds)

        residuals[f"zr_{feat}"] = ((vals - means) / stds).astype("float32")

    out = pd.DataFrame(residuals, index=df.index)
    # Keep raw redshift
    out["redshift"] = df["redshift"].values.astype("float32")
    # Keep categorical / sky coords
    for col in ["alpha", "delta", "spectral_type", "galaxy_population"]:
        if col in df.columns:
            out[col] = df[col].values

    return out


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=fold_seed,
        verbosity=-1,
        device="cpu",
    )


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# Raw feature frames (no target, no id)
X_raw     = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

# ─── Pre-flight leakage check 1-2: target and id not in features ─────────────
assert TARGET not in X_raw.columns, "TARGET in features!"
assert IDC not in X_raw.columns, "ID in features!"
log("Leakage check 1-2: PASS (target/id not in feature set)")

# ─── Pre-flight leakage check 3: single-feature ~ target sweep (sample 50k) ──
log("Leakage check 3: single-feature ~ target sweep ...")
rng_lk = np.random.RandomState(0)
sidx = rng_lk.choice(n_train, min(50_000, n_train), replace=False)
ys_sample = y_all[sidx].astype(float)
# Check raw features (zresid computed in-fold; check raw input cols)
for col in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
    xv = X_raw.iloc[sidx][col].values.astype(float)
    if np.std(xv) > 1e-10:
        corr = abs(np.corrcoef(xv, ys_sample)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {col} ~ target corr={corr:.4f}")
log("Leakage check 3: PASS")

# ─── Pre-flight check 5: folds from frozen file ───────────────────────────────
log(f"Leakage check 5: folds from frozen folds.json, {len(folds_list)} folds — PASS")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
best_iters = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    df_tr_raw  = X_raw.iloc[tr_idx].reset_index(drop=True)
    df_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
    df_te_raw  = X_test_raw.reset_index(drop=True)

    # fit_in_fold: bin edges + per-bin stats from train fold ONLY
    bin_edges, per_bin_stats, global_stats = fit_zresid(df_tr_raw)

    # Apply to all splits
    X_tr_fold  = apply_zresid(df_tr_raw,  bin_edges, per_bin_stats, global_stats)
    X_val_fold = apply_zresid(df_val_raw, bin_edges, per_bin_stats, global_stats)
    X_te_fold  = apply_zresid(df_te_raw,  bin_edges, per_bin_stats, global_stats)

    # Encode categoricals naively (factorize on train fold)
    for col in ["spectral_type", "galaxy_population"]:
        codes_tr, uniq = pd.factorize(X_tr_fold[col], sort=False)
        X_tr_fold[col] = codes_tr.astype("int32")
        code_map = {c: i for i, c in enumerate(uniq)}
        X_val_fold[col] = X_val_fold[col].map(code_map).fillna(-1).astype("int32")
        X_te_fold[col]  = X_te_fold[col].map(code_map).fillna(-1).astype("int32")

    # Sort columns
    cols_sorted = sorted(X_tr_fold.columns)
    X_tr_fold  = X_tr_fold[cols_sorted]
    X_val_fold = X_val_fold[cols_sorted]
    X_te_fold  = X_te_fold[cols_sorted]

    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    if fold_id == 0:
        log(f"  feature set ({len(cols_sorted)} features): {cols_sorted}")

    model = make_lgbm(fold_seed=fold_seed)
    model.fit(
        X_tr_fold, y_tr_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=200),
        ],
    )

    best_iter = model.best_iteration_
    best_iters.append(best_iter)
    log(f"  best_iteration={best_iter}")

    val_proba = model.predict_proba(X_val_fold)
    oof_proba[val_idx] = val_proba.astype("float32")

    test_proba_fold = model.predict_proba(X_te_fold)
    test_proba_accum += test_proba_fold.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold={','.join(f'{s:.6f}' for s in per_fold_scores)}")
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF / test_probs ────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved oof.npy={oof_proba.shape}  test_probs.npy={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

# ─── Post-run output gates ────────────────────────────────────────────────────
assert oof_proba.shape == (n_train, N_CLASSES), f"oof shape {oof_proba.shape}"
assert not np.isnan(oof_proba).any(), "NaN in OOF"
row_sums = oof_proba.sum(axis=1)
assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: {row_sums.mean():.4f}"
log("Output gates: oof_full PASS  no_nan PASS  dist_sane PASS")

# ─── GATE: OOF error-correlation vs bank17+FT-T (n76 feature matrix) ─────────
log("\n=== GATE: pairwise OOF error-correlation vs bank17+FT-T ===")

def logp(a):
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a):
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True); s[s == 0] = 1
    return a / s

def score_fn(y_true, y_pred):
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES) if (y_true == c).any()]))

def rd(path, nr):
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(CLASSES).issubset(c): return d[CLASSES].values.astype(float)
    pc = [f"prob_{l}" for l in CLASSES]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3: return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float); return v.reshape(nr, 3)

def load_ext_csv(path, nr):
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)

B = COMP_DIR / "refs/oof_bank"
K = COMP_DIR / "refs/kernel_out"

MANIFEST = {
    'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",         K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
    'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",           K/"xgb-v1-for-s6e6/test_preds.npy"),
    'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",               B/"test_preds_realmlp0_v12.csv"),
    'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",        K/"realmlp-v1-for-s6e6/test_preds.npy"),
    'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                   B/"test_preds_tabm0_v2.csv"),
    'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
    'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
    'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
    'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
    'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
    'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
    'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
    'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
    'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
    'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
    'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
    'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
    'xgb-6':      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
    'tabm-1':     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
}

POOF = {}; PTEST = {}; good = []
for name, (op, tp) in MANIFEST.items():
    try:
        o = norm(rd(op, n_train)); t = norm(rd(tp, n_test))
        assert o.shape == (n_train, 3) and t.shape == (n_test, 3)
        ba = balanced_accuracy_score(y_all, o.argmax(1))
        if 0.90 < ba < 0.972:
            POOF[name] = o; PTEST[name] = t; good.append(name)
        log(f"  {name}: BA={ba:.6f}  {'OK' if name in POOF else 'SKIP'}")
    except Exception as e:
        log(f"  {name}: FAIL {str(e)[:60]}")

log(f"Loaded {len(good)} bank models")

# FT-Transformer
PILK = COMP_DIR / "refs/ext_oof/pilkwang_5090"
ft_oof_raw  = load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n_train)
ft_test_raw = load_ext_csv(PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n_test)
ft_solo_ba = score_fn(y_all, norm(ft_oof_raw).argmax(1))
log(f"FT-T solo BA={ft_solo_ba:.6f}")
assert ft_solo_ba > 0.85, f"FT-T BA {ft_solo_ba:.4f} too low"

# Pairwise OOF error-correlation: this node's errors vs each bank member
# "error" = 1 - correct; binary mistake indicator per row
this_err = (oof_proba.argmax(1) != y_all).astype(float)
log(f"\nThis node (node_0087) OOF BA={mean_cv:.6f}  error_rate={this_err.mean():.4f}")
log("Pairwise error-correlation vs bank17:")
corrs = []
for name in good:
    bank_err = (POOF[name].argmax(1) != y_all).astype(float)
    corr = np.corrcoef(this_err, bank_err)[0, 1]
    corrs.append(corr)
    log(f"  {name}: err_corr={corr:.4f}")
ft_err = (norm(ft_oof_raw).argmax(1) != y_all).astype(float)
ft_corr = np.corrcoef(this_err, ft_err)[0, 1]
log(f"  ft_transformer: err_corr={ft_corr:.4f}")
all_corrs = corrs + [ft_corr]
log(f"Mean err_corr vs bank18: {np.mean(all_corrs):.4f}  max={np.max(all_corrs):.4f}")

# ─── Reproduce n76 baseline exactly: bank17 + FT-T, 5-seed bagged LogReg ─────
log("\n=== Reproducing n76 baseline (bank17+FT-T 5-seed bagged LogReg) ===")

all_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
all_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
OOF_full = np.concatenate(all_oof_logp, axis=1)
TST_full = np.concatenate(all_test_logp, axis=1)
log(f"Feature matrix: {OOF_full.shape}")

SEEDS = [42, 43, 44, 45, 46]
fval = [np.asarray(f["val_idx"]) for f in folds_list]

def fit_meta(Xtr, ytr, seed=42):
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m

seed_oof_probs  = np.zeros((len(SEEDS), n_train, N_CLASSES))
seed_test_probs = np.zeros((len(SEEDS), n_test, N_CLASSES))

for si, seed in enumerate(SEEDS):
    seed_oof = np.zeros((n_train, N_CLASSES))
    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n_train), vi)
        m = fit_meta(OOF_full[tr_idx], y_all[tr_idx], seed=seed)
        seed_oof[vi] = m.predict_proba(OOF_full[vi])
    seed_oof_probs[si] = seed_oof
    m_full = fit_meta(OOF_full, y_all, seed=seed)
    seed_test_probs[si] = m_full.predict_proba(TST_full)

bagged_oof_n76  = seed_oof_probs.mean(axis=0)
bagged_test_n76 = seed_test_probs.mean(axis=0)

fold_scores_n76 = [score_fn(y_all[vi], bagged_oof_n76[vi].argmax(1)) for vi in fval]
cv_n76 = float(np.mean(fold_scores_n76))
sem_n76 = float(np.std(fold_scores_n76, ddof=1) / np.sqrt(len(fold_scores_n76)))
log(f"n76 baseline: cv={cv_n76:.6f}  sem={sem_n76:.6f}")
log(f"n76 per-fold: {fold_scores_n76}")
# Hard assert ~0.970227 (within 2*sem tolerance)
assert abs(cv_n76 - 0.970227) < 0.0005, f"n76 reproduce drift: got {cv_n76:.6f} expected ~0.970227"
log(f"n76 reproduce: PASS (delta={cv_n76-0.970227:+.6f})")

# ─── Forward-add: n76 bank18 + this node's OOF ──────────────────────────────
log("\n=== Forward-add: bank18 + node_0087 OOF ===")

this_oof_logp  = logp(norm(oof_proba))
this_test_logp = logp(norm(test_proba_accum))

OOF_plus  = np.concatenate([OOF_full, this_oof_logp], axis=1)
TST_plus  = np.concatenate([TST_full, this_test_logp], axis=1)
log(f"Augmented feature matrix: {OOF_plus.shape}")

seed_oof_plus  = np.zeros((len(SEEDS), n_train, N_CLASSES))
seed_test_plus = np.zeros((len(SEEDS), n_test, N_CLASSES))

for si, seed in enumerate(SEEDS):
    seed_oof = np.zeros((n_train, N_CLASSES))
    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n_train), vi)
        m = fit_meta(OOF_plus[tr_idx], y_all[tr_idx], seed=seed)
        seed_oof[vi] = m.predict_proba(OOF_plus[vi])
    seed_oof_plus[si] = seed_oof
    m_full = fit_meta(OOF_plus, y_all, seed=seed)
    seed_test_plus[si] = m_full.predict_proba(TST_plus)

bagged_oof_plus = seed_oof_plus.mean(axis=0)

fold_scores_plus = [score_fn(y_all[vi], bagged_oof_plus[vi].argmax(1)) for vi in fval]
cv_plus = float(np.mean(fold_scores_plus))
sem_plus = float(np.std(fold_scores_plus, ddof=1) / np.sqrt(len(fold_scores_plus)))
log(f"bank18+n87 stack: cv={cv_plus:.6f}  sem={sem_plus:.6f}")
log(f"bank18+n87 per-fold: {fold_scores_plus}")

delta_vs_n76     = cv_plus - cv_n76
delta_vs_champ   = cv_plus - 0.970153
beats_champ_2sem = delta_vs_champ > 2 * sem_plus

log(f"\n=== GATE SUMMARY ===")
log(f"Solo BA (node_0087):        {mean_cv:.6f}  sem={sem_cv:.6f}")
log(f"n76 baseline (reproduce):   {cv_n76:.6f}  sem={sem_n76:.6f}")
log(f"Stack+n87:                  {cv_plus:.6f}  sem={sem_plus:.6f}")
log(f"Delta vs n76:               {delta_vs_n76:+.6f}")
log(f"Delta vs champion 0.970153: {delta_vs_champ:+.6f}  (2*sem={2*sem_plus:.6f})")
log(f"Beats champion by >2*sem:   {beats_champ_2sem}")
log(f"Mean err-corr vs bank18:    {np.mean(all_corrs):.4f}")

log("\nDone.")
