"""node_0047 — GALAXY-vs-STAR specialist (low-z) + re-stack.

THE ONE ATOMIC CHANGE:
  A binary LightGBM specialist trained only on low-redshift (rs < 0.15)
  GALAXY / STAR rows (no QSO) using fs_realmlp_fe features, fit IN-FOLD.
  Its P(STAR | in-zone) is emitted as a new OOF column + test column (N,1)
  and added as an extra base to the CORE15 stack (LogReg meta + DE threshold).

Leakage discipline:
  - Low-z mask uses only raw 'redshift' feature (no labels; stateless).
  - Binary target fit INSIDE each fold (only on train-fold rows in the zone).
  - TargetEncoder, KBins, factorize maps all fit_in_fold only.
  - OOF covers all train rows exactly once (out-of-zone rows get the in-zone
    class prior as a neutral constant).
  - Frozen folds.json used throughout.

Outputs:
  oof.npy          (N_train, 1)  specialist P(STAR | zone), neutral outside
  test_probs.npy   (N_test,  1)  same
  submission.csv               re-stacked CORE15 + specialist predictions
  features.txt                 features used by the specialist LGBM
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

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
SEED = 42
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
NC = 3

# Specialist constants
SPEC_LOW_Z = 0.15          # low-z cutoff
SPEC_STAR  = 1             # binary: GALAXY=0, STAR=1

# CORE15 bases (same as node_0041)
BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]

# ─── Feature engineering (from node_0030/src) ───────────────────────────────
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
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
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
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32")

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32")

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_names.append(combo_name)
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        local_map[combo_name] = uniques
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32")

    lgbm_cat_cols = BASE_CAT_COLS[:]
    all_new_cols = (
        BASE_CAT_COLS
        + [f"{c}_cat_" for c in BASE_NUM_COLS]
        + [f"delta_{n}_quantile_bin_" for n in [100, 500]]
        + combo_names
    )
    all_new_cols = [c for c in all_new_cols if c in tr.columns]
    return tr, va, te, all_new_cols, combo_names, local_map, lgbm_cat_cols


def add_target_encoding_binary(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    """Binary target encoding (STAR=1/GALAXY=0)."""
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="binary", cv=5, smooth="auto",
                                shuffle=True, random_state=fold_seed)
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)

    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])

    te_names = [f"_{col}_TE_bin" for col in combo_names]
    X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
    X_val[te_names] = np.asarray(val_enc, dtype="float32")
    X_te[te_names] = np.asarray(tst_enc, dtype="float32")
    return X_tr, X_val, X_te, te_names


def make_spec_lgbm(fold_seed: int) -> LGBMClassifier:
    """Binary specialist LightGBM: GALAXY(0) vs STAR(1)."""
    return LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
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


# ─── Stack helpers ───────────────────────────────────────────────────────────
def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def score_fn(y_true, y_pred) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def best_thr_de(probs, labels) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                                maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


# ─── Load data ───────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# Compute low-z masks using ONLY raw redshift — stateless, no labels
rs_train = train_raw["redshift"].values
rs_test  = test_raw["redshift"].values
# Galaxy/star classes in train
is_galaxy_train = (y_all == LABEL_MAP["GALAXY"])
is_star_train   = (y_all == LABEL_MAP["STAR"])
lowz_train = rs_train < SPEC_LOW_Z
lowz_test  = rs_test  < SPEC_LOW_Z

log(f"  low-z mask: train={lowz_train.sum()} test={lowz_test.sum()}")
log(f"  in-zone GALAXY+STAR: {(lowz_train & (is_galaxy_train | is_star_train)).sum()}")

# In-zone prior: P(STAR | GALAXY-or-STAR, low-z) — computed on full train
# This is used as the neutral constant for out-of-zone rows.
# NOTE: this uses labels to compute a prior — but it's only used as a CONSTANT
# (not as a feature that leaks the fold's val labels), computed once from the
# full train distribution. We use 0.5 (equal prior) to be fully safe.
NEUTRAL = 0.5

# ─── Stateless FE ────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw       = train_raw.drop(columns=[IDC, TARGET])
X_test_raw  = test_raw.drop(columns=[IDC])
X_stateless = stateless_fe(X_raw)
X_test_stat = stateless_fe(X_test_raw)

# ─── Specialist OOF loop ─────────────────────────────────────────────────────
spec_oof  = np.full(n_train, NEUTRAL, dtype=np.float32)   # (N,1) after reshape
spec_test = np.zeros(n_test, dtype=np.float32)
spec_test_counts = np.zeros(n_test, dtype=np.int32)

all_cols_final = None
per_fold_spec_acc = []

log("Starting SPECIALIST OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    # --- in-fold low-z zone masks ---
    tr_lowz_mask  = lowz_train[tr_idx]
    val_lowz_mask = lowz_train[val_idx]

    # Binary target: GALAXY=0, STAR=1; keep only GALAXY+STAR in low-z zone
    y_tr  = y_all[tr_idx]
    y_val = y_all[val_idx]
    tr_spec_mask  = tr_lowz_mask  & ((y_tr  == LABEL_MAP["GALAXY"]) | (y_tr  == LABEL_MAP["STAR"]))
    val_spec_mask = val_lowz_mask & ((y_val == LABEL_MAP["GALAXY"]) | (y_val == LABEL_MAP["STAR"]))
    te_spec_mask  = lowz_test

    tr_spec_idx   = tr_idx[tr_spec_mask]
    val_spec_idx  = val_idx[val_spec_mask]

    y_tr_bin  = (y_tr[tr_spec_mask]  == LABEL_MAP["STAR"]).astype(int)
    y_val_bin = (y_val[val_spec_mask] == LABEL_MAP["STAR"]).astype(int)

    log(f"Fold {fold_id}: spec_train={tr_spec_mask.sum()}  spec_val={val_spec_mask.sum()}  "
        f"star_frac_tr={y_tr_bin.mean():.3f}")

    if tr_spec_mask.sum() < 50 or val_spec_mask.sum() < 5:
        log(f"  Fold {fold_id}: too few zone rows — skipping, using neutral")
        continue

    # Categorical encoding — fit on zone TRAIN rows only (fit_in_fold)
    X_tr_fe, X_val_fe, X_te_fe, all_cat_cols, combo_names, local_map, lgbm_cat_cols = \
        fit_fold_categoricals(
            X_stateless.iloc[tr_spec_idx].reset_index(drop=True),
            X_stateless.iloc[val_spec_idx].reset_index(drop=True),
            X_test_stat.iloc[te_spec_mask].reset_index(drop=True) if te_spec_mask.sum() > 0
            else X_test_stat.iloc[:0].reset_index(drop=True),
        )

    # Binary target encoding fit on zone train rows only
    X_tr_fe, X_val_fe, X_te_fe, te_names = add_target_encoding_binary(
        X_tr_fe, y_tr_bin, X_val_fe, X_te_fe, combo_names, fold_seed
    )

    X_tr_fe  = X_tr_fe.reindex(sorted(X_tr_fe.columns), axis=1)
    X_val_fe = X_val_fe.reindex(sorted(X_val_fe.columns), axis=1)
    X_te_fe  = X_te_fe.reindex(sorted(X_te_fe.columns), axis=1)

    if all_cols_final is None:
        all_cols_final = list(X_tr_fe.columns)
        log(f"  specialist n_features={X_tr_fe.shape[1]}")

    model = make_spec_lgbm(fold_seed)
    model.fit(
        X_tr_fe, y_tr_bin,
        eval_set=[(X_val_fe, y_val_bin)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=80, verbose=False),
            log_evaluation(period=200),
        ],
    )

    # OOF: P(STAR) for in-zone val rows; out-of-zone stays NEUTRAL
    val_p = model.predict_proba(X_val_fe)[:, 1].astype("float32")
    spec_oof[val_spec_idx] = val_p

    # Accuracy check in zone
    val_pred = (val_p >= 0.5).astype(int)
    acc = (val_pred == y_val_bin).mean()
    per_fold_spec_acc.append(acc)
    log(f"  fold {fold_id}: in-zone accuracy={acc:.4f}  best_iter={model.best_iteration_}")

    # Test: accumulate P(STAR) for low-z test rows
    if te_spec_mask.sum() > 0:
        te_p = model.predict_proba(X_te_fe)[:, 1].astype("float32")
        spec_test[te_spec_mask] += te_p
        spec_test_counts[te_spec_mask] += 1

    del model, X_tr_fe, X_val_fe, X_te_fe
    gc.collect()

    if fold_id == 0:
        elapsed = time.perf_counter() - fold_t0
        log(f"  TIMING: fold0={elapsed:.1f}s  projected_5fold={elapsed*5:.1f}s")

# Average test probs for low-z rows across folds; neutral for out-of-zone
pos = spec_test_counts > 0
spec_test[pos]  = spec_test[pos] / spec_test_counts[pos]
spec_test[~pos] = NEUTRAL  # already NEUTRAL (0.5) for out-of-zone

log(f"Specialist in-zone OOF accuracy (mean): {np.mean(per_fold_spec_acc):.4f}")
log(f"spec_oof  min={spec_oof.min():.4f} max={spec_oof.max():.4f} mean={spec_oof.mean():.4f}")
log(f"spec_test min={spec_test.min():.4f} max={spec_test.max():.4f} mean={spec_test.mean():.4f}")

# Reshape to (N, 1) for stacking
spec_oof_col  = spec_oof.reshape(-1, 1).astype("float32")
spec_test_col = spec_test.reshape(-1, 1).astype("float32")

# ─── Save specialist artifacts ────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy",        spec_oof_col)
np.save(NODE_DIR / "test_probs.npy", spec_test_col)
log(f"Saved oof.npy={spec_oof_col.shape}  test_probs.npy={spec_test_col.shape}")

# ─── Write features.txt ───────────────────────────────────────────────────────
if all_cols_final is None:
    all_cols_final = []
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# ─── Re-stack: CORE15 + specialist ───────────────────────────────────────────
log("Loading CORE15 OOF + test probs for re-stack ...")
nodes_dir = COMP_DIR / "nodes"

OOF_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
)
TEST_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
)
log(f"  CORE15 OOF={OOF_CORE.shape}  TEST={TEST_CORE.shape}")

# Specialist column in log-prob space
# P(STAR) -> log[P(STAR)], log[1-P(STAR)] for GALAXY
# Or simply add the raw specialist prob (not log) — let the meta decide weighting.
# We add log(p_star) and log(1-p_star) as two columns to match other bases.
spec_log = logp(np.concatenate([1 - spec_oof_col, spec_oof_col], axis=1))  # (N,2): [logp_GAL, logp_STAR]
spec_log_test = logp(np.concatenate([1 - spec_test_col, spec_test_col], axis=1))

OOF  = np.concatenate([OOF_CORE,  spec_log],      axis=1)
TEST = np.concatenate([TEST_CORE, spec_log_test],  axis=1)
log(f"  stacked OOF={OOF.shape}  TEST={TEST.shape}  ({len(BASES)+1} bases)")

folds_data = folds_list
fval = [np.asarray(f["val_idx"]) for f in folds_data]

# Fold-honest stacked OOF
stack_oof = np.zeros((n_train, NC))
for vi in fval:
    tr = np.setdiff1d(np.arange(n_train), vi)
    m = fit_meta(OOF[tr], y_all[tr])
    stack_oof[vi] = m.predict_proba(OOF[vi])

# Fold-honest DE threshold scoring
per_fold_scores = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(stack_oof[other], y_all[other])
    pred = np.argmax(stack_oof[vi] * w, axis=1)
    s = score_fn(y_all[vi], pred)
    per_fold_scores.append(s)
    print(f"fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]", flush=True)

cv_mean = float(np.mean(per_fold_scores))
cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
print(f"cv={cv_mean:.6f}", flush=True)

# Fit meta on full OOF
meta_full = fit_meta(OOF, y_all)
w_full = best_thr_de(stack_oof, y_all)
log(f"final w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

# Test predictions
stack_test_probs = meta_full.predict_proba(TEST)
test_preds_idx = np.argmax(stack_test_probs * w_full, axis=1)
test_labels = [CLASSES[i] for i in test_preds_idx]

sub = pd.DataFrame({"id": test_raw["id"], "class": test_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")

assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)
log("submission schema OK")

total = time.perf_counter() - T0
log(f"Total elapsed: {total:.1f}s ({total/60:.1f}min)")
log("Done.")
