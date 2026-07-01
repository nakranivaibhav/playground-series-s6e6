"""node_0132 — improve (gbdt): LightGBM + fs_sedshape on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0030:
  Add fs_sedshape (STATELESS) — z-gated continuum-relative SED-shape features.
  The rest of the pipeline is byte-identical to node_0030.

fs_sedshape features (all STATELESS — pure row-wise, no fit, no target, no cross-row):
  1. bump_excess_fixed (≈z≈0.9 QSO MgII marker):
       bump_excess = mag_r - (0.55*mag_g + 0.45*mag_i)
       (r minus log-flux continuum interpolated from non-line neighbours g,i)
  2. bump_excess_zgated (z-gated form):
       band B = SDSS band nearest to 2800*(1+z) Angstroms
       SDSS λ_eff: u=3651, g=4679, r=6175, i=7494, z=8873
       emit mag_B - continuum_interp(neighbours of B)
       Neighbours: u→(g only, no left; one-sided), g→(u,r), r→(g,i), i→(r,z), z→(i, one-sided)
  3. d4000_proxy (z-gated D4000 break):
       band pair straddling 4000*(1+z): low-z→(u-g), mid-z→(g-r), high-z→(r-i) or (i-z)
       galaxy D4000>1.3, QSO≈1 → positive for galaxies, ≈0 for QSOs

Source: Richards 2001 AJ121 2308 / arXiv:astro-ph/0012449 (quasar λ_eff 3651/4679/6175/7494/8873)
        PAU arXiv:2201.04411, Beck 2016 arXiv:1603.09708 (D4000 break proxy)

LightGBM config (byte-identical to node_0030):
  - class_weight="balanced", n_estimators=2000, lr=0.05, num_leaves=127
  - early stopping 150 rounds, CPU mode, n_jobs=-1

Leakage discipline:
  - fs_sedshape: STATELESS — pure row-wise formula on raw ugriz+z. No fit, no target,
    no cross-row statistics. Identical formula applied to train and test. Safe.
  - KBinsDiscretizer, factorize maps: fit on train-fold only. (fit_in_fold)
  - TargetEncoder: fit on train-fold only (sklearn internal CV=5). (fit_in_fold)
  - Frozen folds.json used throughout; no refitting of folds.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

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
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

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


# ─── SDSS band effective wavelengths (Richards 2001 / astro-ph/0012449) ──────
# u=3651, g=4679, r=6175, i=7494, z=8873  (Angstroms)
_BANDS = ["u", "g", "r", "i", "z"]
_LAMBDA_EFF = np.array([3651.0, 4679.0, 6175.0, 7494.0, 8873.0])  # Angstroms


def _interp_continuum(mag_arr: np.ndarray, band_idx: int) -> np.ndarray:
    """
    Given 5-band magnitude array (shape N×5) and the index of the target band,
    interpolate the continuum at the target band from its two nearest neighbours
    (log-flux interpolation = linear in magnitude space).

    For edge bands (u=0, z=4) where only one neighbour exists, use that single
    neighbour as the continuum (no interpolation — one-sided).

    Returns array of shape (N,) — the continuum magnitude at the target band.
    """
    lam = _LAMBDA_EFF
    b = band_idx

    if b == 0:
        # u: only right neighbour g (index 1)
        return mag_arr[:, 1]
    elif b == 4:
        # z: only left neighbour i (index 3)
        return mag_arr[:, 3]
    else:
        # Linear interpolation in log-wavelength (= linear in mag for power-law SED)
        lam_left = lam[b - 1]
        lam_right = lam[b + 1]
        lam_b = lam[b]
        # Weight for the left band: how close right is to target / span
        w_left = (lam_right - lam_b) / (lam_right - lam_left)
        w_right = 1.0 - w_left
        return w_left * mag_arr[:, b - 1] + w_right * mag_arr[:, b + 1]


def sedshape_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    fs_sedshape — STATELESS z-gated continuum-relative SED-shape features.
    Pure row-wise formula: raw ugriz magnitudes + redshift only.
    No fitting, no target, no cross-row statistics.

    Features added:
      _bump_excess_fixed   : r − (0.55*g + 0.45*i)  [MgII at z≈0.9]
      _bump_excess_zgated  : mag_B − continuum_interp(B), B = nearest band to 2800*(1+z)
      _d4000_proxy         : z-gated D4000 break (galaxy>0, QSO≈0)
    """
    df = df.copy()

    mags = df[["u", "g", "r", "i", "z"]].values.astype("float64")  # N×5
    z = df["redshift"].values.astype("float64")

    # 1. Fixed MgII bump-excess (optimised for z≈0.9 where 2800Å lands in r≈6175Å)
    #    bump_excess = r − (0.55*g + 0.45*i)
    bump_fixed = mags[:, 2] - (0.55 * mags[:, 1] + 0.45 * mags[:, 3])
    df["_bump_excess_fixed"] = bump_fixed.astype("float32")

    # 2. z-gated MgII bump-excess: band B nearest 2800*(1+z) Angstroms
    lambda_mgii_obs = 2800.0 * (1.0 + z)  # observed wavelength of MgII
    # For each row find the band index with minimum |lambda_eff - lambda_mgii_obs|
    diff = np.abs(_LAMBDA_EFF[None, :] - lambda_mgii_obs[:, None])  # N×5
    band_idx_mgii = diff.argmin(axis=1)  # (N,) integers 0..4

    bump_zgated = np.empty(len(df), dtype="float64")
    for bi in range(5):
        mask = band_idx_mgii == bi
        if not mask.any():
            continue
        continuum = _interp_continuum(mags[mask], bi)
        bump_zgated[mask] = mags[mask, bi] - continuum

    df["_bump_excess_zgated"] = bump_zgated.astype("float32")

    # 3. z-gated D4000 break proxy
    #    D4000 = flux ratio just redward vs just blueward of 4000*(1+z) Angstroms
    #    In magnitudes: (mag_blue - mag_red) > 0 for galaxies (flux drops blueward),
    #    ≈ 0 for QSOs (power law), < 0 for some stars.
    #    Band pair straddling 4000*(1+z):
    #      4000*(1+z) < 4679 → z < 0.17  → pair (u, g), index (0, 1)
    #      4679 ≤ 4000*(1+z) < 6175 → 0.17 ≤ z < 0.544 → pair (g, r), index (1, 2)
    #      6175 ≤ 4000*(1+z) < 7494 → 0.544 ≤ z < 0.874 → pair (r, i), index (2, 3)
    #      7494 ≤ 4000*(1+z) < 8873 → 0.874 ≤ z < 1.218 → pair (i, z), index (3, 4)
    #      8873 ≤ 4000*(1+z)         → z ≥ 1.218         → pair (i, z), index (3, 4) [edge]
    lambda_d4000_obs = 4000.0 * (1.0 + z)

    d4000_proxy = np.empty(len(df), dtype="float64")

    # Boundaries between pairs: at the right-band λ_eff values
    # pair (u=0, g=1): λ_break ∈ [−∞, 4679)
    # pair (g=1, r=2): λ_break ∈ [4679, 6175)
    # pair (r=2, i=3): λ_break ∈ [6175, 7494)
    # pair (i=3, z=4): λ_break ∈ [7494, +∞)
    m0 = lambda_d4000_obs < _LAMBDA_EFF[1]                                             # z < 0.170
    m1 = (~m0) & (lambda_d4000_obs < _LAMBDA_EFF[2])                                   # 0.170≤z<0.544
    m2 = (~m0) & (~m1) & (lambda_d4000_obs < _LAMBDA_EFF[3])                           # 0.544≤z<0.874
    m3 = ~(m0 | m1 | m2)                                                               # z≥0.874

    d4000_proxy[m0] = mags[m0, 0] - mags[m0, 1]   # u - g
    d4000_proxy[m1] = mags[m1, 1] - mags[m1, 2]   # g - r
    d4000_proxy[m2] = mags[m2, 2] - mags[m2, 3]   # r - i
    d4000_proxy[m3] = mags[m3, 3] - mags[m3, 4]   # i - z

    df["_d4000_proxy"] = d4000_proxy.astype("float32")

    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.

    Note: columns are encoded as int32 (integer codes, NOT dtype='category').
    LightGBM will treat these as numeric continuous features — safe for LightGBM
    since threshold splits work for ordinal integers. Only a small subset will be
    passed as categorical_feature (spectral_type, galaxy_population).
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

    # Original categorical columns (spectral_type, galaxy_population) — low cardinality
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

    # Integer-floor categorical views of every base numeric — high cardinality, treated as numeric
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

    # Delta quantile bins (100 and 500) — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32")

    # Interaction cross-combos — high cardinality, treated as numeric
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

    # Only low-cardinality BASE_CAT_COLS are flagged for native LightGBM cat treatment
    lgbm_cat_cols = BASE_CAT_COLS[:]
    # All columns produced (including int-floor cats, bins, combos)
    all_new_cols = (
        BASE_CAT_COLS
        + [f"{c}_cat_" for c in BASE_NUM_COLS]
        + [f"delta_{n}_quantile_bin_" for n in [100, 500]]
        + combo_names
    )
    all_new_cols = [c for c in all_new_cols if c in tr.columns]

    return tr, va, te, all_new_cols, combo_names, local_map, lgbm_cat_cols


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


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    """
    Well-tuned LightGBM for Balanced Accuracy (macro per-class recall).
    class_weight="balanced" applies inverse-frequency weighting so the model
    attends equally to GALAXY/QSO/STAR despite the imbalance.
    n_estimators=2000 with early_stopping(150 rounds) provides a generous
    budget; typical convergence is ~400-800 rounds at lr=0.05.
    """
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

# fs_sedshape — STATELESS, applied after base stateless FE
# Needs raw ugriz + redshift columns (still present in X_stateless since stateless_fe does df.copy())
log("Applying fs_sedshape FE (stateless z-gated SED-shape) ...")
X_stateless = sedshape_fe(X_stateless)
X_test_stateless = sedshape_fe(X_test_stateless)

log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")
sedshape_cols = ["_bump_excess_fixed", "_bump_excess_zgated", "_d4000_proxy"]
log(f"  fs_sedshape cols: {sedshape_cols}")

# --- Pre-flight leakage self-check (checks 1+2) ---
assert TARGET not in X_stateless.columns, "LEAK: target in features"
assert IDC not in X_stateless.columns, "LEAK: id in features"
# Single-feature sweep on ≤50k sample for near-perfect correlation (check 3)
_sample = X_stateless.sample(min(50_000, len(X_stateless)), random_state=0)
_ys = pd.factorize(train_raw.loc[_sample.index, TARGET])[0]
for _c in sedshape_cols:
    _x = pd.to_numeric(_sample[_c], errors="coerce").fillna(0)
    if _x.nunique() > 1:
        _corr = abs(float(np.corrcoef(_x.values, _ys)[0, 1]))
        if _corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {_c} corr={_corr:.6f} with target")
log("Pre-flight leakage checks PASSED (target/id absent; sedshape corr < 0.999)")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = None
best_iters = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
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

    if all_cols_final is None:
        all_cols_final = list(X_tr_fold.columns)
        log(f"  n_features={X_tr_fold.shape[1]}  lgbm_cat={lgbm_cat_cols}")

    model = make_lgbm(fold_seed=fold_seed)

    model.fit(
        X_tr_fold, y_tr_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=200),
        ],
        categorical_feature=lgbm_cat_cols,
    )

    best_iter = model.best_iteration_
    best_iters.append(best_iter)
    log(f"  best_iteration={best_iter}")

    # OOF probabilities
    val_proba = model.predict_proba(X_val_fold)
    oof_proba[val_idx] = val_proba.astype("float32")

    # Test predictions — average across folds
    test_proba_fold = model.predict_proba(X_te_fold)
    test_proba_accum += test_proba_fold.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

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
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
