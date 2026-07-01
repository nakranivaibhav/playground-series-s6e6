"""node_0130 — improve (gbdt): LightGBM on fs_realmlp_fe + fs_tmplchi2.

THE ONE ATOMIC CHANGE vs node_0030:
  Add fs_tmplchi2 (STATELESS) — per-class SED template-fit chi-squared features,
  gated on the known redshift z. For each row we fit 5-band ugriz fluxes to
  3 analytic template families (STAR=blackbody, QSO=power-law, GALAXY=4000A-break),
  each redshifted to the row's redshift z, using closed-form amplitude scaling,
  and emit chi2_{gal,qso,star}, their pairwise diffs, log1p-normalized chi2s,
  the argmin template class, and a softmax template posterior.

  Templates are fully analytic / pure-numpy — NO speclite, NO external downloads.
  - STAR: Planck blackbody grid over T in [3000..40000] K sampled at 40 temps.
    We find the best-fit T per row by minimizing chi2 over the grid.
  - QSO: power-law f_lambda ∝ lambda^-1.5 (i.e. f_nu ∝ nu^-0.5), plus a broad
    Gaussian bump representing MgII at rest 2800 Å (sigma ~200 Å, amp ~0.5).
  - GALAXY: 4000Å-break model: red continuum f_lambda ∝ lambda^-1 with a step
    that reduces flux ~2.5x below rest 4000 Å.
  SDSS band effective wavelengths (Å): u=3551, g=4686, r=6166, i=7480, z=8932.
  Per-band sigma set to constant 1 for all rows (arbitrary; only relative fits matter).

  Closed-form: alpha* = sum(f*T/sigma^2) / sum(T^2/sigma^2), chi2 = sum((f-alpha*T)^2/sigma^2).
  fs_tmplchi2 is STATELESS: identical row-wise transform on train and test; no .fit, no target.

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

# ─── fs_tmplchi2 — Analytic SED template chi2 (STATELESS) ───────────────────
# SDSS effective wavelengths in Angstroms
BAND_LAMBDA_AA = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0], dtype=np.float64)  # u g r i z

# Physical constants for Planck blackbody
H_CGS = 6.626e-27    # erg·s
C_CGS = 2.998e10     # cm/s
K_CGS = 1.381e-16    # erg/K
C_AA = C_CGS * 1e8   # cm/s → Å/s

# Blackbody star temperature grid
STAR_TEMPS = np.linspace(3000.0, 40000.0, 40, dtype=np.float64)


def planck_flambda(lam_aa: np.ndarray, T: float) -> np.ndarray:
    """Planck f_lambda (relative, unnormalized) at wavelengths lam_aa (Å) and temperature T (K)."""
    lam_cm = lam_aa * 1e-8  # Å → cm
    x = H_CGS * C_AA / (lam_aa * K_CGS * T)  # dimensionless
    x = np.clip(x, 1e-10, 700.0)
    return lam_cm**(-5) / (np.exp(x) - 1.0)


def qso_template(lam_aa: np.ndarray) -> np.ndarray:
    """
    QSO power-law: f_lambda ∝ lambda^-1.5, plus broad MgII bump at rest 2800 Å (sigma=200 Å, amp=0.5).
    """
    fl = lam_aa ** (-1.5)
    # Broad MgII bump
    mgii_center = 2800.0
    mgii_sigma = 200.0
    bump = 0.5 * np.exp(-0.5 * ((lam_aa - mgii_center) / mgii_sigma) ** 2)
    fl = fl + bump * (mgii_center ** (-1.5))  # scale bump to f_lambda at 2800 Å
    return fl


def galaxy_template(lam_aa: np.ndarray) -> np.ndarray:
    """
    Galaxy 4000Å-break model: red continuum f_lambda ∝ lambda^-1, flux drops 2.5x below 4000 Å.
    """
    fl = lam_aa ** (-1.0)
    # Apply 4000 Å break: reduce by factor 2.5 below rest 4000 Å
    break_mask = lam_aa < 4000.0
    fl = fl.copy()
    fl[break_mask] /= 2.5
    return fl


def closed_form_chi2(f_obs: np.ndarray, T: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Closed-form amplitude and chi2.
    f_obs: (n_rows, n_bands)  — observed flux values
    T:     (n_bands,)         — template SED evaluated at each band
    sigma: scalar per-band noise (uniform)
    Returns chi2 array shape (n_rows,)
    """
    inv_s2 = 1.0 / (sigma ** 2)
    num = np.sum(f_obs * T[None, :] * inv_s2, axis=1)   # (n_rows,)
    denom = np.sum(T ** 2 * inv_s2)                       # scalar
    alpha = num / (denom + 1e-30)                          # (n_rows,)
    residual = f_obs - alpha[:, None] * T[None, :]         # (n_rows, n_bands)
    chi2 = np.sum((residual / sigma) ** 2, axis=1)         # (n_rows,)
    return chi2


def compute_tmplchi2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    STATELESS fs_tmplchi2: per-class SED template-fit chi2, gated on redshift z.
    Emits 12 features: chi2_{gal,qso,star}, log1p_chi2_{gal,qso,star},
    diff_star_gal, diff_qso_gal, diff_qso_star, argmin_template,
    softmax_prob_{gal,qso,star}.

    Uses analytic templates (no speclite, no downloads):
    - STAR = best-fit blackbody over T grid [3000..40000] K (40 steps)
    - QSO  = power-law lambda^-1.5 + broad MgII bump at 2800 Å
    - GALAXY = 4000Å-break: f_lambda ∝ lambda^-1, /2.5 below 4000 Å

    Magnitudes → relative flux: f_band = 10^(-0.4 * mag), constant sigma=1.
    Template evaluated at rest-frame wavelength: lambda_rest = lambda_obs / (1 + z).
    """
    mags = df[["u", "g", "r", "i", "z"]].values.astype(np.float64)  # (n, 5)
    redshift = df["redshift"].values.astype(np.float64)               # (n,)

    # Convert magnitudes to relative flux
    # f = 10^(-0.4 * mag), clipped for robustness
    f_obs = 10.0 ** (-0.4 * np.clip(mags, 10.0, 35.0))  # (n, 5)

    n_rows = len(df)

    # Rest-frame wavelength per row per band: lambda_rest = lambda_obs / (1 + z)
    # Shape: (n_rows, 5)
    z_clip = np.clip(redshift, -0.5, 10.0)  # physical range guard
    lam_rest = BAND_LAMBDA_AA[None, :] / (1.0 + z_clip[:, None])  # (n, 5)

    # ─── GALAXY chi2 ──────────────────────────────────────────────────────
    # Evaluate galaxy template at each row's rest-frame wavelengths
    # lam_rest is (n, 5) — we need to call galaxy_template row-by-row? No:
    # galaxy_template is vectorizable: f_lambda ∝ lam^-1 with step at 4000 Å
    gal_T = lam_rest ** (-1.0)  # (n, 5)
    below_break = (lam_rest < 4000.0)
    gal_T = gal_T.copy()
    gal_T[below_break] /= 2.5
    # Closed-form chi2 with row-varying templates: vectorize
    # alpha_gal[i] = sum_b(f_obs[i,b] * gal_T[i,b]) / sum_b(gal_T[i,b]^2)
    # (sigma=1 throughout)
    num_gal = np.sum(f_obs * gal_T, axis=1)       # (n,)
    denom_gal = np.sum(gal_T ** 2, axis=1)         # (n,)
    alpha_gal = num_gal / (denom_gal + 1e-30)      # (n,)
    res_gal = f_obs - alpha_gal[:, None] * gal_T   # (n, 5)
    chi2_gal = np.sum(res_gal ** 2, axis=1)        # (n,)

    # ─── QSO chi2 ─────────────────────────────────────────────────────────
    qso_T = lam_rest ** (-1.5)  # power-law part (n, 5)
    mgii_center = 2800.0
    mgii_sigma = 200.0
    bump = 0.5 * np.exp(-0.5 * ((lam_rest - mgii_center) / mgii_sigma) ** 2)
    qso_T = qso_T + bump * (mgii_center ** (-1.5))  # (n, 5)
    num_qso = np.sum(f_obs * qso_T, axis=1)
    denom_qso = np.sum(qso_T ** 2, axis=1)
    alpha_qso = num_qso / (denom_qso + 1e-30)
    res_qso = f_obs - alpha_qso[:, None] * qso_T
    chi2_qso = np.sum(res_qso ** 2, axis=1)

    # ─── STAR chi2 ────────────────────────────────────────────────────────
    # Minimize over blackbody temperature grid (40 steps)
    # Stars are at z~0, so rest frame ≈ observed frame (but we use actual lam_rest)
    # Shape: (n, 40) — chi2 for each temperature
    # Precompute planck at each T for each (row, band): (n, 5, 40)
    # But that's n_rows * 5 * 40 = potentially large; use loop over T grid

    # Precompute: for each temperature, evaluate Planck at lam_rest for all rows
    # lam_rest: (n, 5); T: scalar → Planck returns (n, 5)
    star_chi2_grid = np.full((n_rows, len(STAR_TEMPS)), np.inf, dtype=np.float64)
    for ti, T in enumerate(STAR_TEMPS):
        # Planck at rest-frame wavelengths for each row
        lam_cm = lam_rest * 1e-8  # (n, 5) in cm
        x = H_CGS * C_AA / (lam_rest * K_CGS * T)  # (n, 5) dimensionless
        x_clip = np.clip(x, 1e-10, 700.0)
        bb_T = lam_cm ** (-5) / (np.exp(x_clip) - 1.0)  # (n, 5)
        # Closed-form alpha and chi2
        num_s = np.sum(f_obs * bb_T, axis=1)
        denom_s = np.sum(bb_T ** 2, axis=1)
        alpha_s = num_s / (denom_s + 1e-30)
        res_s = f_obs - alpha_s[:, None] * bb_T
        star_chi2_grid[:, ti] = np.sum(res_s ** 2, axis=1)

    chi2_star = np.min(star_chi2_grid, axis=1)  # (n,)

    # ─── Derived features ─────────────────────────────────────────────────
    diff_star_gal = chi2_star - chi2_gal
    diff_qso_gal = chi2_qso - chi2_gal
    diff_qso_star = chi2_qso - chi2_star

    # argmin class (0=galaxy, 1=qso, 2=star)
    chi2_stack = np.stack([chi2_gal, chi2_qso, chi2_star], axis=1)  # (n, 3)
    argmin_tmpl = np.argmin(chi2_stack, axis=1).astype(np.float32)  # (n,)

    # Softmax template posterior (higher chi2 = less likely)
    # Use negative chi2 as logit
    neg_chi2 = -chi2_stack
    neg_chi2 -= neg_chi2.max(axis=1, keepdims=True)  # numerical stability
    exp_nc = np.exp(np.clip(neg_chi2, -50, 0))
    softmax_tmpl = exp_nc / (exp_nc.sum(axis=1, keepdims=True) + 1e-30)  # (n, 3)

    # log1p normalize chi2 values for GBDT
    log1p_chi2_gal = np.log1p(np.clip(chi2_gal, 0, None)).astype(np.float32)
    log1p_chi2_qso = np.log1p(np.clip(chi2_qso, 0, None)).astype(np.float32)
    log1p_chi2_star = np.log1p(np.clip(chi2_star, 0, None)).astype(np.float32)

    out = pd.DataFrame({
        "_chi2_gal": chi2_gal.astype(np.float32),
        "_chi2_qso": chi2_qso.astype(np.float32),
        "_chi2_star": chi2_star.astype(np.float32),
        "_log1p_chi2_gal": log1p_chi2_gal,
        "_log1p_chi2_qso": log1p_chi2_qso,
        "_log1p_chi2_star": log1p_chi2_star,
        "_diff_star_gal": diff_star_gal.astype(np.float32),
        "_diff_qso_gal": diff_qso_gal.astype(np.float32),
        "_diff_qso_star": diff_qso_star.astype(np.float32),
        "_argmin_template": argmin_tmpl,
        "_softmax_gal": softmax_tmpl[:, 0].astype(np.float32),
        "_softmax_qso": softmax_tmpl[:, 1].astype(np.float32),
        "_softmax_star": softmax_tmpl[:, 2].astype(np.float32),
    }, index=df.index)

    return out


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless feature engineering — safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    Includes fs_realmlp_fe features AND fs_tmplchi2 features.
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

    # fs_tmplchi2: per-class SED template chi2 features (STATELESS)
    tmpl_feats = compute_tmplchi2_features(df)
    for col in tmpl_feats.columns:
        df[col] = tmpl_feats[col]

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

# ─── Leakage pre-check 1-2: target/id not in features ────────────────────────
assert TARGET not in BASE_NUM_COLS and TARGET not in BASE_CAT_COLS, "TARGET in features!"
assert IDC not in BASE_NUM_COLS and IDC not in BASE_CAT_COLS, "IDC in features!"

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE (includes fs_tmplchi2 template chi2) ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Leakage pre-check 3: single-feature↔target sweep on a sample ────────────
log("Pre-check 3: single-feature↔target corr sweep ...")
sample_n = min(50000, len(X_stateless))
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(X_stateless), size=sample_n, replace=False)
X_sample = X_stateless.iloc[sample_idx]
y_sample = y_all[sample_idx]
tmpl_cols = [c for c in X_stateless.columns if c.startswith("_chi2") or c.startswith("_log1p") or c.startswith("_diff") or c.startswith("_argmin") or c.startswith("_softmax")]
log(f"  Checking {len(tmpl_cols)} fs_tmplchi2 columns for corr with target ...")
for c in tmpl_cols:
    x = pd.to_numeric(X_sample[c], errors="coerce")
    if x.nunique() > 1:
        corr = abs(np.corrcoef(x.fillna(x.mean()).values, y_sample)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {c} corr={corr:.6f} vs target — STOP!")
log("  No near-perfect single-feature correlations found — OK")

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
        tmpl_in_cols = [c for c in all_cols_final if "_chi2" in c or "_softmax" in c or "_argmin" in c or "_diff_" in c and "log1p" not in c]
        log(f"  fs_tmplchi2 features in model: {tmpl_in_cols}")

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
        # Cheap-kill gate: fold-0 < 0.965 → abort
        if per_fold_scores[0] < 0.965:
            log(f"CHEAP-KILL TRIGGERED: fold0={per_fold_scores[0]:.6f} < 0.965 threshold")
            print(f"CHEAP_KILL: fold0={per_fold_scores[0]:.6f}", flush=True)
            sys.exit(1)

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
