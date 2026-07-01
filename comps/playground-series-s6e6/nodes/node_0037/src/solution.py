"""node_0037 — draft (linear): multinomial LogisticRegression on fs_realmlp_fe.

Built on: root (new draft, cross-family diversity).
FE pipeline is byte-identical to node_0028 (fs_realmlp_fe):
  - stateless FE: color pairs, redshift ratios, log1p_redshift, mag stats
  - fit_in_fold: KBinsDiscretizer (delta bins), factorize maps,
    TargetEncoder on combo cats, StandardScaler for numerics

THE ONE ATOMIC CHANGE vs node_0028:
  Replace the hand-rolled RealMLP with a multinomial LogisticRegression
  (sklearn, solver='saga', multi_class='multinomial', class_weight='balanced',
  C=0.1, max_iter=1000). Categorical features are one-hot encoded (via
  pd.get_dummies) to make them compatible with a linear model. Numerical
  features are StandardScaler-transformed fit-in-fold.

Purpose: maximally de-correlated linear base for stack diversity.
  Even at modest solo strength, orthogonal errors give the meta-stacker
  new exploitable signal.

Leakage discipline:
  - Stateless FE computed once on full dataframes (no target, no cross-row stats).
  - KBinsDiscretizer, factorize maps: fit on train-fold only.
  - TargetEncoder: fit on train-fold only.
  - StandardScaler: fit on train-fold numerical features only.
  - One-hot encoding derived from train-fold categories only (unknown test cats -> 0).
  - Frozen folds.json used throughout.

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, TargetEncoder

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
    Identical to node_0028.
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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.
    Identical to node_0028.
    """
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

    # Original categorical columns (spectral_type, galaxy_population)
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    # Integer-floor categorical views of every base numeric
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_orig in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    # Delta quantile bins (100 and 500) — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

    # Interaction cross-combos
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
    """
    TargetEncoder fit on train fold only (fit_in_fold), transform val and test.
    Returns modified copies and the list of new TE column names.
    Identical to node_0028.
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


def prepare_for_linear(X_tr: pd.DataFrame, X_val: pd.DataFrame, X_te: pd.DataFrame,
                       cat_cols: list, fold_seed: int):
    """
    Convert the feature matrix for a linear model:
      1. For low-cardinality categoricals (<=20 unique in train): one-hot encode
         using train-fold categories only. Unknown cats in val/test -> 0.
      2. For high-cardinality categoricals (>20 unique): treat as ordinal numeric
         (they are integer codes / bin indices of continuous features — ordinal
         interpretation is valid and avoids an explosion of OHE columns with
         potentially sparse coverage, which would make saga very slow).
      3. StandardScaler fit on ALL resulting numeric columns (train fold only).
    Returns (X_tr_arr, X_val_arr, X_te_arr, feature_names).
    """
    OHE_THRESH = 20  # OHE only if n_unique <= this threshold
    base_num_cols = [c for c in X_tr.columns if c not in cat_cols]

    ohe_cat_cols = []   # will be one-hot encoded
    ord_cat_cols = []   # will be treated as ordinal numeric

    for col in cat_cols:
        n_unique = X_tr[col].nunique()
        if n_unique <= OHE_THRESH:
            ohe_cat_cols.append(col)
        else:
            ord_cat_cols.append(col)

    # --- One-hot encode low-cardinality categoricals (train cats only) ---
    oh_tr_parts = []
    oh_va_parts = []
    oh_te_parts = []
    oh_col_names = []

    for col in ohe_cat_cols:
        dummies_tr = pd.get_dummies(
            X_tr[col].astype(str), prefix=col, drop_first=False, dtype="float32"
        )
        oh_tr_parts.append(dummies_tr)
        oh_col_names.extend(dummies_tr.columns.tolist())

        dummies_va = pd.get_dummies(
            X_val[col].astype(str), prefix=col, drop_first=False, dtype="float32"
        ).reindex(columns=dummies_tr.columns, fill_value=0.0)
        oh_va_parts.append(dummies_va)

        dummies_te = pd.get_dummies(
            X_te[col].astype(str), prefix=col, drop_first=False, dtype="float32"
        ).reindex(columns=dummies_tr.columns, fill_value=0.0)
        oh_te_parts.append(dummies_te)

    oh_tr = np.hstack([p.values for p in oh_tr_parts]) if oh_tr_parts else np.zeros((len(X_tr), 0), dtype="float32")
    oh_va = np.hstack([p.values for p in oh_va_parts]) if oh_va_parts else np.zeros((len(X_val), 0), dtype="float32")
    oh_te = np.hstack([p.values for p in oh_te_parts]) if oh_te_parts else np.zeros((len(X_te), 0), dtype="float32")

    # --- Combine: base numerics + ordinal-treated high-cardinality cats ---
    all_num_cols = base_num_cols + ord_cat_cols
    X_tr_num = np.hstack([
        X_tr[base_num_cols].values.astype("float32"),
        X_tr[ord_cat_cols].values.astype("float32") if ord_cat_cols else np.zeros((len(X_tr), 0), dtype="float32"),
    ])
    X_va_num = np.hstack([
        X_val[base_num_cols].values.astype("float32"),
        X_val[ord_cat_cols].values.astype("float32") if ord_cat_cols else np.zeros((len(X_val), 0), dtype="float32"),
    ])
    X_te_num = np.hstack([
        X_te[base_num_cols].values.astype("float32"),
        X_te[ord_cat_cols].values.astype("float32") if ord_cat_cols else np.zeros((len(X_te), 0), dtype="float32"),
    ])

    # --- StandardScaler fit on train-fold numerics + ordinals only ---
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_tr_num)
    X_va_num = scaler.transform(X_va_num)
    X_te_num = scaler.transform(X_te_num)

    # Concatenate: scaled numerics first, then one-hot categoricals
    X_tr_arr = np.hstack([X_tr_num, oh_tr]).astype("float32")
    X_va_arr = np.hstack([X_va_num, oh_va]).astype("float32")
    X_te_arr = np.hstack([X_te_num, oh_te]).astype("float32")

    feature_names = all_num_cols + oh_col_names
    return X_tr_arr, X_va_arr, X_te_arr, feature_names


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
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype="float32")
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype="float32")
per_fold_scores = []
feature_names_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

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

    # Sort columns consistently (same as node_0028)
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    if feature_names_final is None:
        log(f"  n_features_pre_ohe={X_tr_fold.shape[1]}  n_cat={len(cat_cols_sorted)}")

    # --- THE ONE ATOMIC CHANGE: prepare for linear model (StandardScaler + OHE) ---
    X_tr_arr, X_va_arr, X_te_arr, feat_names = prepare_for_linear(
        X_tr_fold, X_val_fold, X_te_fold, cat_cols_sorted, fold_seed
    )

    if feature_names_final is None:
        feature_names_final = feat_names
        log(f"  n_features_linear={X_tr_arr.shape[1]}")

    # Fit LogisticRegression — multinomial, class_weight='balanced', C=0.1
    model = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        class_weight="balanced",
        C=0.1,
        max_iter=1000,
        random_state=fold_seed,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_tr_arr, y_tr_fold)

    # OOF probabilities
    val_proba = model.predict_proba(X_va_arr).astype("float32")
    # Reorder columns to GALAXY=0, QSO=1, STAR=2 (model.classes_ may differ)
    col_order = [list(model.classes_).index(c) for c in range(N_CLASSES)]
    val_proba = val_proba[:, col_order]
    oof_proba[val_idx] = val_proba

    # Test predictions — average across folds
    test_proba_fold = model.predict_proba(X_te_arr).astype("float32")[:, col_order]
    test_proba_accum += test_proba_fold / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold, X_tr_arr, X_va_arr, X_te_arr
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
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
(NODE_SRC / "features.txt").write_text("\n".join(feature_names_final) + "\n")
log(f"Wrote features.txt ({len(feature_names_final)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
