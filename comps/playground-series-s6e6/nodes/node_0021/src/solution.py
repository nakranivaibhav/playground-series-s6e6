"""node_0021 — draft (nn): RealMLP base via pytabkit.

Built on: node_0009 harness (fs_research feature matrix, frozen folds.json loop,
fold-honest OOF + test_probs interface). Data loading, feature engineering, and
fold scaffolding are byte-identical to node_0009.

THE ONE ATOMIC CHANGE: replace TabM (tabm library) with RealMLP_TD_Classifier from
pytabkit. RealMLP is a strongly regularized MLP with tuned defaults that reaches
GBDT-level accuracy solo while being structurally de-correlated from both TabM and
the tree models — a fresh, additive stack column.

Leakage discipline (fit_in_fold):
  - Standardization of the 22 continuous features: mean/std from the TRAIN FOLD only,
    applied to the held val fold. At final refit, full-train stats used.
  - RealMLP_TD_Classifier is instantiated fresh and .fit() called INSIDE each fold on
    train rows only. Its TD tuned defaults include its own internal normalization, but
    we additionally standardize CONT features ourselves (consistent with node_0009
    interface) before passing to pytabkit.

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
for p in (str(REPO_ROOT), str(COMP_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from clean import (  # noqa: E402
    cast_categoricals, add_color_features, add_extended_colors,
    add_redshift_features, add_qso_colorbox, add_galactic_coords, feature_columns,
)

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}

CONT = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
        "u_g", "g_r", "r_i", "i_z", "u_z", "u_r", "u_i", "g_i", "r_z",
        "c_ug_gr", "c_gr_ri", "log1p_redshift", "gal_l", "gal_b"]   # 22, standardized
FLAGS = ["is_star_z", "is_highz", "qso_box", "uv_excess"]           # 4, numeric 0/1
NUMF = CONT + FLAGS                                                  # 26 → x_num
CATF = ["spectral_type", "galaxy_population"]                       # 2 categoricals

SEED = 42
N_CONT = len(CONT)

np.random.seed(SEED)

print("Importing pytabkit RealMLP_TD_Classifier …", flush=True)
from pytabkit import RealMLP_TD_Classifier  # noqa: E402
print("  pytabkit imported OK", flush=True)


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


print("Loading + engineering …", flush=True)
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

Xnum_all = train[NUMF].to_numpy(np.float32)
Xnum_te = test[NUMF].to_numpy(np.float32)

# Categorical codes (stable, leak-safe — fixed categories from cast_categoricals)
cat_cards = [int(train[c].cat.categories.size) for c in CATF]
Xcat_all = np.stack([train[c].cat.codes.to_numpy() for c in CATF], axis=1).astype(np.int64)
Xcat_te = np.stack([test[c].cat.codes.to_numpy() for c in CATF], axis=1).astype(np.int64)
assert Xcat_all.min() >= 0 and Xcat_te.min() >= 0, "unseen category produced code -1"

y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)
print(f"  n_num={len(NUMF)} n_cat={len(CATF)} cat_cards={cat_cards}  rows={n}", flush=True)


def standardize_fit(rows):
    """Fit mean/std on CONT columns of train rows only."""
    mu = Xnum_all[rows, :N_CONT].mean(0)
    sd = Xnum_all[rows, :N_CONT].std(0) + 1e-8
    return mu, sd


def apply_std(Xnum, mu, sd):
    """Standardize CONT columns; leave FLAG columns as-is."""
    out = Xnum.copy()
    out[:, :N_CONT] = (out[:, :N_CONT] - mu) / sd
    return out


def build_df(Xnum_std, Xcat):
    """Build a DataFrame for pytabkit (named columns, categoricals as int)."""
    df = pd.DataFrame(Xnum_std, columns=NUMF)
    for j, c in enumerate(CATF):
        df[c] = Xcat[:, j]
    return df


CAT_INDICATOR = [False] * len(NUMF) + [True] * len(CATF)


def make_model():
    """Fresh RealMLP_TD_Classifier with tuned defaults."""
    return RealMLP_TD_Classifier(
        device="cuda",
        random_state=SEED,
        n_epochs=256,
        batch_size=4096,
        val_metric_name="1-balanced_accuracy",
        verbosity=1,
    )


# ---- OOF loop ----
oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running OOF (RealMLP_TD, pytabkit, CUDA) …", flush=True)
for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)

    # Standardize CONT inside train fold only (fit_in_fold)
    mu, sd = standardize_fit(tr_idx)
    Xn_tr = apply_std(Xnum_all[tr_idx], mu, sd)
    Xn_va = apply_std(Xnum_all[val_idx], mu, sd)

    df_tr = build_df(Xn_tr, Xcat_all[tr_idx])
    df_va = build_df(Xn_va, Xcat_all[val_idx])

    clf = make_model()
    clf.fit(df_tr, y[tr_idx], cat_indicator=CAT_INDICATOR)

    proba = clf.predict_proba(df_va)
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    print(f"  fold {fold_id}: balanced_accuracy = {s:.6f}", flush=True)

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold), flush=True)
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}", flush=True)

np.save(NODE_DIR / "oof.npy", oof_proba)
print(f"  saved oof.npy shape={oof_proba.shape}", flush=True)


# ---- full-train fit → test probs + submission ----
print("Retraining on full train for the test set …", flush=True)
mu_full, sd_full = standardize_fit(np.arange(n))
df_full_tr = build_df(apply_std(Xnum_all, mu_full, sd_full), Xcat_all)
clf_full = make_model()
clf_full.fit(df_full_tr, y, cat_indicator=CAT_INDICATOR)
df_te = build_df(apply_std(Xnum_te, mu_full, sd_full), Xcat_te)
tp = clf_full.predict_proba(df_te)
np.save(NODE_DIR / "test_probs.npy", tp)
print(f"  saved test_probs.npy shape={tp.shape}", flush=True)

labels = np.array([LABEL_ORDER[i] for i in tp.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows)", flush=True)

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
print(f"  OOF full balanced_accuracy={oof_metric:.6f}", flush=True)
print("Done.", flush=True)
