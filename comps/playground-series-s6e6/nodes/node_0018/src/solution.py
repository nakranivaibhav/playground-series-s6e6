"""node_0018 — LightGBM + redshift-conditional target encoding (fs_tgt_enc).

Built on: node_0006's fs_research feature matrix + identical LightGBM hyperparameters
          (n_estimators=500, lr=0.05, num_leaves=63, class_weight='balanced').
          All stateless features (colors, redshift flags, QSO box, galactic coords) are
          byte-identical to node_0006.

Change: adds feature-set fs_tgt_enc (leak-safety: fit_in_fold). Inside each OOF train
fold ONLY, computes 9 smoothed (m=100) target-encoded posterior columns:
  - P(class | spectral_type): 3 cols (one per GALAXY/QSO/STAR), smoothed toward global prior.
  - P(class | galaxy_population): 3 cols, same scheme.
  - P(class | redshift_band): 3 cols; redshift is first binned into ~10 quantile bands
    whose edges are computed from the TRAIN fold only, then target-encoded per band.
For the final full-train refit, the same encoding is fit on all train rows (test is
genuinely unseen -- the target encoding is built from train labels, applied to test rows).
NEVER uses val-fold rows in fitting the encoding statistics or the band edges.

Metric: Balanced Accuracy Score (maximize).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

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
M_SMOOTH = 100          # smoothing pseudocount for target encoding
N_REDSHIFT_BINS = 10    # quantile bands for redshift
TGT_ENC_CATS = ["spectral_type", "galaxy_population"]

TGT_ENC_COLS = (
    [f"te_{col}_{lbl.lower()}" for col in TGT_ENC_CATS for lbl in LABEL_ORDER]
    + [f"te_redshift_band_{lbl.lower()}" for lbl in LABEL_ORDER]
)


def score_fn(yt, yp):
    return balanced_accuracy_score(yt, yp)


def make_model():
    # byte-identical to node_0006 / node_0001 (CV-proven near-optimal)
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
        num_leaves=63, n_jobs=-1, class_weight="balanced", random_state=42, verbosity=-1,
    )


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


def compute_tgt_enc_stats(train_sub: pd.DataFrame) -> dict:
    """Compute smoothed target-encoding stats from a training fold.

    CRITICAL: called only with rows inside the current train fold.
    Never called with validation or test rows.
    """
    y = train_sub[TARGET]
    # global class prior
    global_prior = np.array([(y == lbl).mean() for lbl in LABEL_ORDER])  # shape [3]

    # per-category target-encoding (smoothed toward global prior)
    cat_stats = {}
    for col in TGT_ENC_CATS:
        if col not in train_sub.columns:
            continue
        col_stats = {}
        for val, grp in train_sub.groupby(col, observed=True):
            n_g = len(grp)
            y_g = grp[TARGET]
            raw = np.array([(y_g == lbl).mean() for lbl in LABEL_ORDER])
            smoothed = (n_g * raw + M_SMOOTH * global_prior) / (n_g + M_SMOOTH)
            col_stats[val] = smoothed
        cat_stats[col] = col_stats

    # redshift-band: compute quantile edges from train fold only
    z = train_sub["redshift"].to_numpy()
    quantiles = np.linspace(0, 100, N_REDSHIFT_BINS + 1)
    band_edges = np.unique(np.percentile(z, quantiles))
    if len(band_edges) < 2:
        band_edges = np.array([z.min(), z.max() + 1e-9])

    # assign each train row to a band (interior edges only for digitize)
    band_labels = np.digitize(z, band_edges[1:-1])  # 0..N_REDSHIFT_BINS-1
    band_stats = {}
    for b in np.unique(band_labels):
        mask = band_labels == b
        n_b = int(mask.sum())
        y_b = y.iloc[mask]
        raw = np.array([(y_b == lbl).mean() for lbl in LABEL_ORDER])
        smoothed = (n_b * raw + M_SMOOTH * global_prior) / (n_b + M_SMOOTH)
        band_stats[int(b)] = smoothed

    return {
        "global_prior": global_prior,
        "cat_stats": cat_stats,
        "band_edges": band_edges,
        "band_stats": band_stats,
    }


def apply_tgt_enc(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply pre-computed target-encoding stats to any split (val or test).

    Unseen categories/bands fall back to the global prior (no leakage).
    Uses vectorized pandas map for speed (no Python loops over rows).
    """
    out = df.copy()
    global_prior = stats["global_prior"]

    # categorical target-encoding -- vectorized via pd.Series.map(dict)
    for col, col_stats in stats["cat_stats"].items():
        vals = out[col].astype(object)
        for class_idx, lbl in enumerate(LABEL_ORDER):
            col_name = f"te_{col}_{lbl.lower()}"
            # build a plain dict {value: float} for each class
            enc_dict = {v: float(arr[class_idx]) for v, arr in col_stats.items()}
            mapped = vals.map(enc_dict)  # NaN for unseen
            # fill unseen with global prior
            mapped = mapped.fillna(float(global_prior[class_idx]))
            out[col_name] = mapped.astype(np.float32)

    # redshift-band target-encoding -- np.digitize is vectorized
    band_edges = stats["band_edges"]
    band_stats = stats["band_stats"]
    z = out["redshift"].to_numpy()
    band_labels = np.digitize(z, band_edges[1:-1])
    # build lookup arrays indexed by band label
    all_bands = sorted(band_stats.keys())
    max_band = max(all_bands) if all_bands else 0
    for class_idx, lbl in enumerate(LABEL_ORDER):
        col_name = f"te_redshift_band_{lbl.lower()}"
        # create lookup array (index = band label, fill unknown with global prior)
        lookup = np.full(max_band + 2, float(global_prior[class_idx]), dtype=np.float32)
        for b, arr in band_stats.items():
            lookup[b] = float(arr[class_idx])
        # clip band_labels to valid range and index
        bl_clipped = np.clip(band_labels, 0, max_band + 1)
        out[col_name] = lookup[bl_clipped]

    return out


print("Loading data ...")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

train = engineer(train)
test = engineer(test)

# base feature columns (stateless fs_research from node_0006 -- no te_ cols)
base_feat_cols = [c for c in feature_columns(train) if not c.startswith("te_")]
print(f"  base features ({len(base_feat_cols)}): {base_feat_cols}")

n = len(train)
oof_proba = np.zeros((n, 3))
oof_labels = np.empty(n, dtype=object)
per_fold = []

print("Running 5-fold OOF with fit-in-fold target encoding ...")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)

    # fit encoding stats on TRAIN fold only -- NEVER touches val rows
    stats = compute_tgt_enc_stats(train.iloc[tr_idx])

    # apply encoding to train fold and val fold independently
    train_enc = apply_tgt_enc(train.iloc[tr_idx], stats)
    val_enc = apply_tgt_enc(train.iloc[val_idx], stats)

    feat_cols = base_feat_cols + TGT_ENC_COLS
    X_tr = train_enc[feat_cols]
    y_tr = train_enc[TARGET]
    X_va = val_enc[feat_cols]
    y_va = val_enc[TARGET]

    model = make_model()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)
    co = list(model.classes_)
    for lbl in LABEL_ORDER:
        oof_proba[val_idx, LABEL2IDX[lbl]] = proba[:, co.index(lbl)]
    preds = np.array([co[i] for i in np.argmax(proba, axis=1)])
    oof_labels[val_idx] = preds
    s = score_fn(y_va.values, preds)
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}")

oof_metric = score_fn(train[TARGET].values, oof_labels)
mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"oof_metric={oof_metric:.6f}")
np.save(NODE_DIR / "oof.npy", oof_proba)

# final feature list: base + te cols
feat_cols_final = base_feat_cols + TGT_ENC_COLS
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols_final) + "\n")

print("Retraining on full train (target-encoding fit on all train rows) ...")
# fit encoding stats on full train -- test is genuinely unseen
full_stats = compute_tgt_enc_stats(train)
train_full_enc = apply_tgt_enc(train, full_stats)
test_enc = apply_tgt_enc(test, full_stats)

X_full = train_full_enc[feat_cols_final]
y_full = train_full_enc[TARGET]
X_test = test_enc[feat_cols_final]

fm = make_model()
fm.fit(X_full, y_full)
tp = fm.predict_proba(X_test)
co = list(fm.classes_)
tp_ord = np.zeros((len(X_test), 3))
for lbl in LABEL_ORDER:
    tp_ord[:, LABEL2IDX[lbl]] = tp[:, co.index(lbl)]
np.save(NODE_DIR / "test_probs.npy", tp_ord)
labels = np.array([LABEL_ORDER[i] for i in np.argmax(tp_ord, axis=1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved test_probs.npy")
print("Done.")
