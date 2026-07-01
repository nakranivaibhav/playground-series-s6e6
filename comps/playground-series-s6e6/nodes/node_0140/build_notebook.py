"""Generate the Kaggle notebook (.ipynb) from the VERIFIED kernel.

Model / feature / training code is sliced VERBATIM from kaggle_kernel_realmlp.py
(so the verified CV 0.969305 is preserved byte-for-byte). Only the orchestration
cells are rewritten to print intermediate results, and markdown is added between.
"""
import json
from pathlib import Path

SRC = Path(__file__).parent / "kaggle_kernel_realmlp.py"
OUT = Path(__file__).parent / "kaggle_notebook" / "s6e6-realmlp-single-model.ipynb"
lines = SRC.read_text().splitlines()

def slice_(a, b):
    """1-indexed inclusive line slice from the verified kernel."""
    return "\n".join(lines[a - 1:b])

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
# 🌌 Stellar Classification — a single RealMLP (5-fold bagged)

**Goal:** label each object as **GALAXY**, **QSO** (quasar), or **STAR**.
**Metric:** *balanced accuracy* (the average of the three per-class recall scores).

This notebook trains **one** neural network — no stacking, no blending of different
models. The only ensembling is that we train the same model **5 times** on 5
different 80/20 splits of the data and average their predictions ("bagging").

**Score:** about **0.9693 cross-validation** / **~0.9701 public leaderboard**.

### What makes it work (in plain terms)
1. **Feature engineering** — colours (`u-g`, `g-r`, …), magnitude summaries, and
   redshift ratios. Plus binned + target-encoded versions of every column.
2. **`fs_zsoft`** — four extra features that re-express **redshift** relative to its
   measurement error. The hard GALAXY-vs-STAR mix-up all happens at redshift ≈ 0;
   these features stretch that tiny zone open so the model can see it.
3. **RealMLP** — a strong, known-good tabular neural-net recipe (periodic feature
   embeddings, an 8-way internal ensemble, EMA weight averaging, a cosine learning
   rate). This is the heavy part; you don't need to follow every line.

### How to run
- Accelerator: **GPU** (top-right settings). Internet can stay **off**.
- **Run All** → it writes `submission.csv`. Takes roughly **10–40 min** on a Kaggle GPU.

> Note on leakage: anything that *learns* from the data (the scaler, the category
> codes, the target-encoder) is fit **only on each fold's training rows**, never on
> the validation rows or the test set. The `fs_zsoft` features use a fixed constant,
> so they're safe to compute once. The target and id columns are never inputs.
"""))

# ── 1. Setup ───────────────────────────────────────────────────────────────
cells.append(md(r"""
## 1 · Setup

Import libraries, set the random seed (so results are reproducible), and pick the
GPU if one is available.
"""))
cells.append(code(slice_(48, 67)))                 # imports
cells.append(code(slice_(69, 107)))                # paths, constants, seed, device

# ── 2. Load data ───────────────────────────────────────────────────────────
cells.append(md(r"""
## 2 · Load the data

Read the three CSV files and look at how the classes are balanced — this matters
because the metric (balanced accuracy) weights each class equally even though
GALAXY is far more common than STAR.
"""))
cells.append(code(r"""
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw  = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values   # GALAXY/QSO/STAR -> 0/1/2
n_train, n_test = len(train_raw), len(test_raw)

print(f"train rows: {n_train:,}   test rows: {n_test:,}   columns: {list(train_raw.columns)}")
print("\nclass balance (train):")
print(train_raw[TARGET].value_counts(normalize=True).round(4).to_string())
""".strip("\n")))

# ── 3. Feature engineering ─────────────────────────────────────────────────
cells.append(md(r"""
## 3 · Feature engineering (the leakage-free part)

These are simple **row-by-row** formulas — no information crosses between rows, so
they can be computed once for the whole dataset with zero leakage risk.

- `stateless_fe` — colours, magnitude mean/range, redshift ratios, a smooth log of redshift.
- `zsoft_fe` — the four **`fs_zsoft`** features that open up the redshift ≈ 0 region:
  - `_zsoft_snr` — redshift signal-to-noise vs the error floor
  - `_zsoft_asinh` — a smooth warp (linear near 0, log-like far out)
  - `_zsoft_log` — log of (shifted) redshift
  - `_zsoft_star` — a soft "is this a star?" bump peaking at redshift = 0
"""))
cells.append(code(slice_(145, 189)))               # globals + stateless_fe + zsoft_fe
cells.append(code(r"""
X      = zsoft_fe(stateless_fe(train_raw.drop(columns=[IDC, TARGET])))
X_test = zsoft_fe(stateless_fe(test_raw.drop(columns=[IDC])))
assert TARGET not in X.columns and IDC not in X.columns, "target/id leaked into features!"

zsoft_cols = [c for c in X.columns if "zsoft" in c]
print(f"feature columns built: {X.shape[1]}")
print(f"fs_zsoft features: {zsoft_cols}")
print("\nsample of the fs_zsoft features (first 5 rows):")
print(X[zsoft_cols].head().round(3).to_string())
""".strip("\n")))

# ── 4. Fold-wise encoding ──────────────────────────────────────────────────
cells.append(md(r"""
## 4 · Fold-wise encoding (fit on the training rows only)

These steps **learn** from the data, so to avoid leakage they must be fit inside
each fold on the training rows, then applied to the validation rows and the test set:

- integer codes for the raw categorical columns,
- integer-floor "category" views of every numeric column,
- quantile bins of the `delta` coordinate,
- a couple of interaction crosses, then **multiclass target-encoding** of those crosses.
"""))
cells.append(code(slice_(198, 270)))               # fit_fold_categoricals + add_target_encoding

# ── 5. The model ───────────────────────────────────────────────────────────
cells.append(md(r"""
## 5 · The RealMLP model

First the **settings** (a tuned bundle — treat them as fixed). `n_ens = 8` means each
model is internally 8 sub-networks averaged together.
"""))
cells.append(code(slice_(115, 138)))               # CONFIG
cells.append(md(r"""
Now the **network itself**. This is the known-good "reference recipe" architecture.
The building blocks:

- `NumericalPreprocessor` — center + scale numbers robustly (handles outliers).
- `CategoricalFeatureLayer` — one-hot for small categories, learned embeddings for big ones.
- `PBLDEmbedding` — a periodic (cosine) embedding of each numeric feature.
- `NTPLinear` / `ScalingLayer` — linear layers that run the 8 ensemble members in parallel.
- `RealMLP` — stitches it all into one 8-way ensemble.

*You don't need to read every line — it's a self-contained, well-tested block.*
"""))
cells.append(code(slice_(283, 406)))               # model classes
cells.append(md(r"""
## 6 · Training helpers

The learning-rate / dropout / label-smoothing **schedules**, the per-parameter-group
optimizer settings, the loss function (cross-entropy with label smoothing), and a
small scikit-learn-style **wrapper** that trains one RealMLP and keeps the best
(EMA) weights.
"""))
cells.append(code(slice_(411, 463)))               # apply_schedule, get_parameter_groups, smooth_ce_loss
cells.append(code(slice_(466, 599)))               # RealMLP_TD_Classifier

# ── 7. Train 5 folds ───────────────────────────────────────────────────────
cells.append(md(r"""
## 7 · Train 5 folds and average

We split the training data into 5 stratified folds (same class balance in each).
For each fold we fit the encoders + train a RealMLP on 4/5 of the data, then:
- store its predictions on the held-out 1/5 (used to measure cross-validation), and
- add 1/5 of its **test** predictions to a running average (the bagging step).

Each fold prints its validation balanced-accuracy as it finishes.
"""))
cells.append(code(r"""
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(np.arange(n_train), y_all))

oof = np.zeros((n_train, N_CLASSES), dtype=np.float32)        # out-of-fold preds (for CV)
test_proba = np.zeros((n_test, N_CLASSES), dtype=np.float32)  # averaged test preds
fold_scores = []

for fold_id, (tr_idx, val_idx) in enumerate(folds):
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)
    print(f"\n----- Fold {fold_id}  (train={len(tr_idx):,}  val={len(val_idx):,}) -----", flush=True)

    # fit categoricals + target-encoding on THIS fold's train rows only
    X_tr, X_val, X_te, cat_cols, combos = fit_fold_categoricals(
        X.iloc[tr_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        X_test.copy())
    X_tr, X_val, X_te = add_target_encoding(X_tr, y_all[tr_idx], X_val, X_te, combos, fold_seed)

    # keep a consistent column order across train/val/test
    X_tr  = X_tr.reindex(sorted(X_tr.columns), axis=1)
    X_val = X_val.reindex(sorted(X_val.columns), axis=1)
    X_te  = X_te.reindex(sorted(X_te.columns), axis=1)
    cat_cols = sorted(cat_cols)

    model = RealMLP_TD_Classifier(random_state=fold_seed, device=str(DEVICE))
    model.fit(X_tr, y_all[tr_idx], X_val, y_all[val_idx], cat_col_names=cat_cols, X_test=X_te)

    oof[val_idx] = model.best_val_probs_.astype("float32")
    test_proba += model.predict_proba(X_te).astype("float32") / N_SPLITS

    s = balanced_accuracy_score(y_all[val_idx], oof[val_idx].argmax(1))
    fold_scores.append(s)
    print(f"  Fold {fold_id} balanced accuracy = {s:.6f}", flush=True)

    del model, X_tr, X_val, X_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll folds done.")
""".strip("\n")))

# ── 8. Results + submission ────────────────────────────────────────────────
cells.append(md(r"""
## 8 · Results & submission

Average the 5 fold scores for the final cross-validation number, then write
`submission.csv` from the bagged test predictions.
"""))
cells.append(code(r"""
cv = float(np.mean(fold_scores))
print("per-fold balanced accuracy:", [f"{s:.6f}" for s in fold_scores])
print(f"\nCROSS-VALIDATION (balanced accuracy) = {cv:.6f}")
print("(expected ~0.9693; public leaderboard ~0.9701)")
""".strip("\n")))
cells.append(code(r"""
pred_labels = np.array([CLASSES[i] for i in test_proba.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv("submission.csv", index=False)

print(f"wrote submission.csv  ({len(sub):,} rows)")
print("\npredicted class distribution:")
print(sub[TARGET].value_counts().to_string())
print("\nfirst rows:")
print(sub.head().to_string(index=False))
""".strip("\n")))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
