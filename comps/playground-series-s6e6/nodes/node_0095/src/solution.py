"""node_0095 — draft (gbdt): LightGBM on fs_zresid_strict (STRICT residual-only features).

THE ONE ATOMIC CHANGE vs any prior residual node:
  STRICT residual-only feature set fs_zresid_strict:
  - ONLY z-conditional per-COLOR residual z-scores ((color − mean_zbin)/std_zbin
    over ~40 redshift quantile bins)
  - PLUS one binary STAR-flag (1 if row falls in the z≈0 lowest quantile bin)
  - DROPS: raw redshift-as-continuous, raw magnitudes, raw colors, per-mag z-scores
    (the prior fs_zresid in node_0087 kept raw redshift + per-mag residuals —
    this is the strict no-raw-z/no-raw-mag/no-mag-residual version)

COLORS in fs_zresid_strict: (u-g), (g-r), (r-i), (i-z), (u-z) — 5 residuals.
STAR-flag: binary indicator = 1 if row's redshift falls in the lowest z-bin (z≈0).

DECISIVE CHEAP GATE — fold-0 ONLY:
  Compute fold-0 err-corr vs node_0070's OOF on fold-0 val set.
  KILL (do NOT run folds 1-4) if:
    - fold-0 solo balanced accuracy < 0.96, OR
    - fold-0 err_corr vs node_0070 >= 0.65
  Only if BOTH pass: run full 5-fold OOF + write artifacts.

LEAKAGE DISCIPLINE (fs_zresid_strict is fit_in_fold):
  - Redshift quantile bin EDGES fit on train-fold rows ONLY.
  - Per-bin per-color MEAN/STD fit on train-fold rows ONLY.
  - Global mean/std fallback for sparse bins (< MIN_BIN_SAMPLES) — also train-fold-only.
  - STAR-flag computed from the bin assignment (train-fold bin edges).
  - All of the above applied to val and test using the train-fold-fitted edges/stats.
  - Frozen folds.json used throughout.
  - node_0070 OOF is fold-honest (safe external source — each row was predicted
    by a model that NEVER saw that row).

LightGBM config: reuse node_0030 recipe verbatim.
  - n_estimators=2000, lr=0.05, num_leaves=127, class_weight='balanced',
    early stopping on fold val (150 rounds), CPU mode.

Class order: GALAXY=0, QSO=1, STAR=2
n_train=577347, n_test=247435

Outputs (only if full 5-fold gate passes):
  oof.npy (577347,3), test_probs.npy (247435,3), submission.csv
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

# fs_zresid_strict feature parameters
COLORS = ["u-g", "g-r", "r-i", "i-z", "u-z"]  # ONLY these 5 colors; NO magnitude z-scores
N_ZBINS = 40        # ~40 redshift quantile bins
MIN_BIN_SAMPLES = 5 # sparse bin fallback threshold

# Fold-0 kill gate thresholds (from node.md plan)
FOLD0_BA_THRESHOLD = 0.96
FOLD0_ERR_CORR_THRESHOLD = 0.65


def compute_raw_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 colors needed for fs_zresid_strict (intermediate only; not kept)."""
    d = df.copy()
    d["u-g"] = (d["u"] - d["g"]).astype("float32")
    d["g-r"] = (d["g"] - d["r"]).astype("float32")
    d["r-i"] = (d["r"] - d["i"]).astype("float32")
    d["i-z"] = (d["i"] - d["z"]).astype("float32")
    d["u-z"] = (d["u"] - d["z"]).astype("float32")
    return d


def fit_zresid_strict(df_tr: pd.DataFrame):
    """
    Fit fs_zresid_strict encoder on TRAIN-FOLD ROWS ONLY.

    Returns (bin_edges, per_bin_stats, global_stats, star_bin_id):
      - bin_edges: 1D array of N_ZBINS+1 edges from train-fold redshift quantiles.
      - per_bin_stats: dict {color_name: (bin_means_arr, bin_stds_arr)} for each of
        the actual n_actual_bins bins.
      - global_stats: dict {color_name: (global_mean, global_std)} fallback.
      - star_bin_id: the bin index corresponding to z≈0 (the lowest bin = 0).

    STRICT version: only COLORS (u-g, g-r, r-i, i-z, u-z), NO magnitudes.
    STAR-flag = 1 if the row falls in bin 0 (the z≈0 lowest redshift bin).
    """
    z = df_tr["redshift"].values

    # Quantile edges from training fold only
    quantiles = np.linspace(0, 100, N_ZBINS + 1)
    bin_edges = np.percentile(z, quantiles)
    # Ensure strictly increasing by deduplication
    bin_edges = np.unique(bin_edges)
    n_actual_bins = len(bin_edges) - 1

    # Assign each train row to a bin (0..n_actual_bins-1)
    bin_ids = np.searchsorted(bin_edges[1:-1], z, side="right")
    # searchsorted returns index into the interior edges; clip to valid range
    bin_ids = np.clip(bin_ids, 0, n_actual_bins - 1)

    df_c = compute_raw_colors(df_tr)

    per_bin_stats = {}
    global_stats = {}

    for feat in COLORS:
        vals = df_c[feat].values.astype(float)
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
                # Sparse bin — fall back to global stats (train-fold-only)
                bm = global_mean
                bs = global_std
            bin_means.append(bm)
            bin_stds.append(bs)
        per_bin_stats[feat] = (np.array(bin_means), np.array(bin_stds))

    # STAR-flag: the lowest redshift bin is z≈0 (bin index 0)
    star_bin_id = 0

    return bin_edges, per_bin_stats, global_stats, star_bin_id


def apply_zresid_strict(df: pd.DataFrame, bin_edges, per_bin_stats, global_stats, star_bin_id) -> pd.DataFrame:
    """
    Apply fs_zresid_strict transform to any dataframe (train fold, val, or test).
    Uses train-fold-fitted bin edges, per-bin stats, and global stats.

    Returns a DataFrame with:
      - zr_{color} for each of the 5 colors (residual z-scores) — 5 features
      - star_flag: binary 1 if row is in the lowest redshift bin (z≈0)
      Total: 6 features.

    DROPS: raw redshift, raw magnitudes, raw colors, per-magnitude z-scores.
    """
    z = df["redshift"].values
    n_actual_bins = len(bin_edges) - 1

    # Assign each row to a bin (using the train-fold edges)
    bin_ids = np.searchsorted(bin_edges[1:-1], z, side="right")
    bin_ids = np.clip(bin_ids, 0, n_actual_bins - 1)

    df_c = compute_raw_colors(df)

    residuals = {}
    for feat in COLORS:
        vals = df_c[feat].values.astype(float)
        gm, gs = global_stats[feat]
        bm_arr, bs_arr = per_bin_stats[feat]

        # Vectorized bin lookup (fall back to global if bin index out of range)
        means = np.where(bin_ids < len(bm_arr), bm_arr[bin_ids], gm)
        stds  = np.where(bin_ids < len(bs_arr), bs_arr[bin_ids], gs)
        # Ensure stds are not degenerate
        stds  = np.where(stds < 1e-10, gs, stds)

        residuals[f"zr_{feat}"] = ((vals - means) / stds).astype("float32")

    out = pd.DataFrame(residuals, index=df.index)
    # STAR-flag: 1 if row is in the z≈0 (lowest) redshift bin
    out["star_flag"] = (bin_ids == star_bin_id).astype("int8")

    return out


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    """
    Verbatim node_0030 LightGBM recipe. Class_weight='balanced' for balanced accuracy.
    No change from node_0030 except random_state uses fold_seed.
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
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# ─── Pre-flight leakage checks BEFORE training ───────────────────────────────
# Check 1+2: target and id not in raw feature columns
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

assert TARGET not in X_raw.columns, f"TARGET {TARGET} in raw features — LEAK!"
assert IDC not in X_raw.columns, f"ID {IDC} in raw features — LEAK!"
log("Pre-flight check 1+2: target and id NOT in raw feature columns — PASS")

# Check 3: single-feature ~ target sweep on <=50k sample (raw input features)
log("Pre-flight check 3: single-feature ~ target sweep on raw inputs ...")
rng_lk = np.random.RandomState(0)
sample_size = min(50_000, n_train)
sidx = rng_lk.choice(n_train, sample_size, replace=False)
ys_sample = y_all[sidx].astype(float)
raw_check_cols = ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]
for col in raw_check_cols:
    xv = X_raw.iloc[sidx][col].values.astype(float)
    if np.std(xv) > 1e-10:
        corr = abs(np.corrcoef(xv, ys_sample)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK SMELL: {col} ~ target |corr|={corr:.4f}")
log(f"  check3 PASS: no raw feature has |corr|>=0.999 with target (sample={sample_size})")

# Check 4: fit-inside-fold verification — noted: fs_zresid_strict bin edges/stats
# are computed inside the fold loop on tr_idx only (verified by code structure below).
log("Pre-flight check 4: fs_zresid_strict bin edges/stats WILL be fit inside fold loop — NOTED")

# Check 5: folds from frozen file
log(f"Pre-flight check 5: folds loaded from frozen folds.json — {len(folds_list)} folds — PASS")

# Check 6: verify assembled feature list does NOT contain target or id
# (will be verified once assembled after first fold's FE)
log("Pre-flight check 6: feature list check will run after first FE assembly")

# ─── Load node_0070 OOF for the fold-0 error-correlation gate ───────────────
NODE70_OOF_PATH = COMP_DIR / "nodes/node_0070/oof.npy"
log(f"Loading node_0070 OOF from {NODE70_OOF_PATH} ...")
oof_70 = np.load(NODE70_OOF_PATH)
assert oof_70.shape == (n_train, N_CLASSES), f"node_0070 OOF shape {oof_70.shape} != ({n_train},{N_CLASSES})"
y_pred_70 = np.argmax(oof_70, axis=1)
log(f"  node_0070 OOF shape={oof_70.shape} — loaded OK")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
best_iters = []

fold0_gate_passed = False
fold0_err_corr = None
fold0_ba = None
kill_after_fold0 = False

log("Starting OOF loop (FOLD-0 GATE FIRST) ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ─── fs_zresid_strict: ALL fitting on TRAIN FOLD ONLY (Check 4) ──────────
    df_tr_raw  = X_raw.iloc[tr_idx].reset_index(drop=True)
    df_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
    df_te_raw  = X_test_raw.reset_index(drop=True)

    # fit_in_fold: bin edges + per-bin color stats from train fold ONLY
    bin_edges, per_bin_stats, global_stats, star_bin_id = fit_zresid_strict(df_tr_raw)

    # Apply to train fold, val fold, and test (using train-fold-fitted params)
    X_tr_fold  = apply_zresid_strict(df_tr_raw,  bin_edges, per_bin_stats, global_stats, star_bin_id)
    X_val_fold = apply_zresid_strict(df_val_raw, bin_edges, per_bin_stats, global_stats, star_bin_id)
    X_te_fold  = apply_zresid_strict(df_te_raw,  bin_edges, per_bin_stats, global_stats, star_bin_id)

    # Sort columns consistently
    cols_sorted = sorted(X_tr_fold.columns)
    X_tr_fold  = X_tr_fold[cols_sorted]
    X_val_fold = X_val_fold[cols_sorted]
    X_te_fold  = X_te_fold[cols_sorted]

    # Check 6 (once): verify TARGET and IDC absent from assembled feature list
    if fold_id == 0:
        feature_set = set(cols_sorted)
        assert TARGET not in feature_set, f"TARGET {TARGET} in assembled features — LEAK!"
        assert IDC not in feature_set, f"ID {IDC} in assembled features — LEAK!"
        log(f"  check6 PASS: TARGET and ID absent from {len(feature_set)}-feature assembled matrix")
        log(f"  feature list: {cols_sorted}")

    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

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

    # ─── FOLD-0 GATE ─────────────────────────────────────────────────────────
    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # Error-correlation vs node_0070 on fold-0 val set
        val_pred_95 = np.argmax(val_proba, axis=1)
        val_pred_70 = y_pred_70[val_idx]
        y_val_true  = y_all[val_idx]

        err_95 = (val_pred_95 != y_val_true).astype(np.float32)
        err_70 = (val_pred_70 != y_val_true).astype(np.float32)

        fold0_err_corr = float(np.corrcoef(err_95, err_70)[0, 1])
        fold0_ba       = fold_score

        log(f"  FOLD-0 GATE CHECK:")
        log(f"    solo_BA={fold0_ba:.6f}  (threshold >= {FOLD0_BA_THRESHOLD})")
        log(f"    err_corr_vs_n70={fold0_err_corr:.4f}  (threshold < {FOLD0_ERR_CORR_THRESHOLD})")
        print(f"fold0_solo_ba={fold0_ba:.6f}", flush=True)
        print(f"fold0_err_corr_vs_n70={fold0_err_corr:.6f}", flush=True)

        if fold0_ba < FOLD0_BA_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: solo BA {fold0_ba:.6f} < {FOLD0_BA_THRESHOLD} — KILL")
            print(f"GATE_KILL: solo_BA={fold0_ba:.6f} below threshold {FOLD0_BA_THRESHOLD}", flush=True)
            kill_after_fold0 = True
            fold0_gate_passed = False
        elif fold0_err_corr >= FOLD0_ERR_CORR_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: err_corr {fold0_err_corr:.4f} >= {FOLD0_ERR_CORR_THRESHOLD} — KILL")
            print(f"GATE_KILL: err_corr={fold0_err_corr:.6f} above threshold {FOLD0_ERR_CORR_THRESHOLD}", flush=True)
            kill_after_fold0 = True
            fold0_gate_passed = False
        else:
            log(f"  FOLD-0 GATE PASSED: BA={fold0_ba:.6f}>={FOLD0_BA_THRESHOLD}, "
                f"err_corr={fold0_err_corr:.4f}<{FOLD0_ERR_CORR_THRESHOLD}")
            log("  Continuing to full 5-fold run ...")
            fold0_gate_passed = True

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    # KILL: stop after fold 0 if gate failed
    if fold_id == 0 and kill_after_fold0:
        log("KILL: stopping after fold-0 (gate failed) — not running folds 1-4")
        log(f"  fold0 solo_BA={fold0_ba:.6f}  err_corr_vs_n70={fold0_err_corr:.4f}")
        log(f"  node_0095 is a NULL RESULT — strict residual-only representation "
            f"did NOT pass decorrelation gate")
        break

# ─── Summary ─────────────────────────────────────────────────────────────────
if kill_after_fold0:
    log("\n=== NODE KILLED AT FOLD-0 ===")
    log(f"fold0_solo_BA={fold0_ba:.6f}  fold0_err_corr_vs_n70={fold0_err_corr:.4f}")
    log("status: valid (killed at fold-0, cheap gate, no full OOF produced)")
    log("No oof.npy / test_probs.npy / submission.csv written (not needed for killed node)")
    print(f"cv=null", flush=True)
    log("\nDone — fold-0 kill.")
    sys.exit(0)

# ─── Full 5-fold completed ────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Post-train output checks ─────────────────────────────────────────────────
log("Post-train leakage output checks ...")

# Check 7: OOF complete and no NaN
assert oof_proba.shape == (n_train, N_CLASSES), f"oof shape {oof_proba.shape}"
assert not np.isnan(oof_proba).any(), "NaN in OOF!"
row_sums = oof_proba.sum(axis=1)
assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: {row_sums.mean():.4f}"
log("  check7 PASS: oof_full=True, no_nan=True")

# Check 8: distribution sane
oof_preds = oof_proba.argmax(1)
unique_classes = np.unique(oof_preds)
assert len(unique_classes) >= 2, f"OOF predictions collapsed to {unique_classes}"
log(f"  check8 PASS: dist_sane=True  OOF class distribution: {np.bincount(oof_preds)}")

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

# Check 9: submission schema
# (formal check via validate_submission.py — run separately)
assert list(sub.columns) == list(sample_sub.columns), "submission column mismatch"
assert len(sub) == len(sample_sub), f"submission row count {len(sub)} != {len(sample_sub)}"
log("  check9: submission schema rows/cols match — PASS (formal validate_submission.py run separately)")

# ─── OOF full metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# Check 10: cv-too-good
# Parent baseline_cv = 0.970153 (champion). This is a decorrelation probe;
# if solo cv > 0.970 that would be unexpectedly good. Flag if so.
if mean_cv > 0.970:
    log(f"  cv_too_good WARN: cv={mean_cv:.6f} > 0.970 (champion baseline) — flag for human review")
else:
    log(f"  cv_too_good: cv={mean_cv:.6f} vs champion 0.970153 — no extraordinary flag")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
