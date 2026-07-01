"""node_0133 — TabPFN-v2 FINETUNE on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0033:
  Replace the TabM model with `tabpfn.finetuning.FinetunedTabPFNClassifier`
  (a gradient finetune of the pretrained TabPFN-v2 transformer), trained on the
  SAME fs_realmlp_fe feature-set. First finetuned foundation model in this comp —
  n22/n25/n27 (TabPFN) and n26 (TabICL) were all FROZEN in-context.

FE pipeline (byte-identical to node_0033 / node_0028):
  - Stateless FE: redshift ratios, 7 color pairs, mag_mean, mag_range, log1p_redshift.
  - fit_in_fold factorize maps + KBinsDiscretizer (delta bins) on train fold only.
  - fit_in_fold TargetEncoder (combo cats) on train fold only.
  - NO standardization (TabPFN does its own preprocessing).

Staging (env vars):
  - TABPFN_SMOKE=1 : 30k subsample, 3 epochs, tiny inference subsample — pipeline check.
  - TABPFN_FOLD0=1 : real fold-0 only — tier read + true timing, then exit (cheap-kill).
  - (neither)      : full 5-fold → oof.npy + test_probs.npy + submission.csv.

Outputs (full run): oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tabpfn.finetuning import FinetunedTabPFNClassifier

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE = os.environ.get("TABPFN_SMOKE") == "1"
FOLD0_ONLY = os.environ.get("TABPFN_FOLD0") == "1"

# Cached TabPFN-v2 weights (same checkpoint the frozen nodes n22 used) — pass model_path
# directly so the finetuner loads locally and skips the one-time license/download gate.
TABPFN_CKPT = Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-classifier.ckpt"
assert TABPFN_CKPT.exists(), f"TabPFN-v2 checkpoint not found: {TABPFN_CKPT}"

# ── TabPFN finetune hyperparameters ──
# lr bumped to 3e-5 (research: 3e-5–1e-4 for 10k+ row datasets; default 1e-5 too low).
FT_EPOCHS = 3 if SMOKE else 30
FT_LR = 3e-5
FT_CTX_QUERY = 4000 if SMOKE else 10000      # ctx+query samples per finetune step
FT_INFER_SUBSAMPLE = 8000 if SMOKE else 50000  # support rows used at inference
FT_TIME_LIMIT = None if not SMOKE else 120     # cap finetune wall-time (seconds) in smoke


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# ─── Feature engineering (byte-identical to node_0033) ───────────────────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]
IMPORTANT_COMBOS = sorted([("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")])


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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr, va, te = df_tr.copy(), df_val.copy(), df_te.copy()

    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

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
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    try:
        encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto",
                                shuffle=True, random_state=fold_seed)
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


def to_float_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """All feature columns as float32 (category cols -> their int label -> float)."""
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        s = df[c]
        if str(s.dtype) == "category":
            out[:, j] = s.astype("int32").to_numpy().astype(np.float32)
        else:
            out[:, j] = s.to_numpy().astype(np.float32)
    return out


# ─── Load data ────────────────────────────────────────────────────────────────
log(f"Device={DEVICE}  SMOKE={SMOKE}  FOLD0_ONLY={FOLD0_ONLY}")
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train, n_test = len(train_raw), len(test_raw)

keep_sm = None
if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, fold-0 only")
    keep_sm = set(np.random.default_rng(0).choice(n_train, 30000, replace=False).tolist())
    folds_list = [folds_list[0]]
elif FOLD0_ONLY:
    log("FOLD0_ONLY: real fold-0 tier read")
    folds_list = [folds_list[0]]

# ─── Stateless FE (once) ──────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_stateless = stateless_fe(train_raw.drop(columns=[IDC, TARGET]))
X_test_stateless = stateless_fe(test_raw.drop(columns=[IDC]))
log(f"  X_stateless={X_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
feat_cols_final = None
n_folds_for_test = len(folds_list)

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        tr_idx = np.array([i for i in tr_idx if i in keep_sm])
        val_idx = np.array([i for i in val_idx if i in keep_sm])

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, _ = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed)

    feat_cols = sorted(X_tr_fold.columns)
    if feat_cols_final is None:
        feat_cols_final = feat_cols
        log(f"  n_features={len(feat_cols)}")

    X_tr = to_float_matrix(X_tr_fold, feat_cols)
    X_va = to_float_matrix(X_val_fold, feat_cols)
    X_te = to_float_matrix(X_te_fold, feat_cols)
    del X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    log(f"  fitting FinetunedTabPFNClassifier (epochs={FT_EPOCHS} lr={FT_LR} "
        f"ctxq={FT_CTX_QUERY} infer_sub={FT_INFER_SUBSAMPLE}) on {len(X_tr)} rows ...")
    clf = FinetunedTabPFNClassifier(
        device=DEVICE,
        epochs=FT_EPOCHS,
        learning_rate=FT_LR,
        time_limit=FT_TIME_LIMIT,
        n_finetune_ctx_plus_query_samples=FT_CTX_QUERY,
        n_inference_subsample_samples=FT_INFER_SUBSAMPLE,
        random_state=fold_seed,
        early_stopping=True,
        extra_classifier_kwargs={"model_path": str(TABPFN_CKPT)},
    )
    t_fit0 = time.perf_counter()
    clf.fit(X_tr, y_tr_fold)
    log(f"  fit done in {time.perf_counter()-t_fit0:.1f}s")

    t_pred0 = time.perf_counter()
    val_probs = clf.predict_proba(X_va).astype(np.float32)
    log(f"  val predict ({len(X_va)} rows) in {time.perf_counter()-t_pred0:.1f}s")
    oof_proba[val_idx] = val_probs

    if not (SMOKE or FOLD0_ONLY):
        test_probs_fold = clf.predict_proba(X_te).astype(np.float32)
        test_proba_accum += test_probs_fold / n_folds_for_test

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_probs, axis=1))
    per_fold_scores.append(fold_score)
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={time.perf_counter()-fold_t0:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        log(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    del clf, X_tr, X_va, X_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        ft = time.perf_counter() - fold_t0
        log(f"  TIMING: fold0={ft:.1f}s  projected_5fold≈{ft*5:.1f}s ({ft*5/60:.1f}min)")

if SMOKE:
    log("[smoke] OK — pipeline ran end-to-end. Exiting before saving artifacts.")
    sys.exit(0)
if FOLD0_ONLY:
    log(f"[fold0] tier read: BA={per_fold_scores[0]:.6f} "
        f"({'CLEARS' if per_fold_scores[0] >= 0.960 else 'BELOW'} 0.960 cheap-kill bar)")
    print(f"cv={per_fold_scores[0]:.6f}", flush=True)
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

np.save(NODE_DIR / "oof.npy", oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved oof.npy {oof_proba.shape}  test_probs.npy {test_proba_accum.shape}")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)\n{sub[TARGET].value_counts().to_string()}")

(NODE_SRC / "features.txt").write_text("\n".join(feat_cols_final) + "\n")
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")
log(f"Total elapsed: {time.perf_counter()-T0:.1f}s")
log("Done.")
