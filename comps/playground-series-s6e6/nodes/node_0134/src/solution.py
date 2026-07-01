"""node_0134 — Mitra FINETUNE on fs_realmlp_fe features (via AutoGluon).

THE ONE ATOMIC CHANGE vs node_0133:
  Swap TabPFN-v2 for Mitra (Amazon tabular FM, AutoGluon MitraModel, fine_tune=True),
  driven per-fold by TabularPredictor for fold-honest OOF over the frozen folds.
  SAME fs_realmlp_fe feature-set. Sibling FM draft to n133.

FE pipeline: byte-identical to node_0133 / node_0033 (stateless + fit_in_fold cats + TE).

Staging (env): MITRA_SMOKE=1 (20k subsample, fold-0, time-capped) ·
               MITRA_FOLD0=1 (real fold-0 tier read) · (neither) full 5-fold.
Outputs (full): oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
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

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent
T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


TARGET, IDC, SEED, N_CLASSES = "class", "id", 42, 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

SMOKE = os.environ.get("MITRA_SMOKE") == "1"
FOLD0_ONLY = os.environ.get("MITRA_FOLD0") == "1"

# Mitra inference context (max_rows aux param). Default 10k; bump for a fairer shot
# (n133/TabPFN used 50k inference support). fine_tune adapts the pretrained weights.
MITRA_MAX_ROWS = 8000 if SMOKE else 20000  # 50k estimates 244GB (infeasible); 20k fits the guard
FIT_TIME_LIMIT = 180 if SMOKE else None
# Proper-finetune knobs (AG default is 50 steps / 1000 warmup = near-frozen). Set via env
# to actually finetune. 0/unset -> Mitra defaults.
MITRA_STEPS = int(os.environ.get("MITRA_STEPS", "0"))
MITRA_WARMUP = int(os.environ.get("MITRA_WARMUP", "0"))

BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("g", "i"), ("r", "z")]
IMPORTANT_COMBOS = sorted([("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")])


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr, df_val, df_te):
    def factorize_fit(s):
        codes, uniques = pd.factorize(s, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(s, uniques):
        cm = {cat: i for i, cat in enumerate(uniques)}
        return s.map(cm).fillna(-1).astype("int32")

    tr, va, te = df_tr.copy(), df_val.copy(), df_te.copy()
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            floored = np.floor(dset[col]).astype("float32")
            dset[cat_name] = pd.Series(factorize_transform(floored, uniques), index=dset.index).astype("int32").astype("category")
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        tr[bin_name] = pd.Series(kb.fit_transform(tr[["delta"]]).ravel().astype("int32"), index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            dset[bin_name] = pd.Series(kb.transform(dset[["delta"]]).ravel().astype("int32"), index=dset.index).astype("int32").astype("category")
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_names.append(combo_name)
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            dset[combo_name] = pd.Series(factorize_transform(combo_s, uniques), index=dset.index).astype("int32").astype("category")
    new_cat_cols = sorted([c for c in tr.columns if str(tr[c].dtype) == "category"])
    return tr, va, te, new_cat_cols, combo_names


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    try:
        enc = TargetEncoder(target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    except TypeError:
        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    tr_enc = enc.fit_transform(X_tr[combo_names], y_tr)
    val_enc = enc.transform(X_val[combo_names])
    tst_enc = enc.transform(X_te[combo_names])
    te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(N_CLASSES)]
    X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
    X_val[te_names] = np.asarray(val_enc, dtype="float32")
    X_te[te_names] = np.asarray(tst_enc, dtype="float32")
    return X_tr, X_val, X_te, te_names


def to_float_df(df, cols):
    out = {}
    for c in cols:
        s = df[c]
        out[c] = (s.astype("int32") if str(s.dtype) == "category" else s).to_numpy().astype(np.float32)
    return pd.DataFrame(out)


# ─── Load ─────────────────────────────────────────────────────────────────────
log(f"SMOKE={SMOKE} FOLD0_ONLY={FOLD0_ONLY} max_rows={MITRA_MAX_ROWS}")
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
y_all_str = train_raw[TARGET].astype(str).values
n_train, n_test = len(train_raw), len(test_raw)
log(f"  train={train_raw.shape} test={test_raw.shape} folds={len(folds_list)}")

keep_sm = None
if SMOKE:
    keep_sm = set(np.random.default_rng(0).choice(n_train, 20000, replace=False).tolist())
    folds_list = [folds_list[0]]
elif FOLD0_ONLY:
    folds_list = [folds_list[0]]

log("Applying stateless FE ...")
X_stateless = stateless_fe(train_raw.drop(columns=[IDC, TARGET]))
X_test_stateless = stateless_fe(test_raw.drop(columns=[IDC]))

# AutoGluon imported after data load (slow import)
from autogluon.tabular import TabularPredictor

oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores, feat_cols_final = [], None
n_folds_for_test = len(folds_list)
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    if SMOKE:
        tr_idx = np.array([i for i in tr_idx if i in keep_sm])
        val_idx = np.array([i for i in val_idx if i in keep_sm])
    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy())
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed)

    feat_cols = sorted(X_tr_fold.columns)
    if feat_cols_final is None:
        feat_cols_final = feat_cols
        log(f"  n_features={len(feat_cols)}")

    tr_df = to_float_df(X_tr_fold, feat_cols); tr_df[TARGET] = y_all_str[tr_idx]
    va_df = to_float_df(X_val_fold, feat_cols)
    te_df = to_float_df(X_te_fold, feat_cols)
    del X_tr_fold, X_val_fold, X_te_fold; gc.collect()

    # Mitra (via AG) caps input at ag.max_rows — it's a small-data model. Feed it a
    # proportional-stratified subsample as context; fine_tune adapts on it. AG holds
    # out 10% internally, so target SUB_TARGET so model.fit sees <= MITRA_MAX_ROWS.
    SUB_TARGET = int(MITRA_MAX_ROWS / 0.9) - 100
    if len(tr_df) > SUB_TARGET:
        frac = SUB_TARGET / len(tr_df)
        tr_df = (tr_df.groupby(TARGET, group_keys=False)
                 .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=fold_seed))
                 .reset_index(drop=True))
        log(f"  stratified-subsampled train to {len(tr_df)} (Mitra context cap)")

    ag_path = f"/tmp/ag_mitra_n134_fold{fold_id}"
    shutil.rmtree(ag_path, ignore_errors=True)
    log(f"  fitting Mitra (fine_tune) on {len(tr_df)} rows ...")
    t_fit0 = time.perf_counter()
    mitra_hp = {"fine_tune": True, "n_estimators": 1}
    if MITRA_STEPS > 0:
        mitra_hp["fine_tune_steps"] = MITRA_STEPS
    if MITRA_WARMUP > 0:
        mitra_hp["warmup_steps"] = MITRA_WARMUP
    log(f"  Mitra hp: {mitra_hp}")
    predictor = TabularPredictor(
        label=TARGET, problem_type="multiclass", eval_metric="balanced_accuracy",
        path=ag_path, verbosity=2,
    ).fit(
        tr_df,
        hyperparameters={"MITRA": mitra_hp},
        ag_args_fit={"ag.max_rows": MITRA_MAX_ROWS, "ag.max_memory_usage_ratio": 6.0},
        time_limit=FIT_TIME_LIMIT,
    )
    log(f"  fit done in {time.perf_counter()-t_fit0:.1f}s")

    def proba_in_order(df):
        p = predictor.predict_proba(df)
        return p.reindex(columns=CLASSES).to_numpy().astype(np.float32)

    t_pred0 = time.perf_counter()
    val_probs = proba_in_order(va_df)
    log(f"  val predict ({len(va_df)}) in {time.perf_counter()-t_pred0:.1f}s")
    oof_proba[val_idx] = val_probs
    if not (SMOKE or FOLD0_ONLY):
        test_proba_accum += proba_in_order(te_df) / n_folds_for_test

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_probs, axis=1))
    per_fold_scores.append(fold_score)
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f} elapsed={time.perf_counter()-fold_t0:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del predictor; shutil.rmtree(ag_path, ignore_errors=True); gc.collect()
    if fold_id == 0:
        ft = time.perf_counter() - fold_t0
        log(f"  TIMING: fold0={ft:.1f}s projected_5fold≈{ft*5:.1f}s ({ft*5/60:.1f}min)")

if SMOKE:
    log("[smoke] OK — AG+Mitra ran end-to-end. Exiting."); sys.exit(0)
if FOLD0_ONLY:
    # save fold-0 val OOF + indices so err-corr vs the bank can be computed without a full 5-fold
    f0_val_idx = np.asarray(folds_list[0]["val_idx"])
    np.save(NODE_DIR / "oof_fold0.npy", oof_proba[f0_val_idx])
    np.save(NODE_DIR / "oof_fold0_idx.npy", f0_val_idx)
    log(f"[fold0] saved oof_fold0.npy ({len(f0_val_idx)} rows)")
    log(f"[fold0] tier read: BA={per_fold_scores[0]:.6f} ({'CLEARS' if per_fold_scores[0]>=0.960 else 'BELOW'} 0.960 bar)")
    print(f"cv={per_fold_scores[0]:.6f}", flush=True); sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}"); print(f"cv={mean_cv:.6f}", flush=True)
np.save(NODE_DIR / "oof.npy", oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
(NODE_SRC / "features.txt").write_text("\n".join(feat_cols_final) + "\n")
log(f"OOF full BA={balanced_accuracy_score(y_all, oof_proba.argmax(1)):.6f}  total={time.perf_counter()-T0:.1f}s")
log("Done.")
