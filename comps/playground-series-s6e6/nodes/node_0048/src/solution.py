"""node_0048 — Optuna XGB to stacked-OOF objective.

Built on: node_0031 (XGBoost on fs_realmlp_fe, cv 0.966244).
THE ONE ATOMIC CHANGE: Optuna HPO of XGBoost params, but the objective
is the STACKED-OOF balanced accuracy when this base replaces node_0031
in CORE15, NOT the solo base CV.

Each Optuna trial:
1. Train 5-fold XGB on fs_realmlp_fe with the trial's params.
2. Swap this trial's OOF into CORE15 (replacing node_0031 column).
3. Fit fold-honest balanced-LogReg stack + DE threshold over the new OOF.
4. Return negated stacked balanced accuracy.

CORE15 = [node_0006,node_0004,node_0001,node_0009,node_0011,node_0003,
          node_0019,node_0016,node_0014,node_0028,node_0032,node_0035,
          node_0033,node_0030,node_0039].

After Optuna: re-run final fold with best params, emit oof.npy,
test_probs.npy, submission.csv, best_params.json.

Leakage discipline (identical to node_0031):
- Stateless FE computed once on full frame — no target, no cross-row stats.
- KBinsDiscretizer, factorize maps, TargetEncoder — fit on train-fold only.
- XGBoost — fit on train-fold only with early-stopping on val-fold.
- Frozen folds.json throughout.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
while not (REPO_ROOT / "tools" / "leakage_scan.py").exists():
    REPO_ROOT = REPO_ROOT.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# --- Constants ---
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# CORE15 bases — node_0031 is the one we REPLACE
CORE15 = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030", "node_0039",
]
REPLACE_NODE = "node_0031"

# Time-box: cap Optuna wall-clock
MAX_OPTUNA_SECONDS = 65 * 60   # 65 min; we'll stop after this
EARLY_STOPPING_ROUNDS = 25

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


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(SEED)


# ---------------------------------------------------------------------------
# Feature engineering (identical to node_0031)
# ---------------------------------------------------------------------------

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
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_tr in [(va, df_val), (te, df_te)]:
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
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
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


def df_to_xgb_matrix(df, cat_cols, label=None, sample_weight=None):
    df_copy = df.copy()
    for col in cat_cols:
        if str(df_copy[col].dtype) == "category":
            df_copy[col] = df_copy[col].cat.codes.astype("int32")
        else:
            df_copy[col] = df_copy[col].astype("int32")
    for col in df_copy.columns:
        if col not in cat_cols:
            df_copy[col] = df_copy[col].astype("float32")
    kwargs = {}
    if label is not None:
        kwargs["label"] = label
    if sample_weight is not None:
        kwargs["weight"] = sample_weight
    return xgb.DMatrix(df_copy, **kwargs)


# ---------------------------------------------------------------------------
# Stacker helpers (from node_0041)
# ---------------------------------------------------------------------------

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def fit_meta(Xtr, ytr):
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def score_fn(y_true, y_pred):
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES) if (y_true == c).any()]
    ))


def best_thr_de(probs, labels):
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                               maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


def compute_stacked_cv(trial_oof: np.ndarray, base_oof_others: np.ndarray,
                       y_all: np.ndarray, fval: list) -> float:
    """Compute fold-honest stacked balanced accuracy replacing node_0031 with trial_oof."""
    n = len(y_all)
    nc = N_CLASSES
    # Build combined OOF: trial_oof log-probs + other 14 bases log-probs
    combined = np.concatenate([logp(trial_oof), base_oof_others], axis=1)

    stack_oof = np.zeros((n, nc))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        m = fit_meta(combined[tr], y_all[tr])
        stack_oof[vi] = m.predict_proba(combined[vi])

    per_fold_scores = []
    for vi in fval:
        other = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack_oof[other], y_all[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        per_fold_scores.append(score_fn(y_all[vi], pred))

    return float(np.mean(per_fold_scores))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)
fval = [np.asarray(fi["val_idx"]) for fi in folds_list]

# Stateless FE (computed once, safe)
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)

# ---------------------------------------------------------------------------
# Load CORE15 OOF (minus node_0031 which we replace) — loaded ONCE
# ---------------------------------------------------------------------------
log("Loading CORE15 OOF (14 bases, excluding node_0031) ...")
nodes_dir = COMP_DIR / "nodes"
other_bases = [b for b in CORE15 if b != REPLACE_NODE]
base_oof_others = np.concatenate(
    [logp(np.load(nodes_dir / b / "oof.npy")) for b in other_bases], axis=1
)
log(f"  base_oof_others shape={base_oof_others.shape} ({len(other_bases)} bases)")

# Pre-load test arrays for other bases (used only in final pass)
base_test_others = np.concatenate(
    [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in other_bases], axis=1
)
log(f"  base_test_others shape={base_test_others.shape}")


# ---------------------------------------------------------------------------
# OOF training helper (produces trial OOF + optionally test probs)
# ---------------------------------------------------------------------------
_num_cols_cache = None
_cat_cols_cache = None


def train_xgb_oof(params: dict, n_estimators: int, produce_test: bool = False):
    """
    Train XGB over the 5 folds, return (oof_proba, test_proba_accum, cat_cols, num_cols).
    If produce_test=False, test_proba_accum is None (saves time during search).
    """
    global _num_cols_cache, _cat_cols_cache

    oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
    test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32) if produce_test else None
    cat_cols_final = None
    num_cols_final = None

    for fi in folds_list:
        fold_id = fi["fold"]
        val_idx = np.asarray(fi["val_idx"])
        tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
        fold_seed = SEED + (fold_id + 1) * 100
        seed_everything(fold_seed)

        X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, _ = fit_fold_categoricals(
            X_stateless.iloc[tr_idx].reset_index(drop=True),
            X_stateless.iloc[val_idx].reset_index(drop=True),
            X_test_stateless.copy(),
        )
        y_tr_fold = y_all[tr_idx]
        y_val_fold = y_all[val_idx]
        X_tr_fold, X_val_fold, X_te_fold, _ = add_target_encoding(
            X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
        )
        X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
        X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
        X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)
        cat_cols_sorted = sorted(cat_cols)
        if cat_cols_final is None:
            cat_cols_final = cat_cols_sorted
            num_cols_final = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr_fold)
        dtrain = df_to_xgb_matrix(X_tr_fold, cat_cols_sorted, label=y_tr_fold, sample_weight=sample_weights)
        dval = df_to_xgb_matrix(X_val_fold, cat_cols_sorted, label=y_val_fold)
        dtest = df_to_xgb_matrix(X_te_fold, cat_cols_sorted) if produce_test else None

        xgb_params_fold = {**params, "seed": fold_seed}
        callbacks = [xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS,
                                                 metric_name="mlogloss",
                                                 save_best=True, maximize=False)]
        booster = xgb.train(
            params=xgb_params_fold,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            callbacks=callbacks,
            verbose_eval=False,
        )
        val_proba = booster.predict(dval).reshape(-1, N_CLASSES).astype("float32")
        oof_proba[val_idx] = val_proba
        if produce_test and dtest is not None:
            test_proba_accum += booster.predict(dtest).reshape(-1, N_CLASSES).astype("float32") / len(folds_list)

        del booster, dtrain, dval
        if dtest is not None:
            del dtest
        gc.collect()

    if cat_cols_final is not None:
        _cat_cols_cache = cat_cols_final
        _num_cols_cache = num_cols_final

    return oof_proba, test_proba_accum, cat_cols_final, num_cols_final


# ---------------------------------------------------------------------------
# Time one trial to estimate how many we can run
# ---------------------------------------------------------------------------
log("Timing one trial to size Optuna budget ...")
timing_params = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 1e-8,
    "gamma": 0.0,
    "verbosity": 0,
    "nthread": -1,
}
t_timing = time.perf_counter()
_oof_timing, _, _, _ = train_xgb_oof(timing_params, n_estimators=2000, produce_test=False)
trial_time_xgb = time.perf_counter() - t_timing
_solo_timing = balanced_accuracy_score(y_all, _oof_timing.argmax(1))
log(f"  XGB 5-fold timing: {trial_time_xgb:.1f}s  solo_cv={_solo_timing:.6f}")

# Stack eval time (estimate ~2-5s)
t_stack = time.perf_counter()
_stack_timing = compute_stacked_cv(_oof_timing, base_oof_others, y_all, fval)
stack_time = time.perf_counter() - t_stack
log(f"  Stack eval timing: {stack_time:.1f}s  stacked_cv={_stack_timing:.6f}")

trial_total = trial_time_xgb + stack_time + 1.0
budget_seconds = MAX_OPTUNA_SECONDS - (time.perf_counter() - T0)
n_trials = max(5, int(budget_seconds / trial_total))
log(f"  budget_left={budget_seconds:.0f}s  trial_total={trial_total:.1f}s  n_trials={n_trials}")


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
trial_results = []  # (trial_num, solo_cv, stacked_cv, params, n_est)
optuna_t0 = time.perf_counter()


def objective(trial: optuna.Trial) -> float:
    # Check time budget
    elapsed = time.perf_counter() - optuna_t0
    if elapsed > (budget_seconds - trial_total):
        raise optuna.exceptions.TrialPruned()

    params = {
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "nthread": -1,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    n_est = trial.suggest_int("n_estimators", 300, 3000)

    trial_oof, _, _, _ = train_xgb_oof(params, n_estimators=n_est, produce_test=False)
    solo_cv = balanced_accuracy_score(y_all, trial_oof.argmax(1))
    stacked = compute_stacked_cv(trial_oof, base_oof_others, y_all, fval)

    elapsed = time.perf_counter() - optuna_t0
    log(f"  Trial {trial.number}: solo={solo_cv:.6f}  stacked={stacked:.6f}  elapsed={elapsed:.0f}s")
    trial_results.append((trial.number, solo_cv, stacked, params, n_est))

    return stacked  # maximize


# ---------------------------------------------------------------------------
# Run Optuna
# ---------------------------------------------------------------------------
log(f"Starting Optuna: n_trials={n_trials}, time-box={budget_seconds:.0f}s ...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))

# Seed with the timing-trial params (already know it works)
study.enqueue_trial({
    "max_depth": 6, "learning_rate": 0.3, "subsample": 0.8,
    "colsample_bytree": 0.8, "colsample_bylevel": 0.8,
    "min_child_weight": 5, "gamma": 0.0, "reg_alpha": 1e-8,
    "reg_lambda": 1.0, "n_estimators": 2000,
})

study.optimize(
    objective,
    n_trials=n_trials,
    timeout=budget_seconds,
    catch=(Exception,),
    show_progress_bar=False,
)

best_trial = study.best_trial
log(f"Optuna done. Best stacked CV={best_trial.value:.6f} (trial {best_trial.number})")

# Save trial curve
trial_curve = []
for t in study.trials:
    if t.value is not None:
        matching = [r for r in trial_results if r[0] == t.number]
        if matching:
            _, solo_cv, stacked_cv_val, _, _ = matching[0]
            trial_curve.append({"trial": t.number, "solo_cv": solo_cv, "stacked_cv": stacked_cv_val})
(NODE_DIR / "trial_curve.json").write_text(json.dumps(trial_curve, indent=2))
log(f"Saved trial_curve.json ({len(trial_curve)} trials)")

# ---------------------------------------------------------------------------
# Re-run final fold with best params (produce test predictions too)
# ---------------------------------------------------------------------------
log("Re-running 5-fold with best params to produce final artifacts ...")
best_params_xgb = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
    "nthread": -1,
    "max_depth": best_trial.params["max_depth"],
    "learning_rate": best_trial.params["learning_rate"],
    "subsample": best_trial.params["subsample"],
    "colsample_bytree": best_trial.params["colsample_bytree"],
    "colsample_bylevel": best_trial.params["colsample_bylevel"],
    "min_child_weight": best_trial.params["min_child_weight"],
    "gamma": best_trial.params["gamma"],
    "reg_alpha": best_trial.params["reg_alpha"],
    "reg_lambda": best_trial.params["reg_lambda"],
}
best_n_est = best_trial.params["n_estimators"]

oof_proba, test_proba_accum, cat_cols_final, num_cols_final = train_xgb_oof(
    best_params_xgb, n_estimators=best_n_est, produce_test=True
)

# Per-fold scores (solo)
per_fold_scores = []
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    fold_score = balanced_accuracy_score(y_all[val_idx], oof_proba[val_idx].argmax(1))
    per_fold_scores.append(fold_score)
    log(f"  fold {fi['fold']}: solo_ba={fold_score:.6f}")
    print(f"fold{fi['fold']}_score={fold_score:.6f}", flush=True)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"Solo CV = {mean_cv:.6f} +/- {sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# Final stacked CV with best OOF
final_stacked = compute_stacked_cv(oof_proba, base_oof_others, y_all, fval)
log(f"Stacked CV (best trial OOF in CORE15) = {final_stacked:.6f}")
print(f"stacked_cv={final_stacked:.6f}", flush=True)

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# Submission
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")

# Best params JSON
best_params_out = {**best_trial.params, "best_stacked_cv": best_trial.value,
                   "solo_cv_mean": mean_cv, "solo_cv_sem": sem_cv}
(NODE_DIR / "best_params.json").write_text(json.dumps(best_params_out, indent=2))
log("Saved best_params.json")

# features.txt
if cat_cols_final is None:
    cat_cols_final = _cat_cols_cache
    num_cols_final = _num_cols_cache
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# Final OOF metric
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy (solo) = {oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
