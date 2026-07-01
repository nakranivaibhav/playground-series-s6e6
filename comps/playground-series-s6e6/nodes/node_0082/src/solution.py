"""node_0082 — NODE (Neural Oblivious Decision Ensembles) on fs_realmlp_fe.

Uses pytorch-tabular's NODEModel (library impl per hard-rule 8).
Architecture: 2 layers × 2048 trees depth-6, entmax15/entmoid15, AdamW.
Feature pipeline is byte-identical to node_0033 fs_realmlp_fe loader:
  - Stateless FE (safe to compute once on full df)
  - fit_in_fold: KBinsDiscretizer, TargetEncoder, standardization

CHEAP KILL: if fold-0 BA < 0.9665, stop and exit with status dead.
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
import torch
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


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {DEVICE}  pytorch-tabular NODE")

SMOKE = os.environ.get("NODE_SMOKE") == "1"

# NODE hyperparameters
NUM_LAYERS = 2
NUM_TREES = 512  # 1024 too slow (~237s/epoch with 40 feats); 512→20s/epoch, ~51min 5-fold
DEPTH = 6
CHOICE_FN = "entmax15"
BIN_FN = "entmoid15"
MAX_EPOCHS = 30 if not SMOKE else 2
BATCH_SIZE = 1024  # safe for 32GB GPU with 1024 trees depth-6 (bs=2048 OOMs)
PATIENCE = 5  # early stopping patience (epochs); ~47s/epoch → 30 epochs max = 24min/fold

KILL_THRESHOLD = 0.9665  # fold-0 BA kill criterion


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

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
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

    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32")
        for dset, dset_tr in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32")

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32")

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

    new_cat_cols = []  # we treat all as continuous for NODE
    return tr, va, te, new_cat_cols, combo_names, local_map


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names: list, fold_seed: int):
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

if SMOKE:
    log("SMOKE MODE")
    folds_list = [folds_list[0]]

# ─── Stateless FE ────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # fit_in_fold categorical encoding
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # fit_in_fold target encoding
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    all_cols = sorted(X_tr_fold.columns)
    X_tr_fold = X_tr_fold[all_cols]
    X_val_fold = X_val_fold[all_cols]
    X_te_fold = X_te_fold[all_cols]

    # For NODE: treat ALL features as continuous (avoids OOM from high-card cat embeddings)
    # Convert everything to float32
    Xn_tr = X_tr_fold.astype(np.float32)
    Xn_va = X_val_fold.astype(np.float32)
    Xn_te = X_te_fold.astype(np.float32)

    # fit_in_fold standardization
    mu = Xn_tr.values.mean(0)
    sd = Xn_tr.values.std(0) + 1e-8
    Xn_tr = pd.DataFrame((Xn_tr.values - mu) / sd, columns=all_cols)
    Xn_va = pd.DataFrame((Xn_va.values - mu) / sd, columns=all_cols)
    Xn_te = pd.DataFrame((Xn_te.values - mu) / sd, columns=all_cols)

    if fold_id == 0:
        log(f"  n_features={Xn_tr.shape[1]}")

    # Build pytorch-tabular DataFrames with target column
    y_tr_labels = pd.Series([INV_MAP[v] for v in y_tr_fold], name=TARGET)
    y_va_labels = pd.Series([INV_MAP[v] for v in y_val_fold], name=TARGET)

    df_train_pt = Xn_tr.copy()
    df_train_pt[TARGET] = y_tr_labels.values

    df_val_pt = Xn_va.copy()
    df_val_pt[TARGET] = y_va_labels.values

    df_test_pt = Xn_te.copy()

    continuous_cols = all_cols

    # ─── pytorch-tabular NODE ──────────────────────────────────────────────────
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import NodeConfig

    # Use a temp dir per fold for checkpoints
    ckpt_dir = NODE_DIR / f"_ckpt_fold{fold_id}"
    ckpt_dir.mkdir(exist_ok=True)

    data_config = DataConfig(
        target=[TARGET],
        continuous_cols=continuous_cols,
        categorical_cols=[],
        normalize_continuous_features=False,  # we already standardized
        num_workers=0,  # multiprocessing with large DataFrame is slow
    )

    model_config = NodeConfig(
        task="classification",
        num_layers=NUM_LAYERS,
        num_trees=NUM_TREES,
        depth=DEPTH,
        choice_function=CHOICE_FN,
        bin_function=BIN_FN,
        input_dropout=0.0,
        additional_tree_output_dim=N_CLASSES,
        learning_rate=1e-3,
    )

    trainer_config = TrainerConfig(
        batch_size=BATCH_SIZE,
        data_aware_init_batch_size=256,  # reduce from 2000 default to save VRAM during init
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        early_stopping="valid_loss",
        early_stopping_patience=PATIENCE,
        early_stopping_mode="min",
        checkpoints="valid_loss",
        checkpoints_path=str(ckpt_dir),
        checkpoints_mode="min",
        load_best=True,
        progress_bar="none",
        precision="32",
        seed=fold_seed,
    )

    optimizer_config = OptimizerConfig(
        optimizer="AdamW",
        optimizer_params={"weight_decay": 1e-4},
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_params={"T_max": MAX_EPOCHS, "eta_min": 1e-5},
        lr_scheduler_monitor_metric="valid_loss",
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False,
        suppress_lightning_logger=True,
    )

    tabular_model.fit(
        train=df_train_pt,
        validation=df_val_pt,
        seed=fold_seed,
    )

    # Predict val
    val_pred_df = tabular_model.predict(df_val_pt.drop(columns=[TARGET]))
    # pytorch-tabular returns probability columns named: <TARGET>_<class>_probability
    prob_cols = [f"{TARGET}_{c}_probability" for c in CLASSES]
    # Check if columns exist
    available = val_pred_df.columns.tolist()
    if not all(c in available for c in prob_cols):
        # Try alternate naming
        prob_cols_alt = [c for c in available if "probability" in c]
        log(f"  WARNING: expected prob cols not found. Available: {prob_cols_alt}")
        # Sort to match CLASSES order
        prob_cols = sorted(prob_cols_alt)

    val_probs = val_pred_df[prob_cols].values.astype(np.float32)
    oof_proba[val_idx] = val_probs

    # Predict test
    test_pred_df = tabular_model.predict(df_test_pt)
    test_probs_fold = test_pred_df[prob_cols].values.astype(np.float32)
    test_proba_accum += test_probs_fold / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_probs, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    # Cleanup checkpoint dir
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    del tabular_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

        # CHEAP KILL
        if fold_score < KILL_THRESHOLD and not SMOKE:
            log(f"KILL CRITERION TRIPPED: fold-0 BA={fold_score:.6f} < {KILL_THRESHOLD}")
            log("Stopping — marking dead. Not training remaining folds.")
            print(f"KILL: fold0_BA={fold_score:.6f} < {KILL_THRESHOLD}", flush=True)
            sys.exit(2)

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
