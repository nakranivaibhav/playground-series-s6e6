"""node_0051 — FT-Transformer on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0014:
  Swap feature-set fs_research → fs_realmlp_fe (the rich feature-set that lifted
  every NN base from ~0.949 to ~0.969). The FT-Transformer model architecture and
  training loop are kept from node_0014 (rtdl_revisiting_models.FTTransformer,
  depth=3, d_block=192, 8 heads). The FE/fold/OOF/test-emit scaffold is copied
  from node_0033 (TabM on fs_realmlp_fe).

FE pipeline (byte-identical to node_0033/node_0028):
  - Stateless FE: redshift ratios, 7 color pairs, mag_mean, mag_range,
    log1p_redshift, integer-floor categorical views of every base numeric.
  - fit_in_fold KBinsDiscretizer (delta 100/500 quantile bins) on train fold only.
  - fit_in_fold TargetEncoder (on combo cats) on train fold only.
  - fit_in_fold standardization (mean/std from train fold numerical features only).

FT-Transformer model:
  - rtdl_revisiting_models.FTTransformer (paper defaults: depth=3, d_block=192,
    8 heads, ReGLU FFN, att_dropout=0.2, ffn_dropout=0.1).
  - ALL features treated as numerical (FT-T has its own linear token embeddings;
    no PLR needed). Integer-floor cat codes + TE + delta bins passed as numerics.
  - Training: AdamW lr=1e-4, weight_decay=1e-5, CrossEntropyLoss with class weights.
  - Early stopping: internal 10% val split from train fold (not the OOF fold).

Leakage discipline:
  - Stateless FE: no target, no cross-row stats, no fitting — safe to compute once.
  - KBinsDiscretizer, factorize maps, TargetEncoder: fit on train-fold rows only.
  - Standardization (mean/std): fit on train-fold numerical features only.
  - Frozen folds.json used throughout; no refitting of folds.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from scipy.optimize import differential_evolution

import rtdl_revisiting_models as rtdl

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  rtdl_revisiting_models={rtdl.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"

# FT-Transformer hyperparameters (paper defaults)
D_BLOCK = 192
N_BLOCKS = 3
N_HEADS = 8
ATT_DROPOUT = 0.2
FFN_D_HIDDEN_MULT = 4 / 3
FFN_DROPOUT = 0.1
RESIDUAL_DROPOUT = 0.0
LR = 1e-4
WD = 1e-5

MAX_EPOCHS = 80 if not SMOKE else 4
PATIENCE = 12
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 16384   # FT-T is lighter than TabM (no k=32 ensemble)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_everything(SEED)

# ─── Feature engineering globals (byte-identical to node_0033) ───────────────
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
    """Pure row-wise / stateless FE — safe to apply to the full df before any fold split."""
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
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    fit_in_fold — called INSIDE the fold loop.
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
        for dset, dset_orig in [(va, df_val), (te, df_te)]:
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
    """TargetEncoder fit on train fold only (fit_in_fold), transform val and test."""
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


# ─── FT-Transformer model ─────────────────────────────────────────────────────

def build_ftt_model(n_num: int) -> rtdl.FTTransformer:
    """Build FT-Transformer with n_num numerical features, no categorical inputs.
    All features (including integer-coded cat codes) are treated as numerics for FT-T.
    """
    return rtdl.FTTransformer(
        n_cont_features=n_num,
        cat_cardinalities=[],      # all features passed as numeric
        d_out=N_CLASSES,
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        attention_n_heads=N_HEADS,
        attention_dropout=ATT_DROPOUT,
        ffn_d_hidden_multiplier=FFN_D_HIDDEN_MULT,
        ffn_dropout=FFN_DROPOUT,
        residual_dropout=RESIDUAL_DROPOUT,
    ).to(DEVICE)


def predict_proba_batch(model: rtdl.FTTransformer, Xn: np.ndarray,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    """Run inference in batches, returns (N, 3) probabilities."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(xn, None)   # FT-T with no cat features
            probs = torch.softmax(logits.float(), dim=-1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_ftt(
    Xn_tr: np.ndarray,
    y_tr: np.ndarray,
    fold_seed: int,
) -> rtdl.FTTransformer:
    """
    Train FT-Transformer. Uses internal 10% early-stop split from train fold (fit_in_fold).
    Returns best model.
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    model = build_ftt_model(Xn_tr.shape[1])

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    nt = len(ti)

    yv = y_tr[vi]
    Xn_vi = Xn_tr[vi]

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            y_b = y_t[idx]
            opt.zero_grad()
            logits = model(xn_b, None)   # (B, 3)
            loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        val_probs = predict_proba_batch(model, Xn_vi)
        ba = balanced_accuracy_score(yv, val_probs.argmax(1))
        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    FT-T early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model


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
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
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
num_cols_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

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

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    # FT-Transformer: treat ALL features as numeric (including int-coded cats)
    # This avoids OOM from one-hot expanding high-cardinality integer-floor features.
    all_features = sorted(X_tr_fold.columns)

    if num_cols_final is None:
        num_cols_final = all_features
        log(f"  n_features={len(all_features)} (all passed as numeric to FT-T)")

    Xn_tr = X_tr_fold[all_features].values.astype(np.float32)
    Xn_va = X_val_fold[all_features].values.astype(np.float32)
    Xn_te = X_te_fold[all_features].values.astype(np.float32)

    # Standardize all features — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Train FT-Transformer
    model = train_ftt(Xn_tr, y_tr_fold, fold_seed)

    # OOF predictions
    val_probs = predict_proba_batch(model, Xn_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions — average across folds
    test_probs_fold = predict_proba_batch(model, Xn_te)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

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

# ─── Write features.txt ───────────────────────────────────────────────────────
(NODE_SRC / "features.txt").write_text("\n".join(num_cols_final or []) + "\n")
log(f"Wrote features.txt ({len(num_cols_final or [])} features)")

# ─── Standalone DE-threshold balanced accuracy ────────────────────────────────
def score_fn(y_true, y_pred) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES) if (y_true == c).any()]
    ))


def best_thr_de(probs, labels) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                                maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


log("Computing DE-threshold fold-honest standalone CV ...")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
fval = [np.asarray(f["val_idx"]) for f in folds_data]

per_fold_de = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(oof_proba[other], y_all[other])
    pred = np.argmax(oof_proba[vi] * w, axis=1)
    s = score_fn(y_all[vi], pred)
    per_fold_de.append(s)
    log(f"  DE fold {i}: ba={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

de_mean = float(np.mean(per_fold_de))
de_sem = float(np.std(per_fold_de, ddof=1) / np.sqrt(len(per_fold_de)))
log(f"DE-threshold cv={de_mean:.6f}+/-{de_sem:.6f}")
print(f"cv_de={de_mean:.6f}", flush=True)

# ─── Re-stack A/B: CORE15 + node_0051 ────────────────────────────────────────
log("Re-stack A/B: CORE15 + node_0051 ...")
from sklearn.linear_model import LogisticRegression

BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]

nodes_dir = COMP_DIR / "nodes"


def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


OOF_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
)
TEST_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
)

# node_0051 log-probs (3-class)
n51_oof = logp(oof_proba)
n51_test = logp(test_proba_accum)

OOF_STACK = np.concatenate([OOF_CORE, n51_oof], axis=1)
TEST_STACK = np.concatenate([TEST_CORE, n51_test], axis=1)
log(f"  stacked OOF={OOF_STACK.shape}  TEST={TEST_STACK.shape}  ({len(BASES)+1} bases)")


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


# Fold-honest stacked OOF
stack_oof = np.zeros((n_train, N_CLASSES))
for vi in fval:
    tr = np.setdiff1d(np.arange(n_train), vi)
    m = fit_meta(OOF_STACK[tr], y_all[tr])
    stack_oof[vi] = m.predict_proba(OOF_STACK[vi])

# Fold-honest DE threshold scoring
per_fold_stack = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(stack_oof[other], y_all[other])
    pred = np.argmax(stack_oof[vi] * w, axis=1)
    s = score_fn(y_all[vi], pred)
    per_fold_stack.append(s)
    log(f"  stack fold {i}: ba={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

stack_mean = float(np.mean(per_fold_stack))
stack_sem = float(np.std(per_fold_stack, ddof=1) / np.sqrt(len(per_fold_stack)))
log(f"re-stack cv={stack_mean:.6f}+/-{stack_sem:.6f}  (champ=0.969808)")
print(f"cv_restack={stack_mean:.6f}", flush=True)

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
