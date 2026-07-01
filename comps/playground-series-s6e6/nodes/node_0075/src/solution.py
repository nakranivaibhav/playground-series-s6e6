"""node_0075 — 5-seed bagged TabM on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0033:
  Bag TabM-richFE over 5 seeds (seeds 42,123,456,789,999).
  Per fold: train 5 seeds, average their probabilities → one stronger OOF.
  Everything else byte-identical to node_0033.

INTERIM KILL: after seed-2 (seeds 42 and 123), compute 2-seed-avg solo CV.
  If < 0.968353 (n33 + 0.0003), stop seeds 3-5 and record the wash.
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
from sklearn.utils.class_weight import compute_class_weight

import tabm
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

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
# Bag seeds — 5 seeds; seed-2 interim kill check
BAG_SEEDS = [42, 123, 456, 789, 999]
INTERIM_KILL_THRESHOLD = 0.968353   # n33 cv 0.968053 + 0.0003
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  tabm={tabm.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"

# TabM hyperparameters (byte-identical to n33)
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


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


def build_tabm_model(n_num: int, cat_cards: list[int], bins: list) -> tabm.TabM:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    model = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=N_CLASSES,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    return model.to(DEVICE)


def predict_proba_batch(model: tabm.TabM, Xn: np.ndarray, Xc: np.ndarray | None,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            logits = model(xn, xc)
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
) -> tuple[tabm.TabM, np.ndarray]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    nt = len(ti)

    yv = y_tr[vi]
    Xn_vi = Xn_tr[vi]
    Xc_vi = Xc_tr[vi] if Xc_tr is not None else None

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_t[idx] if Xc_t is not None else None
            y_b = y_t[idx]
            opt.zero_grad()
            logits = model(xn_b, xc_b)
            b, k, c = logits.shape
            loss = loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        val_probs = predict_proba_batch(model, Xn_vi, Xc_vi)
        ba = balanced_accuracy_score(yv, val_probs.argmax(1))
        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    TabM early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model, bins


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
    log("SMOKE MODE: subsample to 30000 rows, 1 fold, 1 seed")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]
    BAG_SEEDS = BAG_SEEDS[:1]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Pre-flight leakage checks (before training)
assert TARGET not in X_stateless.columns, "TARGET leaked into features"
assert IDC not in X_stateless.columns, "ID leaked into features"
log("Pre-flight checks: TARGET and ID absent from feature matrix -- OK")

# ─── Seed-bag OOF loop ────────────────────────────────────────────────────────
# oof_by_seed[seed_idx] = np.zeros((n_train, N_CLASSES))
# test_by_seed[seed_idx] = np.zeros((n_test, N_CLASSES))
oof_by_seed = []
test_by_seed = []
seeds_run = []
per_seed_cv = []

killed = False

for seed_idx, SEED in enumerate(BAG_SEEDS):
    log(f"\n{'='*60}")
    log(f"SEED {seed_idx+1}/{len(BAG_SEEDS)}: {SEED}")
    log(f"{'='*60}")

    seed_everything(SEED)

    oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
    test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    per_fold_scores = []
    cat_cols_final = None
    num_cols_final = None

    fold_t0 = time.perf_counter()

    for fi in folds_list:
        fold_id = fi["fold"]
        val_idx = np.asarray(fi["val_idx"])
        tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
        # fold_seed combines global seed and fold id
        fold_seed = SEED + (fold_id + 1) * 100
        seed_everything(fold_seed)

        if SMOKE:
            keep_set = set(keep_sm.tolist())
            tr_idx = np.array([i for i in tr_idx if i in keep_set])
            val_idx = np.array([i for i in val_idx if i in keep_set])

        log(f"  Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

        X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
            X_stateless.iloc[tr_idx].reset_index(drop=True),
            X_stateless.iloc[val_idx].reset_index(drop=True),
            X_test_stateless.copy(),
        )

        y_tr_fold = y_all[tr_idx]
        y_val_fold = y_all[val_idx]
        X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
            X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
        )

        X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
        X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
        X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

        cat_cols_sorted = sorted(cat_cols)
        TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
        all_cols_sorted = sorted(X_tr_fold.columns)
        num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

        if cat_cols_final is None:
            cat_cols_final = cat_cols_sorted
            num_cols_final = num_for_tabm
            log(f"    n_features={X_tr_fold.shape[1]}  tabm_cat={len(TABM_CAT_COLS)}  tabm_num={len(num_for_tabm)}")

        Xn_tr = X_tr_fold[num_for_tabm].values.astype(np.float32)
        Xn_va = X_val_fold[num_for_tabm].values.astype(np.float32)
        Xn_te = X_te_fold[num_for_tabm].values.astype(np.float32)

        if TABM_CAT_COLS:
            Xc_tr = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
            Xc_va = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
            Xc_te = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
            cat_cards = (Xc_tr.max(axis=0) + 2).tolist()
            card_arr = np.array(cat_cards) - 1
            Xc_tr = np.clip(Xc_tr, 0, card_arr)
            Xc_va = np.clip(Xc_va, 0, card_arr)
            Xc_te = np.clip(Xc_te, 0, card_arr)
        else:
            Xc_tr = Xc_va = Xc_te = None
            cat_cards = []

        mu = Xn_tr.mean(0)
        sd = Xn_tr.std(0) + 1e-8
        Xn_tr = (Xn_tr - mu) / sd
        Xn_va = (Xn_va - mu) / sd
        Xn_te = (Xn_te - mu) / sd

        model, bins = train_tabm(Xn_tr, Xc_tr, y_tr_fold, cat_cards, fold_seed)

        val_probs = predict_proba_batch(model, Xn_va, Xc_va)
        oof_proba[val_idx] = val_probs.astype(np.float32)

        test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
        test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

        fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
        per_fold_scores.append(fold_score)
        fold_elapsed = time.perf_counter() - fold_t0
        log(f"    fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
        print(f"seed{seed_idx+1}_fold{fold_id}_score={fold_score:.6f}", flush=True)

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            log(f"    peak VRAM so far: {vram_gb:.2f} GB")

        del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if seed_idx == 0 and fold_id == 0:
            fold_time = time.perf_counter() - fold_t0
            projected_per_seed = fold_time * len(folds_list)
            projected_total = projected_per_seed * len(BAG_SEEDS)
            log(f"    TIMING: fold0={fold_time:.1f}s  projected_per_seed={projected_per_seed:.1f}s  "
                f"projected_5seeds={projected_total:.1f}s ({projected_total/60:.1f}min)")

    seed_cv = float(np.mean(per_fold_scores))
    seed_sem = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    log(f"Seed {SEED}: per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
    log(f"Seed {SEED}: cv={seed_cv:.6f}+/-{seed_sem:.6f}")
    print(f"seed{seed_idx+1}_cv={seed_cv:.6f}", flush=True)
    per_seed_cv.append(seed_cv)

    oof_by_seed.append(oof_proba.copy())
    test_by_seed.append(test_proba_accum.copy())
    seeds_run.append(SEED)

    # ─── INTERIM KILL CHECK after seed-2 ─────────────────────────────────────
    if seed_idx == 1:  # after 2nd seed (0-indexed)
        avg_oof_2 = np.mean(np.stack(oof_by_seed[:2], axis=0), axis=0)
        avg_test_2 = np.mean(np.stack(test_by_seed[:2], axis=0), axis=0)
        fold_scores_2 = []
        for fi in folds_list:
            val_idx = np.asarray(fi["val_idx"])
            fold_scores_2.append(
                balanced_accuracy_score(y_all[val_idx], avg_oof_2[val_idx].argmax(1))
            )
        interim_cv = float(np.mean(fold_scores_2))
        log(f"\nINTERIM KILL CHECK (2-seed avg): cv={interim_cv:.6f}  threshold={INTERIM_KILL_THRESHOLD}")
        print(f"interim_2seed_cv={interim_cv:.6f}", flush=True)
        if interim_cv < INTERIM_KILL_THRESHOLD and not SMOKE:
            log(f"KILL TRIGGERED: 2-seed avg {interim_cv:.6f} < {INTERIM_KILL_THRESHOLD}. "
                f"Stopping seeds 3-5. Recording wash.")
            killed = True
            break

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

n_seeds = len(seeds_run)
log(f"\nSeeds run: {seeds_run} (n={n_seeds})")

# Average OOF and test probs across seeds
oof_avg = np.mean(np.stack(oof_by_seed, axis=0), axis=0)
test_avg = np.mean(np.stack(test_by_seed, axis=0), axis=0)

# Final fold scores on avg OOF
final_fold_scores = []
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    fs = balanced_accuracy_score(y_all[val_idx], oof_avg[val_idx].argmax(1))
    final_fold_scores.append(fs)

mean_cv = float(np.mean(final_fold_scores))
sem_cv = float(np.std(final_fold_scores, ddof=1) / np.sqrt(len(final_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in final_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

if killed:
    log(f"NOTE: KILL applied after 2 seeds. Final cv based on 2-seed average.")
    print(f"kill_applied=True  seeds_run={n_seeds}", flush=True)

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_avg)
log(f"Saved oof.npy shape={oof_avg.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_avg)
log(f"Saved test_probs.npy shape={test_avg.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_avg.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_avg.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
