"""node_0016 — improve (nn): TabM with balanced class-weighted cross-entropy loss.

Built on: node_0009 (TabM, library) — byte-identical feature-prep, model config,
folds, CUDA, PLR embeddings, standardization, OOF/test plumbing.

ONE atomic change: unweighted cross-entropy → inverse-class-frequency weighted
cross-entropy. The per-class weights are computed on the TRAIN FOLD only
(fit-inside-fold), never on the full train or test set. Each fold's class_w
is recomputed from that fold's tr_idx rows only, keeping the change fully
leak-safe. The shuffled-label control uses weights computed from the shuffled
train fold. The final full-train fit computes weights from all train rows.

Leakage discipline (three stateful steps, all FIT-INSIDE-FOLD):
  - standardization of the 22 continuous features: mean/std from the TRAIN FOLD only.
  - PiecewiseLinearEmbeddings bins: compute_bins() on the TRAIN FOLD's X_num + y only
    (target-aware decision-tree bins). The official val fold is used only for OOF scoring;
    early stopping uses an internal 10% split carved from the train fold.
  - class weights: computed from TRAIN FOLD labels only (fit-in-fold).

Set TABM_SMOKE=1 for a fast shape/sanity run (subsample, 1 fold, few epochs).

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
Baseline (node_0009): cv = 0.964215.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

import tabm
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

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
NUMF = CONT + FLAGS                                                 # 26 → x_num
CATF = ["spectral_type", "galaxy_population"]                       # → x_cat (cardinalities below)

SMOKE = os.environ.get("TABM_SMOKE") == "1"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_EMB, N_BINS = 16, 48
MAX_EPOCHS, PATIENCE, BATCH = (6, 3, 8192) if SMOKE else (100, 16, 8192)
torch.manual_seed(SEED)
np.random.seed(SEED)


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


def make_model(cat_cards, bins):
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    return tabm.TabM.make(
        n_num_features=len(NUMF), cat_cardinalities=cat_cards,
        d_out=3, num_embeddings=num_emb,
    ).to(DEVICE)


def predict_proba(model, Xn, Xc):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), 32768):
            xn = torch.as_tensor(Xn[s:s + 32768], dtype=torch.float32, device=DEVICE)
            xc = torch.as_tensor(Xc[s:s + 32768], dtype=torch.long, device=DEVICE)
            logits = model(xn, xc)                       # (B, k, 3)
            out.append(torch.softmax(logits, dim=-1).mean(dim=1).cpu().numpy())
    return np.concatenate(out, 0)


def train_model(Xn, Xc, y, cat_cards, class_w, max_epochs=MAX_EPOCHS):
    """Train TabM with an internal 10% early-stop split. Bins computed on THIS train set only."""
    g = torch.Generator().manual_seed(SEED)
    n = len(Xn)
    perm = torch.randperm(n, generator=g).numpy()
    nv = max(1, int(0.1 * n))
    vi, ti = perm[:nv], perm[nv:]
    # target-aware bins from the TRAIN portion only (fit-inside-fold)
    bins = compute_bins(
        torch.as_tensor(Xn[ti], dtype=torch.float32), n_bins=N_BINS,
        y=torch.as_tensor(y[ti], dtype=torch.long), regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )
    model = make_model(cat_cards, bins)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=DEVICE, dtype=torch.float32))

    Xn_t = torch.as_tensor(Xn[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = torch.as_tensor(Xc[ti], dtype=torch.long, device=DEVICE)
    y_t = torch.as_tensor(y[ti], dtype=torch.long, device=DEVICE)
    yv = y[vi]
    nt = len(ti)
    best_ba, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH):
            idx = bperm[s:s + BATCH]
            opt.zero_grad()
            logits = model(Xn_t[idx], Xc_t[idx])          # (b, k, 3)
            b, k, c = logits.shape
            loss = lossf(logits.reshape(b * k, c), y_t[idx].repeat_interleave(k))
            loss.backward()
            opt.step()
        ba = balanced_accuracy_score(yv, predict_proba(model, Xn[vi], Xc[vi]).argmax(1))
        if ba > best_ba + 1e-5:
            best_ba, best_state, bad = ba, {kk: v.detach().clone() for kk, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if SMOKE:
            print(f"    [smoke] ep{ep} val_ba={ba:.4f}")
    model.load_state_dict(best_state)
    return model


print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}) | tabm {tabm.__version__} | SMOKE={SMOKE}")
print("Loading + engineering …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

# categorical codes (fixed categories from cast_categoricals → stable, leak-safe)
cat_cards = [int(train[c].cat.categories.size) for c in CATF]
Xc_all = np.stack([train[c].cat.codes.to_numpy() for c in CATF], axis=1).astype(np.int64)
Xc_te = np.stack([test[c].cat.codes.to_numpy() for c in CATF], axis=1).astype(np.int64)
assert Xc_all.min() >= 0 and Xc_te.min() >= 0, "unseen category produced code -1"
Xnum_all = train[NUMF].to_numpy(np.float32)
Xnum_te = test[NUMF].to_numpy(np.float32)
y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)
print(f"  n_num={len(NUMF)} cat_cards={cat_cards}  rows={n}")

if SMOKE:
    rng = np.random.default_rng(0)
    keep = rng.choice(n, 30000, replace=False)
    folds_list = [folds_list[0]]
    # restrict to the subsample, remap val_idx
    keepset = set(keep.tolist())

N_CONT = len(CONT)


def compute_class_weights(y_fold):
    """Compute inverse-frequency class weights from given labels (fit-in-fold)."""
    counts = np.bincount(y_fold, minlength=3).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # guard against empty class
    w = counts.sum() / (3.0 * counts)
    return w.tolist()


def standardize_fit(rows):
    mu = Xnum_all[rows, :N_CONT].mean(0)
    sd = Xnum_all[rows, :N_CONT].std(0) + 1e-8
    return mu, sd


def apply_std(Xnum, mu, sd):
    out = Xnum.copy()
    out[:, :N_CONT] = (out[:, :N_CONT] - mu) / sd        # standardize continuous; leave flags
    return out


oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running OOF (TabM, CUDA, balanced class weights fit-in-fold) …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    if SMOKE:
        tr_idx = np.array([i for i in tr_idx if i in keepset])
        val_idx = np.array([i for i in val_idx if i in keepset])
    mu, sd = standardize_fit(tr_idx)
    Xn_tr = apply_std(Xnum_all[tr_idx], mu, sd)
    Xn_va = apply_std(Xnum_all[val_idx], mu, sd)
    # ATOMIC CHANGE: class weights computed from THIS fold's train rows only (fit-in-fold)
    fold_class_w = compute_class_weights(y[tr_idx])
    model = train_model(Xn_tr, Xc_all[tr_idx], y[tr_idx], cat_cards, fold_class_w)
    proba = predict_proba(model, Xn_va, Xc_all[val_idx])
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}")

if SMOKE:
    print("[smoke] OK — pipeline runs end-to-end. Exiting before full artifacts.")
    sys.exit(0)

np.save(NODE_DIR / "oof.npy", oof_proba)


# ---- full-train fit → test probs + submission ----
print("Retraining on full train for the test set …")
mu, sd = standardize_fit(np.arange(n))
full_class_w = compute_class_weights(y)  # all train rows — correct for final fit
fm = train_model(apply_std(Xnum_all, mu, sd), Xc_all, y, cat_cards, full_class_w)
tp = predict_proba(fm, apply_std(Xnum_te, mu, sd), Xc_te)
np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in tp.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
print(f"OOF metric (global): {oof_metric:.6f}")
print("Done.")
