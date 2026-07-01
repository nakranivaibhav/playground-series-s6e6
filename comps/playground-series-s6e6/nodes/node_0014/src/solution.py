"""node_0014 — draft (nn): FT-Transformer via `rtdl_revisiting_models`.

Built on: node_0009 (TabM arm). Feature prep is byte-identical to node_0009:
  same 26 numeric features (22 continuous standardized fit-inside-fold + 4 binary
  flags) + 2 native categorical inputs (spectral_type, galaxy_population), same
  frozen 5 folds from folds.json, CUDA.

ONE atomic change: the model class is swapped from TabM to FT-Transformer
  (rtdl_revisiting_models.FTTransformer, paper-default depth=3, d_block=192,
  8 heads, ReGLU FFN, lr=1e-4). The training loop, standardization,
  class-weighted CrossEntropy, early-stopping on an internal 10% val split —
  all carry over unchanged from node_0009. No PiecewiseLinearEmbeddings
  (FT-Transformer uses its own linear token embeddings).

Leakage discipline (two stateful steps, both FIT-INSIDE-FOLD):
  - standardization of the 22 continuous features: mean/std from the TRAIN FOLD only.
  - cat codes are stable (fixed categories from cast_categoricals), leak-safe.
  Early stopping uses an internal 10% split carved from the train fold.

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
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

import rtdl_revisiting_models as rtdl

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
NUMF = CONT + FLAGS                                                  # 26 → x_num
CATF = ["spectral_type", "galaxy_population"]                       # → x_cat (cardinalities below)

SMOKE = os.environ.get("TABM_SMOKE") == "1"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS, PATIENCE, BATCH = (6, 3, 8192) if SMOKE else (50, 10, 8192)
torch.manual_seed(SEED)
np.random.seed(SEED)

# FT-Transformer paper defaults (Gorishniy et al. 2021 / 2023)
D_BLOCK = 192
N_BLOCKS = 3
N_HEADS = 8
ATT_DROPOUT = 0.2
FFN_D_HIDDEN_MULT = 4 / 3
FFN_DROPOUT = 0.1
RESIDUAL_DROPOUT = 0.0
LR = 1e-4
WD = 1e-5


def engineer(df):
    df = cast_categoricals(df)
    df = add_color_features(df)
    df = add_extended_colors(df)
    df = add_redshift_features(df)
    df = add_qso_colorbox(df)
    df = add_galactic_coords(df)
    return df


def make_model(cat_cards):
    return rtdl.FTTransformer(
        n_cont_features=len(NUMF),
        cat_cardinalities=cat_cards,
        d_out=3,
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        attention_n_heads=N_HEADS,
        attention_dropout=ATT_DROPOUT,
        ffn_d_hidden_multiplier=FFN_D_HIDDEN_MULT,
        ffn_dropout=FFN_DROPOUT,
        residual_dropout=RESIDUAL_DROPOUT,
    ).to(DEVICE)


def predict_proba(model, Xn, Xc):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), 32768):
            xn = torch.as_tensor(Xn[s:s + 32768], dtype=torch.float32, device=DEVICE)
            xc = torch.as_tensor(Xc[s:s + 32768], dtype=torch.long, device=DEVICE)
            logits = model(xn, xc)          # (B, 3)
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def train_model(Xn, Xc, y, cat_cards, class_w, max_epochs=MAX_EPOCHS):
    """Train FT-Transformer with an internal 10% early-stop split (fit-inside-fold)."""
    g = torch.Generator().manual_seed(SEED)
    n = len(Xn)
    perm = torch.randperm(n, generator=g).numpy()
    nv = max(1, int(0.1 * n))
    vi, ti = perm[:nv], perm[nv:]
    model = make_model(cat_cards)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
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
            logits = model(Xn_t[idx], Xc_t[idx])   # (b, 3)
            loss = lossf(logits, y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ba = balanced_accuracy_score(yv, predict_proba(model, Xn[vi], Xc[vi]).argmax(1))
        if ba > best_ba + 1e-5:
            best_ba, best_state, bad = ba, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if SMOKE:
            print(f"    [smoke] ep{ep} val_ba={ba:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}) | rtdl_revisiting_models {rtdl.__version__} | SMOKE={SMOKE}")
print("Loading + engineering ...")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

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
    keepset = set(keep.tolist())

counts = np.bincount(y, minlength=3).astype(np.float64)
class_w = (counts.sum() / (3 * counts)).tolist()
N_CONT = len(CONT)


def standardize_fit(rows):
    mu = Xnum_all[rows, :N_CONT].mean(0)
    sd = Xnum_all[rows, :N_CONT].std(0) + 1e-8
    return mu, sd


def apply_std(Xnum, mu, sd):
    out = Xnum.copy()
    out[:, :N_CONT] = (out[:, :N_CONT] - mu) / sd   # standardize continuous; leave flags
    return out


oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running OOF (FT-Transformer, CUDA) ...")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    if SMOKE:
        tr_idx = np.array([i for i in tr_idx if i in keepset])
        val_idx = np.array([i for i in val_idx if i in keepset])
    mu, sd = standardize_fit(tr_idx)
    Xn_tr = apply_std(Xnum_all[tr_idx], mu, sd)
    Xn_va = apply_std(Xnum_all[val_idx], mu, sd)
    model = train_model(Xn_tr, Xc_all[tr_idx], y[tr_idx], cat_cards, class_w)
    proba = predict_proba(model, Xn_va, Xc_all[val_idx])
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")

if SMOKE:
    print("[smoke] OK - pipeline runs end-to-end. Exiting before full artifacts.")
    sys.exit(0)

np.save(NODE_DIR / "oof.npy", oof_proba)


# ---- full-train fit -> test probs + submission ----
print("Retraining on full train for the test set ...")
mu_full, sd_full = standardize_fit(np.arange(n))
fm = train_model(apply_std(Xnum_all, mu_full, sd_full), Xc_all, y, cat_cards, class_w)
tp = predict_proba(fm, apply_std(Xnum_te, mu_full, sd_full), Xc_te)
np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in tp.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
print(f"  oof_metric={oof_metric:.6f}")
print("Done.")
