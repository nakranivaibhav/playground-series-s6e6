"""node_0008 — draft (nn family): a CUDA MLP as a de-correlated blend arm.

The GBDT lineage is tapped out (tuning regressed, threshold within-noise, blend
harvested the cross-model signal). Per the search policy this is the architecture
PIVOT: a structurally different model family whose errors should be de-correlated
from the trees, so it can lift the combine even at a lower solo CV.

Leakage discipline (the only stateful step is standardization):
  - 22 continuous features standardized with mean/std computed FROM THE TRAIN FOLD
    ONLY (manual, numpy — no global .fit on full/val data), applied to the val fold.
  - 4 binary flags passed through 0/1.
  - 2 fixed-category bins (spectral_type, galaxy_population) one-hot encoded — the
    bin definitions are stateless (set in clean.cast_categoricals), so encoding the
    whole frame at once introduces no train→val leakage.
  - the official val fold is used ONLY for OOF scoring; early stopping uses an
    internal 10% split carved from the train fold.

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

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
FLAGS = ["is_star_z", "is_highz", "qso_box", "uv_excess"]           # 4, pass-through
CATS = ["spectral_type", "galaxy_population"]                       # one-hot (fixed cats)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS, PATIENCE, BATCH = 80, 8, 8192
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


class MLP(nn.Module):
    def __init__(self, d_in, n_cls=3):
        super().__init__()
        def block(a, b):
            return [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(0.2)]
        self.net = nn.Sequential(
            *block(d_in, 256), *block(256, 128), *block(128, 64), nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def build_matrix(df):
    """Stateless part of features: flags + one-hot cats. Continuous returned raw (scaled later)."""
    cont = df[CONT].to_numpy(np.float32)
    flags = df[FLAGS].to_numpy(np.float32)
    dummies = pd.get_dummies(df[CATS], columns=CATS)       # fixed categories → stable columns
    return cont, flags, dummies.to_numpy(np.float32), list(dummies.columns)


def train_mlp(Xtr, ytr, d_in, class_w, max_epochs=MAX_EPOCHS):
    """Train with an internal 10% early-stopping split (stratified). Returns the best model."""
    g = torch.Generator().manual_seed(SEED)
    n = len(Xtr)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]
    Xt = torch.tensor(Xtr[ti], device=DEVICE)
    yt = torch.tensor(ytr[ti], device=DEVICE, dtype=torch.long)
    Xv = torch.tensor(Xtr[vi], device=DEVICE)
    yv = ytr[vi]
    model = MLP(d_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=DEVICE, dtype=torch.float32))
    best_ba, best_state, bad = -1.0, None, 0
    nt = len(ti)
    for ep in range(max_epochs):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH):
            idx = bperm[s:s + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        ba = balanced_accuracy_score(yv, pv)
        if ba > best_ba + 1e-5:
            best_ba, best_state, bad = ba, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_proba(model, X):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 65536):
            xb = torch.tensor(X[s:s + 65536], device=DEVICE)
            out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out, 0)


print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'})")
print("Loading + engineering …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
test = engineer(pd.read_csv(COMP_DIR / "data/test.csv"))
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]

(NODE_SRC / "features.txt").write_text("\n".join(feature_columns(train)) + "\n")

y = train[TARGET].map(LABEL2IDX).to_numpy()
n = len(train)
cont_tr, flags_tr, dum_tr, dcols = build_matrix(train)
cont_te, flags_te, dum_te, dcols_te = build_matrix(test)
assert dcols == dcols_te, f"dummy column mismatch {dcols} vs {dcols_te}"
d_in = len(CONT) + len(FLAGS) + len(dcols)
print(f"  input dim = {d_in}  (cont {len(CONT)} + flags {len(FLAGS)} + onehot {len(dcols)}={dcols})")

# class weights = balanced (inverse frequency), matching the GBDT arms
counts = np.bincount(y, minlength=3).astype(np.float64)
class_w = (counts.sum() / (3 * counts)).tolist()
print(f"  class counts {counts.tolist()}  weights {[round(w,3) for w in class_w]}")


def assemble(cont, flags, dum, mu, sd):
    return np.concatenate([(cont - mu) / sd, flags, dum], axis=1).astype(np.float32)


oof_proba = np.zeros((n, 3), dtype=np.float64)
per_fold = []
print("Running 5-fold OOF (CUDA MLP) …")
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    mu = cont_tr[tr_idx].mean(0)                       # FIT-INSIDE-FOLD: train rows only
    sd = cont_tr[tr_idx].std(0) + 1e-8
    Xtr = assemble(cont_tr[tr_idx], flags_tr[tr_idx], dum_tr[tr_idx], mu, sd)
    Xva = assemble(cont_tr[val_idx], flags_tr[val_idx], dum_tr[val_idx], mu, sd)
    model = train_mlp(Xtr, y[tr_idx], d_in, class_w)
    proba = predict_proba(model, Xva)
    oof_proba[val_idx] = proba
    s = balanced_accuracy_score(y[val_idx], proba.argmax(1))
    per_fold.append(s)
    print(f"  fold {fi['fold']}: balanced_accuracy = {s:.6f}")

oof_metric = balanced_accuracy_score(y, oof_proba.argmax(1))
mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}  (oof_metric={oof_metric:.6f})")
np.save(NODE_DIR / "oof.npy", oof_proba)


# ---- full-train fit → test probs + submission ----
print("Retraining on full train for the test set …")
mu = cont_tr.mean(0); sd = cont_tr.std(0) + 1e-8       # full train (test never involved)
Xall = assemble(cont_tr, flags_tr, dum_tr, mu, sd)
Xtest = assemble(cont_te, flags_te, dum_te, mu, sd)
fm = train_mlp(Xall, y, d_in, class_w)
tp = predict_proba(fm, Xtest)
np.save(NODE_DIR / "test_probs.npy", tp)
labels = np.array([LABEL_ORDER[i] for i in tp.argmax(1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0008 metrics
metric: Balanced Accuracy Score (maximize)
model: PyTorch MLP [256,128,64]+BN+Dropout0.2, class-weighted CE, Adam, early-stop, CUDA
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (oof_metric={oof_metric:.6f})
input_dim: {d_in}
change: non-GBDT diversity arm (MLP on CUDA). Standardization fit-inside-fold. Saves oof+test_probs.
""")
print("Done.")
