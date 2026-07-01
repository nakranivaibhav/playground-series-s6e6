"""
================================================================================
 Stellar Classification (Playground S6E6) — single RealMLP, 5-fold bagged
 ~0.9693 local CV  ·  ~0.97009 public LB
================================================================================

WHAT THIS IS
------------
ONE neural-network model (no stacking, no blending) that classifies each object
as GALAXY / QSO / STAR. It is trained 5 times on 5 different 80/20 splits of the
training data, and the 5 models' probabilities are averaged for the test set
("bagging"). That averaging is the only ensembling here.

The score comes from three ingredients, in order of importance:

  1. FEATURE ENGINEERING  — colours (u-g, g-r, ...), magnitude summaries,
     redshift ratios, plus integer-binned + target-encoded categorical views of
     every column. (Section 2 + 3.)

  2. fs_zsoft  — four extra features that re-express REDSHIFT relative to the
     SDSS measurement-error floor (~3e-4). The hard GALAXY-vs-STAR confusion all
     happens at redshift ~ 0, where stars (noise-z) and the nearest galaxies are
     crushed together; these features stretch that tiny neighbourhood open.
     (Section 2b.)

  3. RealMLP  — a strong tabular-NN "reference recipe": periodic numerical
     embeddings (PBLD), an 8-model internal ensemble, EMA weight averaging,
     cosine LR schedule, label smoothing. This is the heavy part. (Section 4-5.)

HOW TO RUN ON KAGGLE
--------------------
  • Add the competition data (it mounts at /kaggle/input/playground-series-s6e6).
  • Settings -> Accelerator: GPU (T4 or P100). Internet: OFF is fine.
  • Run All. It writes /kaggle/working/submission.csv.
  • Wall-clock: roughly 20-40 min on a Kaggle GPU.

LEAKAGE NOTE
------------
Everything that "learns" from the data (the scaler, the categorical codes, the
target-encoder) is fit ONLY on each fold's training rows, then applied to that
fold's validation rows and to the test set. The fs_zsoft features use a FIXED
constant (3e-4), not anything learned from the data, so they are computed once up
front. The target column and the id column are never used as inputs.
================================================================================
"""
from __future__ import annotations

import gc
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# Where the competition CSVs live. Kaggle mounts them under /kaggle/input;
# this also falls back to the current directory for local runs.
DATA_DIR = Path("/kaggle/input/playground-series-s6e6")
if not (DATA_DIR / "train.csv").exists():
    DATA_DIR = Path(".")                       # local fallback
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

T0 = time.perf_counter()
def log(msg: str):
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Basic constants + reproducibility
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
N_SPLITS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}")


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — RealMLP hyper-parameters (the "reference recipe", as-is)
#  These are tuned; treat them as a fixed bundle. n_ens=8 means each model is
#  internally 8 sub-networks averaged together.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # architecture
    "n_ens": 8, "embed_dim": 7, "onehot_thresh": 10,
    "hidden_dims": [512, 512, 512], "dropout": 0.044,
    "p_drop_sched": "expm4t", "activation": nn.GELU, "add_front_scale": True,
    # periodic numerical embedding (PBLD)
    "pbld_hidden_dim": 16, "pbld_out_dim": 5, "pbld_freq_scale": 2.33,
    "pbld_activation": nn.PReLU, "pbld_lr_factor": 0.115,
    # optimiser / objective
    "lr": 0.01, "mom": 0.9, "sq_mom": 0.98, "lr_sched": "flat_cos",
    "flat_ratio": 0.20, "first_layer_lr_factor": 1.0, "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0, "lr_bias_mult": 0.1, "weight_decay": 0.0125,
    "wd_scale_mult": 0.1, "wd_bias_mult": 0.5, "grad_clip": 1.0,
    "class_weight_power": 0.0, "loss_prior_power": 1.075, "focal_gamma": 0.0,
    # label smoothing
    "ls_eps": 0.04, "ls_eps_sched": "cos",
    # preprocessing
    "tfms": ["median_center", "robust_scale"],
    # training loop
    "epochs": 6, "train_bs": 256, "eval_bs": 10240, "ema_decay": 0.997875,
    "verbosity": 1,
    "device": str(DEVICE), "random_state": SEED,
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Stateless feature engineering (safe to compute once, no fitting)
#  These are simple row-by-row formulas: colours, magnitude stats, redshift
#  ratios. No information crosses between rows, so there is no leakage risk.
# ─────────────────────────────────────────────────────────────────────────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
               ("u", "r"), ("g", "i"), ("r", "z")]
IMPORTANT_COMBOS = sorted([("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")])

_Z_EPS = 3e-4         # SDSS redshift-measurement error floor (a FIXED constant)
_Z_LOG_SHIFT = 0.011  # fixed shift so log10 stays positive (min redshift ≈ -0.01)


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only — no fitting, no target, no cross-row stats."""
    df = df.copy()
    # redshift ratios
    for col in ("g", "i"):
        df[f"_{col}_div_redshift"] = (
            (df[col] / (df["redshift"] + 1e-6))
            .replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        )
    # colours (differences of adjacent magnitudes)
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    # magnitude summaries
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    # smooth log of (shifted) redshift
    shifted = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted).astype("float32")
    return df


# ── SECTION 2b — fs_zsoft: redshift re-expressed vs the error floor (ε=3e-4) ──
def zsoft_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Four extra features that stretch open the redshift ≈ 0 neighbourhood,
    where the GALAXY-vs-STAR confusion lives. All use the fixed constant ε, so
    they are stateless (no fitting, no leakage)."""
    df = df.copy()
    z = df["redshift"].astype("float64")                          # float64 near z≈0
    df["_zsoft_snr"]   = (z / _Z_EPS).astype("float32")          # redshift signal-to-noise
    df["_zsoft_asinh"] = np.arcsinh(z / _Z_EPS).astype("float32")# linear near 0, log-like far out
    df["_zsoft_log"]   = np.log10(z + _Z_LOG_SHIFT + _Z_EPS).astype("float32")
    df["_zsoft_star"]  = (2.0 / (1.0 + np.exp(np.abs(z) / _Z_EPS))).astype("float32")  # bump at z=0
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Fold-wise categorical + target encoding (FIT ON TRAIN FOLD ONLY)
#  These DO learn from the data, so they must be fit inside each fold to avoid
#  leakage. They produce: integer codes for the raw categoricals, integer-floor
#  "category" views of every numeric column, quantile bins of `delta`, a couple
#  of interaction crosses, and multiclass target-encodings of those crosses.
# ─────────────────────────────────────────────────────────────────────────────
def fit_fold_categoricals(df_tr, df_val, df_te):
    """Fit category encodings on the train fold, apply to val + test."""
    def fac_fit(s):
        codes, uniques = pd.factorize(s, sort=False)
        return codes.astype("int32"), uniques
    def fac_apply(s, uniques):
        m = {cat: i for i, cat in enumerate(uniques)}
        return s.map(m).fillna(-1).astype("int32")

    tr, va, te = df_tr.copy(), df_val.copy(), df_te.copy()

    # raw categoricals -> integer codes
    for col in BASE_CAT_COLS:
        codes_tr, uniq = fac_fit(tr[col])
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(fac_apply(va[col], uniq), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(fac_apply(te[col], uniq), index=te.index).astype("int32").astype("category")

    # integer-floor "category" view of every numeric column
    for col in BASE_NUM_COLS:
        name = f"{col}_cat_"
        codes_tr, uniq = fac_fit(np.floor(tr[col]).astype("float32"))
        tr[name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset in (va, te):
            codes = fac_apply(np.floor(dset[col]).astype("float32"), uniq)
            dset[name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    # quantile bins of `delta` (100 and 500 bins)
    for n_bins in (100, 500):
        name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        tr[name] = pd.Series(kb.fit_transform(tr[["delta"]]).ravel().astype("int32"),
                             index=tr.index).astype("int32").astype("category")
        for dset in (va, te):
            dset[name] = pd.Series(kb.transform(dset[["delta"]]).ravel().astype("int32"),
                                   index=dset.index).astype("int32").astype("category")

    # interaction crosses (e.g. alpha_cat_ x delta_cat_)
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        name = "__".join(cols) + "__"
        combo_names.append(name)
        joined_tr = tr[cols[0]].astype(str)
        for c in cols[1:]:
            joined_tr = joined_tr + "|" + tr[c].astype(str)
        codes_tr, uniq = pd.factorize(joined_tr, sort=False)
        tr[name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32").astype("category")
        for dset in (va, te):
            joined = dset[cols[0]].astype(str)
            for c in cols[1:]:
                joined = joined + "|" + dset[c].astype(str)
            dset[name] = pd.Series(fac_apply(joined, uniq), index=dset.index).astype("int32").astype("category")

    cat_cols = sorted([c for c in tr.columns if str(tr[c].dtype) == "category"])
    return tr, va, te, cat_cols, combo_names


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    """Multiclass target-encode the interaction crosses (fit on train fold only)."""
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    try:
        enc = TargetEncoder(target_type="multiclass", cv=5, smooth="auto",
                            shuffle=True, random_state=fold_seed)
    except TypeError:
        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    tr_enc = enc.fit_transform(X_tr[combo_names], y_tr)
    val_enc = enc.transform(X_val[combo_names])
    te_enc = enc.transform(X_te[combo_names])
    names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(N_CLASSES)]
    X_tr[names] = np.asarray(tr_enc, dtype="float32")
    X_val[names] = np.asarray(val_enc, dtype="float32")
    X_te[names] = np.asarray(te_enc, dtype="float32")
    return X_tr, X_val, X_te


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — The RealMLP network
#  This is the "reference recipe" architecture. The pieces:
#    NumericalPreprocessor  — median-center + scale by IQR (robust to outliers)
#    CategoricalFeatureLayer — one-hot small cats, learned embeddings for big ones
#    PBLDEmbedding          — periodic (cosine) embedding of each numeric feature
#    NTPLinear / ScalingLayer — the ensemble-aware linear layers (n_ens in parallel)
#    RealMLP                — stitches it together into an 8-way internal ensemble
#  You don't need to follow every line; it's a known-good tabular-NN block.
# ─────────────────────────────────────────────────────────────────────────────
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """Median-center then divide by IQR — fit on train fold only."""
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale",
                                                "smooth_clip", "l2_normalize")]
    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero = q == 0.0
            q[zero] = 0.5 * (X.max(axis=0)[zero] - X.min(axis=0)[zero])
            self._iqr = 1.0 / (q + 1e-30)
            self._iqr[q == 0.0] = 0.0
        return self
    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for t in self._tfms:
            if t == "median_center":   X -= self._median[None, :]
            elif t == "robust_scale":  X *= self._iqr[None, :]
            elif t == "smooth_clip":   X = X / np.sqrt(1 + (X / 3) ** 2)
            elif t == "l2_normalize":
                n = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(n == 0, 1.0, n)
        return X


class CategoricalFeatureLayer(nn.Module):
    """One-hot for small-cardinality cats; per-ensemble embeddings for big ones."""
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens, self.cat_dims = n_ens, cat_dims
        self.onehot_features, self.embed_layers, self._embed_idx = [], nn.ModuleList(), []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                self.embed_layers.append(nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)]))
                self._embed_idx.append(i)
    def forward(self, x):
        bs, n_ens, _ = x.shape
        feats = []
        if self.onehot_features:
            oh = x[:, :, self.onehot_features]
            dims = [self.cat_dims[i] for i in self.onehot_features]
            enc = torch.zeros(bs, n_ens, sum(dims), device=x.device)
            start = 0
            for idx, dim in enumerate(dims):
                enc.scatter_(2, oh[:, :, idx:idx + 1].long() + start, 1.0)
                start += dim
            feats.append(enc)
        for emb_list, fi in zip(self.embed_layers, self._embed_idx):
            per = [emb_list[m](x[:, m, fi:fi + 1].long()) for m in range(n_ens)]
            feats.append(torch.cat(per, dim=1))
        return torch.cat(feats, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))
    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    """Linear layer with an independent weight per ensemble member."""
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None
    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        return x + self.bias if self.bias is not None else x


class PBLDEmbedding(nn.Module):
    """Periodic (cosine) embedding of each numeric feature, then a small MLP."""
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)
    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens, embed_dim = cfg["n_ens"], cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens, cat_dims, embed_dim, cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(n_ens, n_numerical, cfg["pbld_hidden_dim"],
                                       cfg["pbld_out_dim"], cfg["pbld_freq_scale"], cfg["pbld_activation"])
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total = num_emb_dim + cat_emb_dim
        act = cfg["activation"]
        layers = [ScalingLayer(n_ens, total)] if cfg["add_front_scale"] else []
        self._dropout_modules = []
        in_dim = total
        for i, h in enumerate(cfg["hidden_dims"]):
            lin = NTPLinear(n_ens, in_dim, h)
            if i == 0:
                self.first_linear = lin
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [lin, act(), drop]
            in_dim = h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens, in_dim, output_dim)
    def forward(self, x_num, x_cat):
        x_num = self.num_embed(x_num.unsqueeze(1).expand(-1, self.n_ens, -1))
        x_cat = self.cate(x_cat.unsqueeze(1).expand(-1, self.n_ens, -1))
        x = self.hidden(torch.cat([x_num, x_cat], dim=2))
        return F.softmax(self.output_layer(x), dim=2)   # (batch, n_ens, classes)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Training helpers + the sklearn-style classifier wrapper
# ─────────────────────────────────────────────────────────────────────────────
def apply_schedule(init, progress, sched, flat_ratio=0.3):
    """LR / dropout / label-smoothing schedules over training progress (0..1)."""
    if sched == "constant": return init
    if sched == "cos":      return init * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio: return init
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init * (math.cos(math.pi * t) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio: return init
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init * (1 - t)
    if sched == "sqrt_cos": return init * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    if sched == "expm4t":   return init * math.exp(-4 * progress)
    raise ValueError(sched)


def get_parameter_groups(model, p):
    """Different LR / weight-decay for scale / embedding / first / other / bias params."""
    first_w_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w, other_w, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:          pbld_p.append(param)
        elif "scale" in name:            scale_p.append(param)
        elif id(param) == first_w_id:    first_w.append(param)
        elif "bias" in name:             bias_p.append(param)
        else:                            other_w.append(param)
    LR, WD = p["lr"], p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"]},
        {"params": pbld_p,  "lr": LR * p["pbld_lr_factor"], "weight_decay": WD},
        {"params": first_w, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"]},
        {"params": other_w, "lr": LR, "weight_decay": WD},
        {"params": bias_p,  "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"]},
    ]


def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None, focal_gamma=0.0, loss_prob_multipliers=None):
    """Cross-entropy with label smoothing (+ optional class weights / prior reweighting)."""
    n_classes = y_pred.size(1)
    if loss_prob_multipliers is not None:
        y_pred = y_pred * loss_prob_multipliers[None, :]
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-15)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if focal_gamma > 0:
        pt = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1).clamp(1e-15, 1.0)
        loss = loss * torch.pow(1.0 - pt, focal_gamma)
    if class_weights is not None:
        w = class_weights[y_true]
        return (loss * w).sum() / w.sum()
    return loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    """Trains a RealMLP for a fixed number of epochs, keeps the best-validation
    (EMA) weights, and predicts averaged class probabilities."""
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        cat_col_names = cat_col_names or []
        num_cols = [c for c in X_train.columns if c not in cat_col_names]

        Xtn = X_train[num_cols].values.astype(np.float32)
        Xvn = X_val[num_cols].values.astype(np.float32)
        Xtc = X_train[cat_col_names].values.astype(np.int64)
        Xvc = X_val[cat_col_names].values.astype(np.int64)
        y_tr, y_v = np.asarray(y_train), np.asarray(y_val)

        # numeric scaling — fit on train fold only
        self.pre_ = NumericalPreprocessor(p["tfms"]).fit(Xtn)
        Xtn, Xvn = self.pre_.transform(Xtn), self.pre_.transform(Xvn)
        self.cat_col_names_, self.num_col_names_ = cat_col_names, num_cols

        # embedding sizes need each categorical's max code (over train+val+test)
        if cat_col_names:
            all_cat = [Xtc, Xvc] + ([X_test[cat_col_names].values.astype(np.int64)] if X_test is not None else [])
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
            cmax = np.array(cat_dims) - 1
            Xtc, Xvc = np.clip(Xtc, 0, cmax), np.clip(Xvc, 0, cmax)
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        # balanced class weights + optional prior reweighting
        classes = np.unique(y_tr)
        self.classes_ = classes
        w = compute_class_weight("balanced", classes=classes, y=y_tr)
        if float(p["class_weight_power"]) != 1.0:
            w = np.power(w, float(p["class_weight_power"]))
        class_weights = torch.as_tensor(w, dtype=torch.float32, device=dev)

        loss_mult = None
        if float(p["loss_prior_power"]) != 0.0:
            counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            counts = counts / np.exp(np.log(counts).mean())
            loss_mult = torch.as_tensor(np.power(counts, float(p["loss_prior_power"])),
                                        dtype=torch.float32, device=dev)

        self.model_ = RealMLP(len(classes), cat_dims, Xtn.shape[1], p).to(dev)
        groups = get_parameter_groups(self.model_, p)
        for g in groups:
            g["lr_base"] = g["lr"]
        opt = torch.optim.AdamW(groups, betas=(p["mom"], p["sq_mom"]))

        Xtn_t = torch.as_tensor(Xtn, dtype=torch.float32, device=dev)
        Xtc_t = torch.as_tensor(Xtc, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn_t = torch.as_tensor(Xvn, dtype=torch.float32, device=dev)
        Xvc_t = torch.as_tensor(Xvc, dtype=torch.long, device=dev)

        n_ens, train_bs, eval_bs = p["n_ens"], p["train_bs"], p["eval_bs"]
        epochs, total_steps = p["epochs"], p["epochs"] * len(y_tr)
        order = np.arange(len(y_tr))
        ema = {k: v.detach().clone() for k, v in self.model_.state_dict().items()} if p["ema_decay"] > 0 else None
        best_score, best_probs, best_state = -np.inf, None, None

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx = order[start:start + train_bs]
                for g in opt.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, p["lr_sched"], p["flat_ratio"])
                opt.zero_grad()
                pred = self.model_(Xtn_t[idx], Xtc_t[idx])     # (bs, n_ens, classes)
                ls = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], p["flat_ratio"])
                drop = apply_schedule(p["dropout"], progress, p["p_drop_sched"], p["flat_ratio"])
                for dm in self.model_._dropout_modules:
                    dm.p = drop
                loss = smooth_ce_loss(ytt[idx].repeat_interleave(n_ens),
                                      pred.reshape(-1, len(classes)), ls=ls,
                                      class_weights=class_weights,
                                      focal_gamma=float(p["focal_gamma"]),
                                      loss_prob_multipliers=loss_mult)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for k, v in self.model_.state_dict().items():
                            if torch.is_floating_point(v):
                                ema[k].mul_(p["ema_decay"]).add_(v.detach(), alpha=1 - p["ema_decay"])
                            else:
                                ema[k].copy_(v)
            np.random.shuffle(order)

            # validation with EMA weights
            self.model_.eval()
            live = None
            if ema is not None:
                live = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn_t[s:s + eval_bs], Xvc_t[s:s + eval_bs]).mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)], axis=0)
            if live is not None:
                self.model_.load_state_dict(live, strict=True)

            score = balanced_accuracy_score(y_v, val_probs.argmax(1))
            if score > best_score:
                best_score, best_probs = score, val_probs.copy()
                src = ema if ema is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in src.items()}
            if p["verbosity"] >= 1:
                log(f"    epoch {epoch+1}/{epochs}  val_BA={score:.5f}  best={best_score:.5f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_val_probs_, self._dev = best_probs, dev
        return self

    def predict_proba(self, X):
        eb = self.params["eval_bs"]
        Xn = self.pre_.transform(X[self.num_col_names_].values.astype(np.float32))
        Xc = np.clip(X[self.cat_col_names_].values.astype(np.int64), 0, np.array(self.cat_dims_) - 1)
        Xn_t = torch.as_tensor(Xn, dtype=torch.float32, device=self._dev)
        Xc_t = torch.as_tensor(Xc, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn_t[s:s + eb], Xc_t[s:s + eb]).mean(dim=1).cpu().numpy()
                for s in range(0, len(Xn), eb)], axis=0)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN: load data, build features, 5-fold train, average, submit
# ═════════════════════════════════════════════════════════════════════════════
def main():
    log("Loading data …")
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    log(f"  train={train_raw.shape}  test={test_raw.shape}")

    y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
    n_train, n_test = len(train_raw), len(test_raw)

    # ---- features (computed once; the "stateless" part is leakage-free) ----
    log("Building features (stateless FE + fs_zsoft) …")
    X = zsoft_fe(stateless_fe(train_raw.drop(columns=[IDC, TARGET])))
    X_test = zsoft_fe(stateless_fe(test_raw.drop(columns=[IDC])))
    assert TARGET not in X.columns and IDC not in X.columns, "target/id leaked into features!"

    # ---- 5-fold stratified split (same scheme as the original: seed 42) ----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n_train), y_all))

    oof = np.zeros((n_train, N_CLASSES), dtype=np.float32)         # out-of-fold preds (for CV)
    test_proba = np.zeros((n_test, N_CLASSES), dtype=np.float32)   # averaged test preds
    fold_scores = []

    for fold_id, (tr_idx, val_idx) in enumerate(folds):
        fold_seed = SEED + (fold_id + 1) * 100
        seed_everything(fold_seed)
        log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

        # fit categoricals + target-encoding on THIS fold's train rows only
        X_tr, X_val, X_te, cat_cols, combos = fit_fold_categoricals(
            X.iloc[tr_idx].reset_index(drop=True),
            X.iloc[val_idx].reset_index(drop=True),
            X_test.copy())
        X_tr, X_val, X_te = add_target_encoding(
            X_tr, y_all[tr_idx], X_val, X_te, combos, fold_seed)

        # keep a consistent column order across train/val/test
        X_tr = X_tr.reindex(sorted(X_tr.columns), axis=1)
        X_val = X_val.reindex(sorted(X_val.columns), axis=1)
        X_te = X_te.reindex(sorted(X_te.columns), axis=1)
        cat_cols = sorted(cat_cols)

        model = RealMLP_TD_Classifier(random_state=fold_seed, device=str(DEVICE))
        model.fit(X_tr, y_all[tr_idx], X_val, y_all[val_idx],
                  cat_col_names=cat_cols, X_test=X_te)

        oof[val_idx] = model.best_val_probs_.astype("float32")
        test_proba += model.predict_proba(X_te).astype("float32") / N_SPLITS

        s = balanced_accuracy_score(y_all[val_idx], oof[val_idx].argmax(1))
        fold_scores.append(s)
        log(f"  fold {fold_id} balanced_accuracy = {s:.6f}")

        del model, X_tr, X_val, X_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv = float(np.mean(fold_scores))
    log(f"per-fold = {[f'{s:.6f}' for s in fold_scores]}")
    log(f"CV (balanced accuracy) = {cv:.6f}")

    # ---- write submission ----
    pred_labels = np.array([CLASSES[i] for i in test_proba.argmax(1)])
    sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
    sub = sub[list(sample_sub.columns)]
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    log(f"Wrote {OUT_DIR / 'submission.csv'} ({len(sub)} rows)")
    log(f"Class distribution:\n{sub[TARGET].value_counts().to_string()}")
    log("Done.")


if __name__ == "__main__":
    main()
