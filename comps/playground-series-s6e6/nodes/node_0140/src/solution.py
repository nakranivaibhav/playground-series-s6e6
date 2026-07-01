"""node_0140 — draft (nn): RealMLP + fs_zsoft (redshift-error-aware re-expression).

Built on: nodes/node_0028/src (RealMLP reference recipe scaffold, byte-identical).
ONE ATOMIC CHANGE: add stateless feature-set fs_zsoft alongside fs_realmlp_fe.

fs_zsoft (4 features, all stateless — row-wise deterministic on raw redshift, FIXED
constant ε=3e-4, no fit / no target / no cross-row stat):
  1. z_snr   = redshift / 3e-4   (z signal-to-noise vs SDSS z-error floor)
  2. z_asinh = np.arcsinh(redshift / 3e-4)   (well-conditioned z-warp)
  3. z_log   = log10(redshift + 3e-4)   (log z-warp, expands z≈0 neighbourhood)
  4. z_star_soft = sigmoid((redshift + 0.0) / 3e-4 * -1) × 2
     = soft STAR-likelihood: smooth bump centred at z=0, width ~error floor.
     Implementation: 1 / (1 + exp(|redshift| / 3e-4))  × 2  (symmetric around 0)
     → =1.0 at z=0, falls to 0 at |z| >> 3e-4.

Hypothesis: the GAL↔STAR bottleneck lives at z≈0 where stellar-noise-z and
smallest real galaxy z (~0.02) are crushed into a sub-0.01 interval. Re-expressing z
relative to its error floor gives that neighbourhood a wide, stable margin (z_snr /
asinh / log-warp) plus a soft "is this stellar noise?" axis.

Recipe source: research.md L244 — z_warp = log10(z+3e-4) / asinh(z/ε), ε≈3e-4.

CHEAP-KILL (fold-0 only):
  Continue ONLY if fold-0 err-corr vs node_0070 < 0.65 AND solo BA ≥ 0.965.
  Else STOP (mark status dead or valid-but-washed).

Leakage discipline:
  - fs_zsoft: ALL 4 features use FIXED constant ε=3e-4 (NOT data-fit).
    No target, no cross-row stat, no fitted transform → stateless, safe to
    compute once on train+test before any fold loop.
  - fs_realmlp_fe: same as node_0028 (stateless parts once; fit_in_fold for
    KBins, TargetEncoder, NumericalPreprocessor — all inside fold loop).
  - Folds from frozen folds.json; no refitting of folds.

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import math
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
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent
# REPO_ROOT: two levels above COMP_DIR (comps/<slug> → comps → repo)
REPO_ROOT = COMP_DIR.parent.parent

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

# ─── CONFIG (faithful port from realmlp-v5-for-s6e6.py) ──────────────────────
CONFIG = {
    # Model architecture
    "n_ens": 8,
    "embed_dim": 7,
    "onehot_thresh": 10,
    "hidden_dims": [512, 512, 512],
    "dropout": 0.044,
    "p_drop_sched": "expm4t",
    "activation": nn.GELU,
    "add_front_scale": True,
    # PBLD / periodic numerical embedding
    "pbld_hidden_dim": 16,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 2.33,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.115,
    # Optimizer and training objective
    "lr": 0.01,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.20,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.0125,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    "class_weight_power": 0.0,
    "class_weight_multipliers": None,
    "sample_weight_power": 0.0,
    "loss_prior_power": 1.075,
    "focal_gamma": 0.0,
    "eval_class_multipliers": None,
    # Label smoothing
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    # Preprocessing
    "tfms": ["median_center", "robust_scale"],
    # Training loop
    "epochs": 6,
    "train_bs": 256,
    "eval_bs": 10240,
    "numeric_noise_std": 0.0,
    "ema_decay": 0.997875,
    "verbosity": 2,
    # Early stopping
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    # Device and seed
    "device": str(DEVICE),
    "random_state": SEED,
}

# ─── Feature engineering globals ─────────────────────────────────────────────
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
    """
    Pure row-wise / stateless feature engineering — safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    """
    df = df.copy()

    # Redshift ratios
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")

    # Color pairs
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")

    # Magnitude aggregates
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")

    # Log1p of shifted redshift
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")

    return df


# ── fs_zsoft: redshift-error-aware re-expression (4 stateless features) ─────────
# ε = SDSS z-measurement error floor (3e-4), FIXED constant — NOT data-fit.
# All features are row-wise deterministic on raw redshift. Safe to compute once.
_Z_EPS = 3e-4  # SDSS z-error floor


def zsoft_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fs_zsoft features — stateless re-expression of z relative to the SDSS
    z-error floor (ε=3e-4). Attacks the GAL↔STAR z≈0 bottleneck by expanding
    the stellar-noise-z neighbourhood into a wide, stable margin.

    All 4 features use the FIXED constant ε=3e-4 (no fit, no target, no cross-row
    stat) → leak-safety class: stateless.
    """
    df = df.copy()
    z = df["redshift"].astype("float64")  # use float64 for precision near z≈0

    # 1. z_snr: redshift signal-to-noise ratio vs the error floor
    #    Stars cluster tightly near 0; galaxies/QSOs spread to high values.
    df["_zsoft_snr"] = (z / _Z_EPS).astype("float32")

    # 2. z_asinh: well-conditioned warp, expands z≈0 neighbourhood
    #    arcsinh(z/ε) is linear near 0, log-like at large |z|.
    df["_zsoft_asinh"] = np.arcsinh(z / _Z_EPS).astype("float32")

    # 3. z_log: log10(z + ε) — same neighbourhood expansion in log space
    #    Note: redshift can be slightly negative (measurement noise, floor at -0.01).
    #    Use a fixed shift of 0.011 (> |min(redshift)| ≈ 0.00997) to guarantee
    #    z_shifted > 0 for ALL rows in both train and test. FIXED CONSTANT — NOT
    #    data-derived, so this is truly stateless.
    _Z_LOG_SHIFT = 0.011  # fixed constant; |min(redshift)| ≈ 0.00997 < this
    z_shifted = z + _Z_LOG_SHIFT + _Z_EPS  # guaranteed > 0
    df["_zsoft_log"] = np.log10(z_shifted).astype("float32")

    # 4. z_star_soft: soft STAR-likelihood — smooth bump centred at z=0
    #    1 / (1 + exp(|z|/ε)) × 2 → =1.0 at z=0, decays exponentially
    #    This is the soft membership of the z≈0 stellar-noise band.
    df["_zsoft_star"] = (2.0 / (1.0 + np.exp(np.abs(z) / _Z_EPS))).astype("float32")

    return df


# Global category_map (populated during fit)
_category_map: dict = {}


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names).
    Called INSIDE the fold loop — fit_in_fold.
    """
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    # Work on copies
    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    # Original categorical columns (spectral_type, galaxy_population)
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    # Integer-floor categorical views of every base numeric
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

    # Delta quantile bins (100 and 500) — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

    # Interaction cross-combos
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
    """
    TargetEncoder fit on train fold only (fit_in_fold), transform val and test.
    Returns modified copies and the list of new TE column names.
    """
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


# ─── Model components (faithful port from cdeotte reference) ──────────────────

class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """Median-center + robust-scale (IQR) — fit on train fold only."""

    def __init__(self, tfms):
        self._tfms = [t for t in tfms
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim: int = 8, onehot_thresh: int = 8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []

        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []

        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx: idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)

        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(n_ens):
                indices = x[:, model_idx, feat_idx: feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)

        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay embedding for numerical features."""

    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16,
                 out_dim: int = 4, freq_scale: float = 0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        # x: (batch, n_ens, n_features)
        periodic = torch.cos(
            2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0)
        )
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim: int, cat_dims, n_numerical: int, cfg: dict):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens

        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"],
        )

        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim

        act = cfg["activation"]
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))

        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(cfg["hidden_dims"]):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h

        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)  # (batch, n_ens, output_dim)


def apply_schedule(init_value: float, progress: float, sched: str, flat_ratio: float = 0.3) -> float:
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model: RealMLP, p: dict):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)

    LR = p["lr"]
    WD = p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD, "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR, "weight_decay": WD, "group": "other_w"},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"], "group": "bias"},
    ]


def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None,
                   focal_gamma=0.0, loss_prob_multipliers=None):
    n_classes = y_pred.size(1)
    if loss_prob_multipliers is not None:
        y_pred = y_pred * loss_prob_multipliers[None, :]
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-15)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if focal_gamma > 0:
        pt = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1).clamp(1e-15, 1.0)
        per_sample_loss = per_sample_loss * torch.pow(1.0 - pt, focal_gamma)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    """Sklearn-compatible wrapper — port of cdeotte reference."""

    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            cat_col_names=None, X_test: pd.DataFrame = None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        # Numerical preprocessing — fit on train fold only
        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names

        # Cat dims — computed from train+val+test union (for embedding size only)
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        # Class weights
        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        cw_power = float(p.get("class_weight_power", 1.0))
        if cw_power != 1.0:
            weights_np = np.power(weights_np, cw_power)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        # loss_prior_power — down-weight majority class in loss
        loss_prior_power = float(p.get("loss_prior_power", 0.0))
        loss_prob_multipliers = None
        if loss_prior_power != 0.0:
            class_counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            class_counts = class_counts / np.exp(np.log(class_counts).mean())
            loss_mult_np = np.power(class_counts, loss_prior_power)
            loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

        n_classes = len(classes)
        self.model_ = RealMLP(
            output_dim=n_classes, cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1], cfg=p,
        ).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        lr_sched = p["lr_sched"]
        flat_ratio = p["flat_ratio"]
        ema_decay = float(p.get("ema_decay", 0.0))
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        best_state = None
        ema_state = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start: start + train_bs]

                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)

                optimizer.zero_grad()
                x_num_batch = Xtn[idx_batch]
                y_pred = self.model_(x_num_batch, Xtc[idx_batch])  # (bs, n_ens, C)

                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val

                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val,
                    class_weights=class_weights,
                    focal_gamma=float(p.get("focal_gamma", 0.0)),
                    loss_prob_multipliers=loss_prob_multipliers,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()

                if ema_state is not None:
                    with torch.no_grad():
                        model_state = self.model_.state_dict()
                        for key, value in model_state.items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)

            np.random.shuffle(train_order)

            # Validation — use EMA weights if available
            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)

            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s: s + eval_bs], Xvc[s: s + eval_bs])
                        .mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)

            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)

            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}

            if verbose >= 2:
                log(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                    f"best={best_score:.5f}  ls={ls_val:.4f}  drop={drop_val:.4f}"
                    + ("  *" if improved else ""))

            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        log(f"  Early stopping at epoch {epoch + 1} (best {best_epoch})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            log(f"  best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s: s + eval_bs], Xc[s: s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data …")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE (fs_realmlp_fe + fs_zsoft) …")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
# Add fs_zsoft features (stateless — fixed constant ε=3e-4, no fit/target/cross-row)
X_stateless = zsoft_fe(X_stateless)
X_test_stateless = zsoft_fe(X_test_stateless)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── PRE-FLIGHT LEAKAGE CHECK ────────────────────────────────────────────────
# 1. Confirm TARGET and ID not in feature columns
assert TARGET not in X_stateless.columns, f"LEAK: {TARGET} in features!"
assert IDC not in X_stateless.columns, f"LEAK: {IDC} in features!"
# 2. Quick single-feature vs target sweep (sample 50k)
_sample_size = min(50_000, len(train_raw))
_s = X_stateless.sample(_sample_size, random_state=0)
_ys = y_all[_s.index]
for _c in ["_zsoft_snr", "_zsoft_asinh", "_zsoft_log", "_zsoft_star"]:
    _x = _s[_c].fillna(_s[_c].mean())
    _corr = abs(np.corrcoef(_x.values, _ys)[0, 1])
    if _corr >= 0.999:
        raise SystemExit(f"LEAK smell: {_c} |corr|={_corr:.4f} >= 0.999 vs target!")
    log(f"  leak-check {_c}: |corr|={_corr:.4f} — OK")
log("Pre-flight leakage checks PASSED")

# ─── Load node_0070 OOF for cheap-kill err-corr check ────────────────────────
N070_OOF_PATH = COMP_DIR / "nodes/node_0070/oof.npy"
n070_oof = np.load(N070_OOF_PATH).astype("float32")  # (577347, 3)
log(f"Loaded n070 OOF: shape={n070_oof.shape}")

# cheap-kill thresholds (per node plan)
KILL_ERRCORR_THRESH = 0.65   # err-corr vs n070 must be < this
KILL_BA_THRESH = 0.965       # solo BA must be >= this

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None  # will be set on first fold

log("Starting OOF loop …")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

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

    cat_cols_sorted = sorted(cat_cols)
    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]
        log(f"  n_features={X_tr_fold.shape[1]}  n_cat={len(cat_cols_sorted)}  n_num={len(num_cols_final)}")

    cfg_fold = {**CONFIG, "random_state": fold_seed, "device": str(DEVICE)}
    model = RealMLP_TD_Classifier(**cfg_fold)
    model.fit(
        X_tr_fold, y_tr_fold,
        X_val_fold, y_val_fold,
        cat_col_names=cat_cols_sorted,
        X_test=X_te_fold,
    )

    # OOF probabilities (indexed into original row positions via val_idx)
    oof_proba[val_idx] = model.best_val_probs_.astype("float32")

    # Test predictions — average across folds
    test_proba_accum += model.predict_proba(X_te_fold).astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # ── MANDATORY CHEAP-KILL: err-corr vs n070 + solo BA ──────────────────
        # Compute error-correlation vs node_0070 on fold-0 val rows
        val_preds_0140 = np.argmax(oof_proba[val_idx], axis=1)
        val_preds_n070 = np.argmax(n070_oof[val_idx], axis=1)
        # err-corr: correlation of ERRORS (wrong predictions)
        # err_0140[i] = 1 if wrong, 0 if correct
        err_0140 = (val_preds_0140 != y_all[val_idx]).astype(float)
        err_n070 = (val_preds_n070 != y_all[val_idx]).astype(float)
        if err_0140.std() == 0 or err_n070.std() == 0:
            err_corr = 1.0  # degenerate case
        else:
            err_corr = float(np.corrcoef(err_0140, err_n070)[0, 1])

        fold0_ba = per_fold_scores[0]
        log(f"  CHEAP-KILL CHECK: fold-0 BA={fold0_ba:.6f}  err-corr-vs-n070={err_corr:.4f}")
        log(f"  Kill thresholds: BA >= {KILL_BA_THRESH}, err-corr < {KILL_ERRCORR_THRESH}")
        print(f"fold0_ba={fold0_ba:.6f}", flush=True)
        print(f"fold0_errcorr_vs_n070={err_corr:.4f}", flush=True)

        kill_reason = []
        if err_corr >= KILL_ERRCORR_THRESH:
            kill_reason.append(f"err-corr={err_corr:.4f} >= {KILL_ERRCORR_THRESH}")
        if fold0_ba < KILL_BA_THRESH:
            kill_reason.append(f"BA={fold0_ba:.6f} < {KILL_BA_THRESH}")

        if kill_reason and os.environ.get("FORCE_FULL"):
            log(f"  CHEAP-KILL would trip ({'; '.join(kill_reason)}) but FORCE_FULL set "
                f"(human-directed LB probe) — continuing full 5-fold for a submittable artifact")
            kill_reason = []
        if kill_reason:
            log(f"  CHEAP-KILL TRIPPED: {'; '.join(kill_reason)} — STOPPING after fold-0")
            print(f"CHEAP_KILL_TRIPPED: {'; '.join(kill_reason)}", flush=True)
            # Save partial OOF (fold-0 only) for diagnostics
            np.save(NODE_DIR / "oof.npy", oof_proba)
            log(f"Saved partial oof.npy (fold-0 only) shape={oof_proba.shape}")
            # Compute final mean over completed folds
            mean_cv_partial = float(np.mean(per_fold_scores))
            log(f"partial cv (fold-0)={mean_cv_partial:.6f}")
            print(f"cv={mean_cv_partial:.6f}", flush=True)
            log("STOPPED by cheap-kill. No full OOF, no submission.")
            sys.exit(0)
        else:
            log(f"  CHEAP-KILL PASSED: err-corr={err_corr:.4f} < {KILL_ERRCORR_THRESH} AND "
                f"BA={fold0_ba:.6f} >= {KILL_BA_THRESH} — continuing full 5-fold")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF ────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ─────────────────────────────────────────────────────────
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
# Use the sorted column list from the last fold
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
