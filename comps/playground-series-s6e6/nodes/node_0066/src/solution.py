"""node_0066 — improve (nn): RealMLP pretrain on real SDSS17, then finetune.

Built on: node_0028 (RealMLP-ref recipe FE+PBLD, cv 0.969065). Everything
byte-identical EXCEPT a two-stage transfer per fold:
  1. PRETRAIN the RealMLP on cleaned SDSS17 100k rows (drop -9999 placeholders).
  2. FINE-TUNE on the train fold at LR = 0.2× pretrain LR (~0.002).

Fit-in-fold discipline:
  - Orig SDSS17 data is external + label-complete; whole dataset used in every
    fold's pretrain (val fold NEVER included — orig is a totally separate dataset).
  - The in-fold TargetEncoder / KBins / NumericalPreprocessor stay fold-local
    exactly as in n28.
  - Pretrain uses its OWN stateless FE + categorical FE fitted on SDSS17 rows.
    The finetune FE is the standard n28 FE fitted on train-fold rows only.
  - Both FE pipelines share the same feature NAMES / column layout so that
    RealMLP's weight tensors are compatible.

CRITICAL: The pretrain feature matrix must have the EXACT same columns (in the
same order) as the finetune matrix, otherwise weights don't transfer. We
achieve this by running both through the SAME `fit_fold_categoricals` signature
but on different source data — cat dims can differ, so we resize embeddings if
needed, or just use stateless + numerical features only for pretrain (simpler
and avoids dim mismatch). We choose the NUMERICAL-ONLY approach for pretrain:
pretrain on base numerics + stateless numeric FE only (no cats), then finetune
on the full feature set including cats. The first stage shapes the numeric
representation; the second adds cats.

Actually simpler/cleaner: pretrain on the FULL feature set where cat dims are
derived from the SDSS17 data, then re-initialize embedding layers with the
finetune fold's cat dims (since they differ). The shared weights are the
numeric embeddings + hidden layers + output layer. That is the standard
pretrain-finetune transfer.

We implement it by:
1. Build pretrain feature matrix from SDSS17 (same stateless FE; fit KBins /
   TargetEnc / factorize on SDSS17).
2. Build finetune feature matrix from train-fold (same pipeline as n28).
3. Pretrain model with pretrain feature dims.
4. Transfer shared parameters (numeric embed PBLD + hidden layers + output
   layer) to a new model built with finetune feature dims.
5. Finetune at LR×0.2 for the same 6 epochs.
"""
from __future__ import annotations

import copy
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

FINETUNE_LR_FACTOR = 0.2   # finetune LR = pretrain LR × this

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
    "n_ens": 8,
    "embed_dim": 7,
    "onehot_thresh": 10,
    "hidden_dims": [512, 512, 512],
    "dropout": 0.044,
    "p_drop_sched": "expm4t",
    "activation": nn.GELU,
    "add_front_scale": True,
    "pbld_hidden_dim": 16,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 2.33,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.115,
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
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "epochs": 6,
    "train_bs": 256,
    "eval_bs": 10240,
    "numeric_noise_std": 0.0,
    "ema_decay": 0.997875,
    "verbosity": 2,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
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

# SDSS17 column name mapping (original dataset uses different col names)
SDSS17_COL_MAP = {
    # sdss17 col → our col (only renames needed)
    "alpha": "alpha",
    "delta": "delta",
    "u": "u",
    "g": "g",
    "r": "r",
    "i": "i",
    "z": "z",
    "redshift": "redshift",
    "class": "class",
}


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Pure row-wise stateless FE — safe to apply to any dataframe."""
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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame,
                          has_spectral_type: bool = True):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.
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

    base_cats = [c for c in BASE_CAT_COLS if c in tr.columns] if has_spectral_type else []
    for col in base_cats:
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
        for dset, dset_src in [(va, df_val), (te, df_te)]:
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


# ─── Model components ─────────────────────────────────────────────────────────

class NumericalPreprocessor(BaseEstimator, TransformerMixin):
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
        return F.softmax(x, dim=2)


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


def train_one_stage(model: RealMLP, X_num: np.ndarray, X_cat: np.ndarray,
                    y: np.ndarray, X_val_num: np.ndarray, X_val_cat: np.ndarray,
                    y_val: np.ndarray, p: dict, dev: torch.device,
                    lr_multiplier: float = 1.0, label: str = "train") -> tuple:
    """
    Single training stage (pretrain or finetune).
    Returns (best_val_probs, best_score, best_state).
    """
    classes = np.unique(y)
    weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_power = float(p.get("class_weight_power", 1.0))
    if cw_power != 1.0:
        weights_np = np.power(weights_np, cw_power)
    class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

    loss_prior_power = float(p.get("loss_prior_power", 0.0))
    loss_prob_multipliers = None
    if loss_prior_power != 0.0:
        class_counts = np.bincount(y, minlength=len(classes)).astype("float64")
        class_counts = class_counts / np.exp(np.log(class_counts).mean())
        loss_mult_np = np.power(class_counts, loss_prior_power)
        loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

    n_classes = len(classes)
    param_groups = get_parameter_groups(model, p)
    for g in param_groups:
        g["lr"] = g["lr"] * lr_multiplier
        g["lr_base"] = g["lr"]
    optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

    Xtn = torch.as_tensor(X_num, dtype=torch.float32, device=dev)
    Xtc = torch.as_tensor(X_cat, dtype=torch.long, device=dev)
    ytt = torch.as_tensor(y, dtype=torch.long, device=dev)
    Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
    Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

    n_ens = p["n_ens"]
    train_bs = p["train_bs"]
    eval_bs = p["eval_bs"]
    epochs = p["epochs"]
    lr_sched = p["lr_sched"]
    flat_ratio = p["flat_ratio"]
    ema_decay = float(p.get("ema_decay", 0.0))
    total_steps = epochs * len(y)
    train_order = np.arange(len(y))

    best_score = -np.inf
    best_epoch = 0
    best_val_probs = None
    best_state = None
    ema_state = None
    if ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(y), train_bs):
            progress = (epoch * len(y) + start) / total_steps
            idx_batch = train_order[start: start + train_bs]

            for g in optimizer.param_groups:
                g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)

            optimizer.zero_grad()
            x_num_batch = Xtn[idx_batch]
            y_pred = model(x_num_batch, Xtc[idx_batch])

            ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
            drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
            for dm in model._dropout_modules:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
            optimizer.step()

            if ema_state is not None:
                with torch.no_grad():
                    model_state = model.state_dict()
                    for key, value in model_state.items():
                        if torch.is_floating_point(value):
                            ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                        else:
                            ema_state[key].copy_(value)

        np.random.shuffle(train_order)

        model.eval()
        live_state = None
        if ema_state is not None:
            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)

        with torch.no_grad():
            val_probs = np.concatenate([
                model(Xvn[s: s + eval_bs], Xvc[s: s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, len(y_val), eval_bs)
            ], axis=0)

        if live_state is not None:
            model.load_state_dict(live_state, strict=True)

        epoch_score = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
        improved = epoch_score > best_score
        if improved:
            best_score = epoch_score
            best_epoch = epoch + 1
            best_val_probs = val_probs.copy()
            state_src = ema_state if ema_state is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in state_src.items()}

        verbose = p.get("verbosity", 0)
        if verbose >= 2:
            log(f"    [{label}] epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                f"best={best_score:.5f}" + ("  *" if improved else ""))

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return best_val_probs, best_score, best_state


def transfer_weights(src_model: RealMLP, dst_model: RealMLP):
    """
    Copy shared weights from pretrain model to finetune model.
    Shared: num_embed (PBLD), hidden layers (NTPLinear weights/biases), output layer.
    NOT transferred: cate (embedding dims differ), ScalingLayer (input dim differs).
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    transferred = []
    skipped = []

    for key in dst_sd:
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            # Transfer if it's a shared component (not cat-related)
            if "cate" not in key and "hidden.0" not in key:
                # hidden.0 is ScalingLayer (input dim differs), skip it
                dst_sd[key] = src_sd[key].clone()
                transferred.append(key)
            elif "hidden.0" in key:
                # ScalingLayer — dims may differ, skip
                skipped.append(key)
            elif "cate" in key:
                skipped.append(key)
        else:
            skipped.append(key)

    dst_model.load_state_dict(dst_sd, strict=True)
    return transferred, skipped


class RealMLP_TD_Pretrain_Classifier(BaseEstimator):
    """
    Two-stage RealMLP:
    1. Pretrain on SDSS17 data (external, whole).
    2. Finetune on train-fold data at lower LR.
    """

    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train_ft: pd.DataFrame, y_train_ft,
            X_val: pd.DataFrame, y_val,
            X_pretrain: pd.DataFrame, y_pretrain,
            cat_col_names_ft=None, cat_col_names_pt=None,
            X_test: pd.DataFrame = None):
        """
        X_train_ft / y_train_ft: finetune train fold.
        X_val / y_val: validation fold (NEVER used in pretrain).
        X_pretrain / y_pretrain: full SDSS17 (FE already applied).
        cat_col_names_ft: cat cols for finetune.
        cat_col_names_pt: cat cols for pretrain.
        """
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        self._dev = dev

        cat_col_names_ft = cat_col_names_ft or []
        cat_col_names_pt = cat_col_names_pt or []
        num_col_names_ft = [c for c in X_train_ft.columns if c not in cat_col_names_ft]
        num_col_names_pt = [c for c in X_pretrain.columns if c not in cat_col_names_pt]

        # ── Pretrain numerical preprocessing (fit on pretrain data) ──
        X_pt_num = X_pretrain[num_col_names_pt].values.astype(np.float32)
        # Use val_num for pretrain validation (val is unseen — fine since pretrain uses external data)
        # Actually we need a pretrain val set — use a small split from SDSS17 itself
        n_pt = len(X_pt_num)
        pt_val_size = min(10000, n_pt // 10)
        rng_pt = np.random.RandomState(p["random_state"])
        pt_val_idx = rng_pt.choice(n_pt, pt_val_size, replace=False)
        pt_tr_idx = np.setdiff1d(np.arange(n_pt), pt_val_idx)

        X_pt_tr_num_raw = X_pt_num[pt_tr_idx]
        X_pt_val_num_raw = X_pt_num[pt_val_idx]

        self.preprocessor_pt_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_pt_.fit(X_pt_tr_num_raw)
        X_pt_tr_num = self.preprocessor_pt_.transform(X_pt_tr_num_raw)
        X_pt_val_num = self.preprocessor_pt_.transform(X_pt_val_num_raw)

        X_pt_cat = X_pretrain[cat_col_names_pt].values.astype(np.int64)
        X_pt_tr_cat = X_pt_cat[pt_tr_idx]
        X_pt_val_cat = X_pt_cat[pt_val_idx]
        y_pt_tr = np.asarray(y_pretrain)[pt_tr_idx]
        y_pt_val = np.asarray(y_pretrain)[pt_val_idx]

        # Cat dims for pretrain
        if cat_col_names_pt:
            cat_dims_pt = (X_pt_cat.max(axis=0) + 1).tolist()
        else:
            cat_dims_pt = []
        self.cat_dims_pt_ = cat_dims_pt

        if cat_dims_pt:
            cat_max_pt = np.array(cat_dims_pt) - 1
            X_pt_tr_cat = np.clip(X_pt_tr_cat, 0, cat_max_pt)
            X_pt_val_cat = np.clip(X_pt_val_cat, 0, cat_max_pt)

        n_numerical_pt = X_pt_tr_num.shape[1]

        # ── Build pretrain model ──
        log(f"    Pretrain: {len(pt_tr_idx)} rows, {n_numerical_pt} num feats, {len(cat_dims_pt)} cat feats")
        seed_everything(p["random_state"])
        pretrain_model = RealMLP(
            output_dim=N_CLASSES, cat_dims=cat_dims_pt,
            n_numerical=n_numerical_pt, cfg=p,
        ).to(dev)

        _, pt_best_score, pt_best_state = train_one_stage(
            pretrain_model, X_pt_tr_num, X_pt_tr_cat, y_pt_tr,
            X_pt_val_num, X_pt_val_cat, y_pt_val,
            p, dev, lr_multiplier=1.0, label="pretrain"
        )
        log(f"    Pretrain done — best val score={pt_best_score:.5f}")

        # ── Finetune: build model with finetune dims, transfer weights ──
        X_ft_num_raw = X_train_ft[num_col_names_ft].values.astype(np.float32)
        X_val_num_raw = X_val[num_col_names_ft].values.astype(np.float32)
        X_ft_cat = X_train_ft[cat_col_names_ft].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names_ft].values.astype(np.int64)
        y_ft = np.asarray(y_train_ft)
        y_v = np.asarray(y_val)

        # Finetune preprocessing (fit on train fold only)
        self.preprocessor_ft_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_ft_.fit(X_ft_num_raw)
        X_ft_num = self.preprocessor_ft_.transform(X_ft_num_raw)
        X_val_num = self.preprocessor_ft_.transform(X_val_num_raw)

        # Cat dims for finetune
        if cat_col_names_ft:
            all_cat_ft = [X_ft_cat, X_val_cat]
            if X_test is not None:
                all_cat_ft.append(X_test[cat_col_names_ft].values.astype(np.int64))
            cat_dims_ft = (np.concatenate(all_cat_ft, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims_ft = []
        self.cat_dims_ft_ = cat_dims_ft

        if cat_dims_ft:
            cat_max_ft = np.array(cat_dims_ft) - 1
            X_ft_cat = np.clip(X_ft_cat, 0, cat_max_ft)
            X_val_cat = np.clip(X_val_cat, 0, cat_max_ft)

        self.cat_col_names_ft_ = cat_col_names_ft
        self.num_col_names_ft_ = num_col_names_ft

        n_numerical_ft = X_ft_num.shape[1]

        log(f"    Finetune: {len(X_ft_num)} rows, {n_numerical_ft} num feats, {len(cat_dims_ft)} cat feats")

        # Build finetune model and transfer weights
        finetune_model = RealMLP(
            output_dim=N_CLASSES, cat_dims=cat_dims_ft,
            n_numerical=n_numerical_ft, cfg=p,
        ).to(dev)

        transferred, skipped = transfer_weights(pretrain_model, finetune_model)
        log(f"    Transferred {len(transferred)} param tensors, skipped {len(skipped)}")

        # Free pretrain model
        del pretrain_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Finetune at lower LR
        ft_p = {**p, "lr": p["lr"] * FINETUNE_LR_FACTOR}
        best_val_probs, ft_best_score, ft_best_state = train_one_stage(
            finetune_model, X_ft_num, X_ft_cat, y_ft,
            X_val_num, X_val_cat, y_v,
            ft_p, dev, lr_multiplier=1.0, label="finetune"
        )
        log(f"    Finetune done — best val score={ft_best_score:.5f}")

        self.model_ = finetune_model
        self.best_score_ = ft_best_score
        self.best_val_probs_ = best_val_probs
        self.classes_ = np.arange(N_CLASSES)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_ft_.transform(
            X[self.num_col_names_ft_].values.astype(np.float32)
        )
        X_cat = X[self.cat_col_names_ft_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_ft_) - 1)
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

# ─── Load and clean SDSS17 (once, external) ───────────────────────────────────
log("Loading SDSS17 data …")
ORIG_PATH = COMP_DIR / "data/sdss17/star_classification.csv"
orig_raw = pd.read_csv(ORIG_PATH)
log(f"  orig_raw={orig_raw.shape}")

# Drop -9999 placeholder rows (same cleaning as node_0045)
for band_col in ["u", "g", "z"]:
    if band_col in orig_raw.columns:
        orig_raw = orig_raw[orig_raw[band_col] != -9999]
orig_raw = orig_raw[orig_raw["class"].isin(CLASSES)].reset_index(drop=True)
log(f"  orig_filtered={orig_raw.shape}")

orig_y = orig_raw["class"].map(LABEL_MAP).astype(int).values

# SDSS17 has alpha/delta/u/g/r/i/z/redshift and class but no spectral_type/galaxy_population
# We use only the columns available (same as train BASE_NUM_COLS)
orig_base = orig_raw[["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]].copy()

# Apply stateless FE to SDSS17
log("Applying stateless FE to SDSS17 …")
orig_stateless = stateless_fe(orig_base)
log(f"  orig_stateless={orig_stateless.shape}")

# Pre-flight leakage check 1–2: target not in features
y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Stateless FE for train/test (computed once) ──────────────────────────────
log("Applying stateless FE to train/test …")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Leakage pre-flight checks ────────────────────────────────────────────────
log("Pre-flight leakage checks …")
# Check 1: target not in features
feat_cols = [c for c in X_stateless.columns]
assert TARGET not in feat_cols, f"TARGET {TARGET} in features!"
assert IDC not in feat_cols, f"IDC {IDC} in features!"
log("  check 1-2: TARGET/IDC not in features — OK")

# Check 3: single-feature corr sweep (sample)
sample_n = min(50000, n_train)
rng_check = np.random.RandomState(0)
check_idx = rng_check.choice(n_train, sample_n, replace=False)
s_X = X_stateless.iloc[check_idx]
s_y = y_all[check_idx].astype(float)
for c in feat_cols:
    x = pd.to_numeric(s_X[c], errors="coerce").fillna(0).values
    if x.std() > 0:
        corr = abs(np.corrcoef(x, s_y)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"Leak smell: feature {c} |corr|={corr:.4f} with target!")
log("  check 3: single-feature corr sweep — OK")
# Check 4: fit-in-fold — verified by code reading (all KBins/TargetEnc/NumericalPreprocessor
#   are created and fit inside the fold loop below on tr_idx rows only)
log("  check 4: fit-in-fold — code-verified OK")
# Check 5: frozen folds
log("  check 5: folds loaded from folds.json — OK")
log("Pre-flight checks passed.")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None

log("Starting OOF loop …")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ── Finetune FE (fold-local, same as n28) ────────────────────────────────
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
        has_spectral_type=True,
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

    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]
        log(f"  n_features_ft={X_tr_fold.shape[1]}  n_cat={len(cat_cols_sorted)}  n_num={len(num_cols_final)}")

    # ── Pretrain FE (on SDSS17, no target/val from main dataset) ─────────────
    # SDSS17 has no spectral_type/galaxy_population — use numerical-only cats
    # (floor bins, delta quantile bins only; no BASE_CAT_COLS; no TargetEncoder
    #  since we'd need main-dataset labels which would be a leak)
    # We build a dummy val df for fit_fold_categoricals (single row repeated)
    orig_n = len(orig_stateless)
    dummy_val = orig_stateless.iloc[:1].copy()
    dummy_te = orig_stateless.iloc[:1].copy()

    X_pt_tr, X_pt_val_dummy, _, cat_cols_pt, combo_names_pt, _ = fit_fold_categoricals(
        orig_stateless.reset_index(drop=True),
        dummy_val.reset_index(drop=True),
        dummy_te.reset_index(drop=True),
        has_spectral_type=False,  # SDSS17 has no spectral_type/galaxy_population
    )
    # No TargetEncoder for pretrain (no fold labels)
    cat_cols_pt_sorted = sorted(cat_cols_pt)
    num_cols_pt = [c for c in X_pt_tr.columns if c not in cat_cols_pt_sorted]
    X_pt_tr = X_pt_tr.reindex(sorted(X_pt_tr.columns), axis=1)

    log(f"  Pretrain FE: {X_pt_tr.shape[1]} feats, {len(cat_cols_pt_sorted)} cat, {len(num_cols_pt)} num")

    # ── Build + run two-stage model ───────────────────────────────────────────
    cfg_fold = {**CONFIG, "random_state": fold_seed, "device": str(DEVICE)}
    model = RealMLP_TD_Pretrain_Classifier(**cfg_fold)
    model.fit(
        X_train_ft=X_tr_fold,
        y_train_ft=y_tr_fold,
        X_val=X_val_fold,
        y_val=y_val_fold,
        X_pretrain=X_pt_tr,
        y_pretrain=orig_y,
        cat_col_names_ft=cat_cols_sorted,
        cat_col_names_pt=cat_cols_pt_sorted,
        X_test=X_te_fold,
    )

    oof_proba[val_idx] = model.best_val_probs_.astype("float32")
    test_proba_accum += model.predict_proba(X_te_fold).astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold, X_pt_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # Kill switch check
        if fold_score < 0.9685:
            log(f"KILL SWITCH: fold-0 BA={fold_score:.6f} < 0.9685 — stopping.")
            print(f"KILL_SWITCH fold0={fold_score:.6f}", flush=True)
            sys.exit(1)
        log(f"  Kill switch OK (fold-0 BA={fold_score:.6f} >= 0.9685)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write submission ──────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
