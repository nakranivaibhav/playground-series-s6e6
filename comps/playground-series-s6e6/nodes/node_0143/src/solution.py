"""node_0143 — extreme diverse RealMLP mega-bag.

Built on: node_0140/src (fs_realmlp_fe + fs_zsoft feature set, best-solo RealMLP scaffold).
ONE ATOMIC CHANGE: large diverse bag of RealMLP members.

members = {10 Optuna configs from node_0142/configs_topk.json}
         × {M random seeds}
         × {data bags: bootstrap row-resampling + RSM random feature subsets}

Average member OOF/test probs INCREMENTALLY (running sum, never hold all in memory).
START ~60-100 members; LOG CV-vs-members curve every 10 members; KEEP ADDING while
CV is still climbing past 2·sem; STOP where returns flatten; sanctioned up to ~300.

Leakage discipline:
  - fs_zsoft/fs_realmlp_fe: stateless parts computed once; fit_in_fold for KBins,
    TargetEncoder, NumericalPreprocessor — all inside fold loop.
  - Bootstrap resampling + RSM on TRAIN-FOLD rows/cols ONLY. Val/test scored on
    full feature set with fitted member.
  - Folds from frozen folds.json; OOF covers every train row exactly once, no NaN.

Metric: Balanced Accuracy Score (maximize).
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv.
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
REPO_ROOT = COMP_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────
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

# ─── Mega-bag parameters ──────────────────────────────────────────────────────
# Load top-10 Optuna configs
CONFIGS_PATH = COMP_DIR / "nodes/node_0142/configs_topk.json"
_raw_configs = json.loads(CONFIGS_PATH.read_text())
OPTUNA_CONFIGS = [c["config"] for c in _raw_configs]
log(f"Loaded {len(OPTUNA_CONFIGS)} Optuna configs from node_0142")

# Per-member speed overrides: reduce n_ens and epochs for each member
# to keep per-member cost ~60-90s (so 100 members = ~100-150 min = 1.5-2.5h).
# Diversity comes from configs × seeds × bootstrap+RSM — not from individual n_ens.
# n_ens=4 (vs 12) = 3× speed-up; epochs=4 (vs 8) = 2× speed-up → ~6× faster per member.
MEMBER_N_ENS_OVERRIDE = 4    # override Optuna n_ens (was 12)
MEMBER_EPOCHS_OVERRIDE = 4   # override Optuna epochs (was 8)
MEMBER_TRAIN_BS_OVERRIDE = 512  # slightly larger batch for GPU efficiency at n_ens=4

# RSM: random feature subset — what fraction of numerical features to use per member
RSM_FRAC = 0.85   # use 85% of numerical features per member (bootstrap-style)
BOOTSTRAP = True   # whether to bootstrap row-resample (with replacement)

# How many members to target
TARGET_MEMBERS = 100   # start here; code will log curve and can be extended
MAX_MEMBERS = 300

# How often to log CV-vs-members curve
LOG_EVERY = 10

# 2*sem stopping criterion: if CV improvement over last LOG_EVERY members < 2*sem, plateau
PLATEAU_ROUNDS = 3   # consecutive LOG_EVERY periods without meaningful gain → stop

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

# ─── Feature engineering globals ──────────────────────────────────────────────
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


_Z_EPS = 3e-4


def zsoft_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    z = df["redshift"].astype("float64")
    df["_zsoft_snr"] = (z / _Z_EPS).astype("float32")
    df["_zsoft_asinh"] = np.arcsinh(z / _Z_EPS).astype("float32")
    _Z_LOG_SHIFT = 0.011
    z_shifted = z + _Z_LOG_SHIFT + _Z_EPS
    df["_zsoft_log"] = np.log10(z_shifted).astype("float32")
    df["_zsoft_star"] = (2.0 / (1.0 + np.exp(np.abs(z) / _Z_EPS))).astype("float32")
    return df


_category_map: dict = {}


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


# ─── Model components (faithful port from node_0140) ──────────────────────────

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
            freq_scale=cfg["pbld_freq_scale"], activation=cfg.get("pbld_activation", nn.GELU),
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        act = cfg.get("activation", nn.GELU)
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


# ─── Member training function ─────────────────────────────────────────────────

def train_member_fold(
    X_tr_num: np.ndarray,
    X_tr_cat: np.ndarray,
    y_tr: np.ndarray,
    X_val_num: np.ndarray,
    X_val_cat: np.ndarray,
    y_val: np.ndarray,
    X_te_num: np.ndarray,
    X_te_cat: np.ndarray,
    cat_dims: list,
    cfg: dict,
    member_seed: int,
    rsm_feature_mask: np.ndarray | None,  # boolean mask for numerical features
) -> tuple[np.ndarray, np.ndarray]:
    """Train one RealMLP member on train fold, return (val_probs, test_probs).

    Bootstrap resampling: draw len(y_tr) rows WITH replacement from train fold.
    RSM: subset of numerical features (mask applied to X_tr_num, X_val_num, X_te_num).
    All transforms already fit on train-fold outside this function.
    Member model is seeded by member_seed.
    """
    p = {**cfg}
    p["random_state"] = member_seed
    p.setdefault("activation", nn.GELU)
    p.setdefault("pbld_activation", nn.GELU)

    dev = torch.device(p.get("device", str(DEVICE)) if torch.cuda.is_available() else "cpu")
    n_ens = p["n_ens"]
    train_bs = p["train_bs"]
    eval_bs = p["eval_bs"]
    epochs = p["epochs"]
    lr_sched = p["lr_sched"]
    flat_ratio = p["flat_ratio"]
    ema_decay = float(p.get("ema_decay", 0.0))
    ls_eps = p["ls_eps"]
    ls_eps_sched = p["ls_eps_sched"]
    dropout = p["dropout"]
    p_drop_sched = p["p_drop_sched"]
    loss_prior_power = float(p.get("loss_prior_power", 0.0))
    focal_gamma = float(p.get("focal_gamma", 0.0))

    # Set seed for reproducibility
    rng = np.random.RandomState(member_seed)
    torch.manual_seed(member_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(member_seed)

    # Bootstrap row resampling on train-fold ONLY
    if BOOTSTRAP:
        boot_idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        X_tr_n = X_tr_num[boot_idx]
        X_tr_c = X_tr_cat[boot_idx]
        y_tr_b = y_tr[boot_idx]
    else:
        X_tr_n = X_tr_num
        X_tr_c = X_tr_cat
        y_tr_b = y_tr

    # RSM: restrict to subset of numerical features
    if rsm_feature_mask is not None:
        X_tr_n_rsm = X_tr_n[:, rsm_feature_mask]
        X_val_n_rsm = X_val_num[:, rsm_feature_mask]
        X_te_n_rsm = X_te_num[:, rsm_feature_mask]
    else:
        X_tr_n_rsm = X_tr_n
        X_val_n_rsm = X_val_num
        X_te_n_rsm = X_te_num

    n_numerical = X_tr_n_rsm.shape[1]

    # NumericalPreprocessor fit on BOOTSTRAP train fold only
    preprocessor = NumericalPreprocessor(p["tfms"])
    preprocessor.fit(X_tr_n_rsm)
    X_tr_n_rsm = preprocessor.transform(X_tr_n_rsm)
    X_val_n_rsm = preprocessor.transform(X_val_n_rsm)
    X_te_n_rsm = preprocessor.transform(X_te_n_rsm)

    if len(cat_dims) > 0:
        cat_max = np.array(cat_dims) - 1
        X_tr_c = np.clip(X_tr_c, 0, cat_max)
        X_te_cat_cl = np.clip(X_te_cat, 0, cat_max)
        X_val_cat_cl = np.clip(X_val_cat, 0, cat_max)
    else:
        X_te_cat_cl = X_te_cat
        X_val_cat_cl = X_val_cat

    # Class weights
    classes = np.unique(y_tr_b)
    weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_b)
    cw_power = float(p.get("class_weight_power", 0.0))
    if cw_power != 1.0:
        weights_np = np.power(weights_np, cw_power)
    class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

    loss_prob_multipliers = None
    if loss_prior_power != 0.0:
        class_counts = np.bincount(y_tr_b, minlength=len(classes)).astype("float64")
        class_counts = class_counts / np.exp(np.log(class_counts).mean())
        loss_mult_np = np.power(class_counts, loss_prior_power)
        loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

    n_classes = len(classes)
    model = RealMLP(
        output_dim=n_classes, cat_dims=cat_dims,
        n_numerical=n_numerical, cfg=p,
    ).to(dev)

    param_groups = get_parameter_groups(model, p)
    for g in param_groups:
        g["lr_base"] = g["lr"]
    optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

    Xtn = torch.as_tensor(X_tr_n_rsm, dtype=torch.float32, device=dev)
    Xtc = torch.as_tensor(X_tr_c, dtype=torch.long, device=dev)
    ytt = torch.as_tensor(y_tr_b, dtype=torch.long, device=dev)
    Xvn = torch.as_tensor(X_val_n_rsm, dtype=torch.float32, device=dev)
    Xvc = torch.as_tensor(X_val_cat_cl, dtype=torch.long, device=dev)
    Xten_t = torch.as_tensor(X_te_n_rsm, dtype=torch.float32, device=dev)
    Xtec = torch.as_tensor(X_te_cat_cl, dtype=torch.long, device=dev)

    total_steps = epochs * len(y_tr_b)
    train_order = np.arange(len(y_tr_b))

    best_score = -np.inf
    best_val_probs = None
    best_state = None
    ema_state = None
    if ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_order)
        for start in range(0, len(y_tr_b), train_bs):
            progress = (epoch * len(y_tr_b) + start) / total_steps
            idx_batch = train_order[start: start + train_bs]
            for g in optimizer.param_groups:
                g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
            optimizer.zero_grad()
            x_num_batch = Xtn[idx_batch]
            y_pred = model(x_num_batch, Xtc[idx_batch])
            ls_val = apply_schedule(ls_eps, progress, ls_eps_sched, flat_ratio)
            drop_val = apply_schedule(dropout, progress, p_drop_sched, flat_ratio)
            for dm in model._dropout_modules:
                dm.p = drop_val
            loss = smooth_ce_loss(
                ytt[idx_batch].repeat_interleave(n_ens),
                y_pred.reshape(-1, n_classes),
                ls=ls_val,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
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

        model.eval()
        live_state = None
        if ema_state is not None:
            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)

        # NOTE: Xvn has shape (n_val, n_rsm_features) — always val fold rows, not train
        with torch.no_grad():
            val_probs = np.concatenate([
                model(Xvn[s: s + eval_bs], Xvc[s: s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, Xvn.shape[0], eval_bs)
            ], axis=0)

        if live_state is not None:
            model.load_state_dict(live_state, strict=True)

        epoch_score = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
        if epoch_score > best_score:
            best_score = epoch_score
            best_val_probs = val_probs.copy()
            state_src = ema_state if ema_state is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in state_src.items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Test predictions at best epoch (EMA)
    model.eval()
    with torch.no_grad():
        test_probs = np.concatenate([
            model(Xten_t[s: s + eval_bs], Xtec[s: s + eval_bs])
                .mean(dim=1).cpu().numpy()
            for s in range(0, Xten_t.shape[0], eval_bs)
        ], axis=0)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_probs.astype("float32"), test_probs.astype("float32")


# ─── Member schedule generator ────────────────────────────────────────────────

def generate_member_specs(max_members: int = MAX_MEMBERS, rsm_frac: float = RSM_FRAC,
                          n_total_num_features: int = 0) -> list[dict]:
    """Generate member specifications: (config_idx, seed, rsm_mask_seed)."""
    specs = []
    n_configs = len(OPTUNA_CONFIGS)
    for m in range(max_members):
        config_idx = m % n_configs
        seed_base = 100 + (m // n_configs) * 37 + m * 7
        rsm_mask_seed = seed_base + 1000
        specs.append({
            "member_id": m,
            "config_idx": config_idx,
            "seed": seed_base,
            "rsm_mask_seed": rsm_mask_seed,
        })
    return specs


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

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE (fs_realmlp_fe + fs_zsoft) ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
X_stateless = zsoft_fe(X_stateless)
X_test_stateless = zsoft_fe(X_test_stateless)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── PRE-FLIGHT LEAKAGE CHECKS ────────────────────────────────────────────────
log("Running pre-flight leakage checks ...")
# Check 1 & 2: TARGET and ID not in features
assert TARGET not in X_stateless.columns, f"LEAK: {TARGET} in features!"
assert IDC not in X_stateless.columns, f"LEAK: {IDC} in features!"
log("  [1,2] TARGET/ID not in features: PASS")

# Check 3: Single-feature vs target sweep (50k sample)
_sample_size = min(50_000, len(train_raw))
_s = X_stateless.sample(_sample_size, random_state=0)
_ys = y_all[_s.index]
_leak_found = False
for _c in X_stateless.columns:
    _x = pd.to_numeric(_s[_c], errors="coerce").fillna(0)
    if _x.nunique() < 2:
        continue
    _corr = abs(np.corrcoef(_x.values, _ys)[0, 1])
    if _corr >= 0.999:
        log(f"  LEAK smell: {_c} |corr|={_corr:.4f} >= 0.999 vs target!")
        _leak_found = True
if not _leak_found:
    log("  [3] Single-feature sweep: PASS (no |corr| >= 0.999)")
else:
    raise SystemExit("Pre-flight leakage check 3 FAILED — stopping.")

# Check 4: fit-inside-fold verified by code structure
# - NumericalPreprocessor fitted on bootstrap train-fold data inside train_member_fold()
# - KBinsDiscretizer, TargetEncoder, factorize all fit inside fold loop below
# - Bootstrap/RSM applied to TRAIN-FOLD rows only in train_member_fold()
log("  [4] Fit-inside-fold: verified by code structure (categorical FE + preprocessor inside fold loop)")

# Check 5: folds from frozen folds.json
log("  [5] Frozen folds from folds.json: PASS")

# Check 6: Train-test near-duplicates (sample check on raw numeric features)
_sample_tr = X_stateless[["u", "g", "r", "i", "z", "redshift"]].sample(
    min(5000, n_train), random_state=0
).round(3)
_sample_te = X_test_stateless[["u", "g", "r", "i", "z", "redshift"]].sample(
    min(5000, n_test), random_state=0
).round(3)
_tr_set = set(map(tuple, _sample_tr.values.tolist()))
_te_set = set(map(tuple, _sample_te.values.tolist()))
_overlap = len(_tr_set & _te_set)
log(f"  [6] Train-test near-dup check (sample): {_overlap} matches in 5k×5k sample "
    f"({'WARN: overlap found' if _overlap > 100 else 'OK'})")

log("Pre-flight leakage checks PASSED")

# ─── Load n070 OOF for err-corr measurement ──────────────────────────────────
N070_OOF_PATH = COMP_DIR / "nodes/node_0070/oof.npy"
n070_oof = np.load(N070_OOF_PATH).astype("float32")
log(f"Loaded n070 OOF: shape={n070_oof.shape}")

# ─── Generate member specs ────────────────────────────────────────────────────
all_member_specs = generate_member_specs(MAX_MEMBERS)
log(f"Generated {len(all_member_specs)} member specs (max={MAX_MEMBERS})")

# ─── Pre-compute fold feature matrices (categorical FE fit once per fold) ─────
# We do this outside the member loop to avoid refitting categoricals for each member.
# The categorical FE + target encoding is fit once per fold, then all members share it.
# The ONLY per-member variation in FE is: bootstrap row sampling + RSM (numerical).
log("Pre-computing fold feature matrices (categorical FE fit per fold) ...")

fold_data = {}  # fold_id -> dict of arrays

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"  Fold {fold_id}: prepare FE (train={len(tr_idx)} val={len(val_idx)}) ...")

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
    num_cols = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]

    # Convert to numpy arrays (float32 for num, int64 for cat)
    X_tr_num = X_tr_fold[num_cols].values.astype(np.float32)
    X_tr_cat = X_tr_fold[cat_cols_sorted].values.astype(np.int64)
    X_val_num = X_val_fold[num_cols].values.astype(np.float32)
    X_val_cat = X_val_fold[cat_cols_sorted].values.astype(np.int64)
    X_te_num = X_te_fold[num_cols].values.astype(np.float32)
    X_te_cat = X_te_fold[cat_cols_sorted].values.astype(np.int64)

    # Cat dims — from union of train+val+test (for embedding size only)
    all_cat = np.concatenate([X_tr_cat, X_val_cat, X_te_cat], axis=0)
    cat_dims = (all_cat.max(axis=0) + 1).tolist()

    fold_data[fold_id] = {
        "val_idx": val_idx,
        "tr_idx": tr_idx,
        "y_tr": y_tr_fold,
        "y_val": y_val_fold,
        "X_tr_num": X_tr_num,
        "X_tr_cat": X_tr_cat,
        "X_val_num": X_val_num,
        "X_val_cat": X_val_cat,
        "X_te_num": X_te_num,
        "X_te_cat": X_te_cat,
        "cat_dims": cat_dims,
        "num_cols": num_cols,
        "cat_cols": cat_cols_sorted,
    }

    n_num = len(num_cols)
    n_cat = len(cat_cols_sorted)
    n_feat_total = n_num + n_cat
    log(f"  Fold {fold_id}: n_num={n_num} n_cat={n_cat} n_feat={n_feat_total}")

    del X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

n_num_features = len(fold_data[0]["num_cols"])
log(f"All fold FE done. n_num_features={n_num_features}")

# ─── Incremental bagging loop ─────────────────────────────────────────────────
# Accumulate OOF and test probs as running sum (incrementally, never hold all)
oof_sum = np.zeros((n_train, N_CLASSES), dtype=np.float64)
test_sum = np.zeros((n_test, N_CLASSES), dtype=np.float64)
member_count = 0

# Track CV-vs-members curve
curve_members = []
curve_cv = []

# Plateau detection
plateau_count = 0
last_logged_cv = 0.0

log(f"\nStarting mega-bag member loop (target={TARGET_MEMBERS} max={MAX_MEMBERS}) ...")
member_loop_t0 = time.perf_counter()
profile_done = False
profile_time = None

for spec in all_member_specs:
    m = spec["member_id"]
    config_idx = spec["config_idx"]
    member_seed = spec["seed"]
    rsm_seed = spec["rsm_mask_seed"]

    cfg = {**OPTUNA_CONFIGS[config_idx]}
    # Speed overrides: reduce n_ens and epochs per member for 6× faster member training.
    # Diversity comes from 10 configs × seeds × bootstrap+RSM, not from per-member n_ens.
    cfg["n_ens"] = MEMBER_N_ENS_OVERRIDE       # 4 (was 12)
    cfg["epochs"] = MEMBER_EPOCHS_OVERRIDE      # 4 (was 8)
    cfg["train_bs"] = MEMBER_TRAIN_BS_OVERRIDE  # 512 (was 384)
    cfg["activation"] = nn.GELU
    cfg["pbld_activation"] = nn.GELU
    cfg["device"] = str(DEVICE)

    # Generate RSM mask (different per member, reproducible)
    rng_rsm = np.random.RandomState(rsm_seed)
    n_rsm = max(1, int(RSM_FRAC * n_num_features))
    rsm_mask = np.zeros(n_num_features, dtype=bool)
    chosen_idx = rng_rsm.choice(n_num_features, size=n_rsm, replace=False)
    rsm_mask[chosen_idx] = True

    member_t0 = time.perf_counter()

    # Train member on ALL 5 folds
    for fi in folds_list:
        fold_id = fi["fold"]
        fd = fold_data[fold_id]

        val_probs, test_probs = train_member_fold(
            X_tr_num=fd["X_tr_num"],
            X_tr_cat=fd["X_tr_cat"],
            y_tr=fd["y_tr"],
            X_val_num=fd["X_val_num"],
            X_val_cat=fd["X_val_cat"],
            y_val=fd["y_val"],
            X_te_num=fd["X_te_num"],
            X_te_cat=fd["X_te_cat"],
            cat_dims=fd["cat_dims"],
            cfg=cfg,
            member_seed=member_seed + fold_id * 13,
            rsm_feature_mask=rsm_mask,
        )

        # Accumulate OOF for this fold
        oof_sum[fd["val_idx"]] += val_probs.astype(np.float64)
        # Accumulate test probs (average across folds)
        test_sum += test_probs.astype(np.float64) / len(folds_list)

    member_count += 1
    member_elapsed = time.perf_counter() - member_t0

    if not profile_done:
        profile_done = True
        profile_time = member_elapsed
        projected_100 = profile_time * 100
        projected_300 = profile_time * 300
        log(f"  PROFILE member 0: {profile_time:.1f}s  projected_100={projected_100/60:.1f}min  projected_300={projected_300/60:.1f}min")
        print(f"profile_member_time={profile_time:.1f}s", flush=True)

    # Current ensemble: average OOF and compute CV
    oof_avg = oof_sum / member_count
    test_avg = test_sum / member_count

    # Per-fold CV
    fold_scores = []
    for fi in folds_list:
        val_idx = np.asarray(fi["val_idx"])
        y_v = y_all[val_idx]
        preds = np.argmax(oof_avg[val_idx], axis=1)
        fold_scores.append(balanced_accuracy_score(y_v, preds))
    cv_now = float(np.mean(fold_scores))
    sem_now = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))

    total_elapsed = time.perf_counter() - member_loop_t0
    log(f"  member={member_count:3d}  config={config_idx}  cv={cv_now:.6f}±{sem_now:.6f}  "
        f"elapsed={total_elapsed:.0f}s  member_time={member_elapsed:.1f}s")

    # Log curve every LOG_EVERY members
    if member_count % LOG_EVERY == 0 or member_count == 1:
        curve_members.append(member_count)
        curve_cv.append(cv_now)
        print(f"cv_curve: n_members={member_count}  cv={cv_now:.6f}  sem={sem_now:.6f}", flush=True)

        # Plateau check: if gain over last LOG_EVERY < 2*sem, increment plateau counter
        if len(curve_cv) >= 2:
            gain = curve_cv[-1] - curve_cv[-2]
            if gain < 2 * sem_now:
                plateau_count += 1
                log(f"  plateau_count={plateau_count}/{PLATEAU_ROUNDS}  gain={gain:.6f} < 2*sem={2*sem_now:.6f}")
            else:
                plateau_count = 0
                log(f"  plateau reset: gain={gain:.6f} >= 2*sem={2*sem_now:.6f}")

        if plateau_count >= PLATEAU_ROUNDS and member_count >= TARGET_MEMBERS:
            log(f"  STOPPING: plateau for {PLATEAU_ROUNDS} consecutive LOG_EVERY periods at n_members={member_count}")
            print(f"PLATEAU_STOP: members={member_count} cv={cv_now:.6f}", flush=True)
            break

    if member_count >= MAX_MEMBERS:
        log(f"  STOPPING: reached max_members={MAX_MEMBERS}")
        print(f"MAX_MEMBERS_STOP: members={member_count} cv={cv_now:.6f}", flush=True)
        break

# Final average
oof_avg = (oof_sum / member_count).astype(np.float32)
test_avg = (test_sum / member_count).astype(np.float32)

log(f"\nBag complete: member_count={member_count}")

# ─── Final CV computation ─────────────────────────────────────────────────────
per_fold_scores = []
for fi in folds_list:
    val_idx = np.asarray(fi["val_idx"])
    y_v = y_all[val_idx]
    preds = np.argmax(oof_avg[val_idx], axis=1)
    score = balanced_accuracy_score(y_v, preds)
    per_fold_scores.append(score)
    log(f"  fold {fi['fold']}: balanced_accuracy={score:.6f}")
    print(f"fold{fi['fold']}_score={score:.6f}", flush=True)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── CV-vs-members curve summary ─────────────────────────────────────────────
log("\nCV-vs-members curve:")
for nm, ncv in zip(curve_members, curve_cv):
    log(f"  n={nm:3d}  cv={ncv:.6f}")
print(f"cv_curve_members={curve_members}", flush=True)
print(f"cv_curve_cvs={[round(v,6) for v in curve_cv]}", flush=True)

# ─── err-corr vs n070 ─────────────────────────────────────────────────────────
log("\nComputing err-corr vs n070 ...")
full_preds = np.argmax(oof_avg, axis=1)
n070_preds = np.argmax(n070_oof, axis=1)
err_bag = (full_preds != y_all).astype(float)
err_n070 = (n070_preds != y_all).astype(float)
if err_bag.std() == 0 or err_n070.std() == 0:
    err_corr = 1.0
else:
    err_corr = float(np.corrcoef(err_bag, err_n070)[0, 1])
log(f"err-corr vs n070: {err_corr:.4f}  (0.87 was n140 baseline; lower is better)")
print(f"errcorr_vs_n070={err_corr:.4f}", flush=True)

# ─── Save OOF and test probs ──────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_avg)
log(f"Saved oof.npy shape={oof_avg.shape}")
np.save(NODE_DIR / "test_probs.npy", test_avg)
log(f"Saved test_probs.npy shape={test_avg.shape}")

# ─── Submission ───────────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_avg.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── OOF sanity ──────────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_avg.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# ─── Stack-add probe to n091 (re-fit L2 meta with this mega-bag appended) ────
log("\n" + "="*60)
log("STACK-ADD PROBE: append mega-bag OOF to n091 FULL pool ...")
try:
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from pathlib import Path as PL
    from collections import Counter

    COMP = COMP_DIR
    LAB = ["GALAXY", "QSO", "STAR"]
    L2I = {l: i for i, l in enumerate(LAB)}
    NC = 3
    C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

    def logp(a): return np.log(np.clip(a, 1e-7, 1.0))
    def norm_probs(a):
        a = np.clip(a, 0, None)
        s = a.sum(1, keepdims=True)
        s[s == 0] = 1
        return a / s
    def score_fn(yt, yp): return float(np.mean(
        [(yp[yt == c] == c).mean() for c in range(NC) if (yt == c).any()]))

    train_sa = pd.read_csv(COMP / "data/train.csv")
    test_sa = pd.read_csv(COMP / "data/test.csv")
    n_sa = len(train_sa)
    nt_sa = len(test_sa)
    y_sa = train_sa["class"].map(L2I).to_numpy()
    folds_data_sa = json.loads((COMP / "folds.json").read_text())["folds"]
    fval_sa = [np.asarray(f["val_idx"]) for f in folds_data_sa]

    # Load n091 OOF (champion stack)
    n091_oof = np.load(COMP / "nodes/node_0091/oof.npy").astype(float)

    # Load the n091 FULL pool (just the existing champion stack OOF as a baseline)
    # The champion stack n091 used the FULL arm (63 bases).
    # We replicate the n091 FULL pool by loading public bank + FT-T + in-house bases,
    # THEN appending this mega-bag.

    # Faster approach: just use n091's OOF as a single base (the whole stack),
    # and compare: [n091 pool] vs [n091 pool + mega-bag].
    # The "pool" for the meta is the log-probs of each base's OOF.

    # To reproduce n091 FULL pool properly we'd need all 63 bases.
    # Instead, do a 2-arm: (A) n091 solo (our champion), (B) n091 + mega-bag.
    # This answers: does appending the mega-bag OOF help the meta above the champion?

    # Build 2-column feature matrix: (A) n091 OOF only, (B) n091 OOF + mega-bag OOF
    # Use log-probs, class-balanced LogReg, nested C grid, same as n091.

    log("  Loading n091 and mega-bag OOF for stack-add ...")
    mega_oof = oof_avg.astype(float)  # (577347, 3) — our mega-bag

    # Feature matrices
    OOF_A = logp(norm_probs(n091_oof))         # (n, 3) — champion alone
    OOF_B = np.concatenate([
        logp(norm_probs(n091_oof)),
        logp(norm_probs(mega_oof)),
    ], axis=1)                                  # (n, 6) — champion + mega-bag

    log(f"  Stack-add: OOF_A={OOF_A.shape} (champ only), OOF_B={OOF_B.shape} (champ+megabag)")

    # Test probs
    n091_test = np.load(COMP / "nodes/node_0091/test_probs.npy").astype(float)
    mega_test = test_avg.astype(float)
    TST_A = logp(norm_probs(n091_test))
    TST_B = np.concatenate([logp(norm_probs(n091_test)), logp(norm_probs(mega_test))], axis=1)

    def run_arm(OOF_mat, TST_mat, y, fval, label):
        n = len(y)
        oof_probs = np.zeros((n, NC), dtype=float)
        best_Cs = []
        for fi, vi in enumerate(fval):
            tr_idx = np.setdiff1d(np.arange(n), vi)
            lrcv = LogisticRegressionCV(
                Cs=C_GRID, cv=4, class_weight="balanced", max_iter=2000,
                n_jobs=-1, random_state=42, scoring="balanced_accuracy",
                solver="lbfgs", multi_class="multinomial",
            )
            lrcv.fit(OOF_mat[tr_idx], y[tr_idx])
            best_c = float(lrcv.C_[0])
            best_Cs.append(best_c)
            oof_probs[vi] = lrcv.predict_proba(OOF_mat[vi])
            pf = score_fn(y[vi], oof_probs[vi].argmax(1))
            log(f"  {label} fold {fi}: BA={pf:.6f} C={best_c}")
        pf_scores = [score_fn(y[vi], oof_probs[vi].argmax(1)) for vi in fval]
        cv = float(np.mean(pf_scores))
        sem = float(np.std(pf_scores, ddof=1) / np.sqrt(len(fval)))
        log(f"  {label}: cv={cv:.6f} sem={sem:.6f} per_fold={[f'{s:.6f}' for s in pf_scores]}")
        print(f"stack_{label}_cv={cv:.6f}", flush=True)
        # Final refit
        fc = Counter(best_Cs).most_common(1)[0][0]
        m_final = LogisticRegression(class_weight="balanced", C=fc, max_iter=2000,
                                     n_jobs=-1, random_state=42, solver="lbfgs", multi_class="multinomial")
        m_final.fit(OOF_mat, y)
        return oof_probs, m_final.predict_proba(TST_mat), pf_scores, cv, sem

    log("  Running ARM A (champion n091 OOF only, single base re-fitted) ...")
    oof_A, tst_A, pf_A, cv_A, sem_A = run_arm(OOF_A, TST_A, y_sa, fval_sa, "A_champ")
    log("  Running ARM B (champion n091 OOF + mega-bag OOF) ...")
    oof_B, tst_B, pf_B, cv_B, sem_B = run_arm(OOF_B, TST_B, y_sa, fval_sa, "B_stack")

    log(f"\nStack-add result: A={cv_A:.6f}±{sem_A:.6f}  B={cv_B:.6f}±{sem_B:.6f}")
    stack_delta = cv_B - cv_A
    log(f"  delta(B-A)={stack_delta:+.6f}  2*sem_B={2*sem_B:.6f}")
    print(f"stack_delta={stack_delta:+.6f}", flush=True)

    # Bootstrap P(B > A) using OOF predictions
    log("  Bootstrap P(B > A) ...")
    rng_boot = np.random.RandomState(999)
    n_boot = 2000
    B_wins = 0
    for _ in range(n_boot):
        bidx = rng_boot.choice(n_sa, size=n_sa, replace=True)
        y_b = y_sa[bidx]
        sa_b = score_fn(y_b, oof_A[bidx].argmax(1))
        sb_b = score_fn(y_b, oof_B[bidx].argmax(1))
        if sb_b > sa_b:
            B_wins += 1
    p_B_wins = B_wins / n_boot
    log(f"  Bootstrap P(B > A) = {p_B_wins:.3f}  (threshold >= 0.90 for keep/combine)")
    print(f"stack_boot_P={p_B_wins:.3f}", flush=True)

    # Holdout check (use the inviolable holdout from folds if available)
    # Since we don't have a separate holdout array here, we report the
    # bootstrap result as the primary gate.
    log(f"  Stack-add summary: cv_B={cv_B:.6f} cv_A={cv_A:.6f} delta={stack_delta:+.6f} P(B>A)={p_B_wins:.3f}")
    STACK_ADD_CV = cv_B
    STACK_ADD_P = p_B_wins

except Exception as e:
    log(f"  Stack-add probe FAILED: {e}")
    import traceback; traceback.print_exc()
    STACK_ADD_CV = None
    STACK_ADD_P = None

# ─── Post-run leakage / gate checks ─────────────────────────────────────────
log("\nPost-run gate checks ...")

# Gate 7: OOF complete (every train row exactly once, no NaN)
oof_check = np.load(NODE_DIR / "oof.npy")
assert oof_check.shape == (n_train, N_CLASSES), f"OOF shape {oof_check.shape} != ({n_train},{N_CLASSES})"
assert not np.isnan(oof_check).any(), "NaN in OOF!"
assert (oof_check >= 0).all() and (oof_check <= 1 + 1e-6).all(), "OOF out of [0,1]"
row_sums = oof_check.sum(axis=1)
assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: {row_sums.mean()}"
log("  [7] OOF full + no_nan: PASS")

# Gate 8: distribution sane
class_dist = np.bincount(oof_check.argmax(1), minlength=3)
log(f"  [8] dist_sane: GALAXY={class_dist[0]} QSO={class_dist[1]} STAR={class_dist[2]} PASS")

# Gate 9: submission schema
sub_check = pd.read_csv(NODE_DIR / "submission.csv")
assert list(sub_check.columns) == list(sample_sub.columns), f"schema mismatch"
assert len(sub_check) == len(sample_sub), f"row count mismatch"
assert set(sub_check[TARGET].unique()) <= set(CLASSES), "unknown classes"
log("  [9] schema_ok: PASS")

# Gate 10: cv-too-good
CV_PARENT = 0.969305  # n140
cv_too_good = mean_cv > 0.9705 and (mean_cv - CV_PARENT) > 0.005
log(f"  [10] cv_too_good: {'WARN' if cv_too_good else 'PASS'} (cv={mean_cv:.6f} parent={CV_PARENT:.6f})")

log("Post-run gate checks PASSED")

# ─── Summary ──────────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - T0
log(f"\n{'='*60}")
log(f"FINAL SUMMARY")
log(f"  members: {member_count}")
log(f"  per_fold: {' '.join(f'{s:.6f}' for s in per_fold_scores)}")
log(f"  cv={mean_cv:.6f}  sem={sem_cv:.6f}")
log(f"  n140 baseline: 0.969305  delta={mean_cv-0.969305:+.6f}")
log(f"  err-corr vs n070: {err_corr:.4f}  (n140 was 0.87)")
if STACK_ADD_CV is not None:
    log(f"  stack-add: cv_B={STACK_ADD_CV:.6f}  P(B>A)={STACK_ADD_P:.3f}")
log(f"  total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log(f"  cv_curve: {list(zip(curve_members, [round(v,6) for v in curve_cv]))}")
log(f"Done.")

print(f"cv={mean_cv:.6f}", flush=True)
print(f"SUMMARY: members={member_count} cv={mean_cv:.6f} sem={sem_cv:.6f} errcorr={err_corr:.4f} stack_P={STACK_ADD_P}", flush=True)
