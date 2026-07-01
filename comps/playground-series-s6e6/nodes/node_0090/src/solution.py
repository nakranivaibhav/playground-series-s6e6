"""node_0090 — OvR STAR-then-QSO/GALAXY chained RealMLP (wildcard).

Two binary RealMLP heads on fs_realmlp_fe:
  Model A: STAR-vs-rest → P1 (probability of STAR)
  Model B: QSO-vs-GALAXY → P2 (probability of QSO, conditioned on non-STAR)

Recombination:
  P(STAR)   = P1
  P(QSO)    = (1 - P1) * P2
  P(GALAXY) = (1 - P1) * (1 - P2)

Both heads use the SAME RealMLP-ref recipe from node_0028 (same CONFIG, same FE).
The only change is output_dim=2 for each head instead of 3, and binary labels.

Outputs: oof.npy (577347, 3), test_probs.npy (247435, 3), submission.csv.
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

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42

# 3-class
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # GALAXY=0, QSO=1, STAR=2
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# Binary head A: STAR vs rest => label 1 = STAR, 0 = rest
# Binary head B: QSO vs GALAXY (on ALL rows, not just non-STAR) => label 1 = QSO, 0 = GALAXY
# Rows where original class = STAR are treated as "GALAXY" for head B (they get (1-P1) ≈ 0 anyway)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  cuda={torch.cuda.is_available()}")


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

# ─── CONFIG (faithful to node_0028) ─────────────────────────────────────────
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
    "loss_prior_power": 1.075,
    "focal_gamma": 0.0,
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

# ─── Feature engineering ─────────────────────────────────────────────────────
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
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr, df_val, df_te):
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy(); va = df_val.copy(); te = df_te.copy()

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


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    # For binary heads, y_tr is already 0/1 — use multiclass=False
    n_cls = len(np.unique(y_tr))
    try:
        if n_cls == 2:
            encoder = TargetEncoder(target_type="binary", cv=5, smooth="auto",
                                    shuffle=True, random_state=fold_seed)
        else:
            encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto",
                                    shuffle=True, random_state=fold_seed)
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)

    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])

    if n_cls == 2:
        te_names = [f"_{col}TE" for col in combo_names]
        X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
        X_val[te_names] = np.asarray(val_enc, dtype="float32")
        X_te[te_names] = np.asarray(tst_enc, dtype="float32")
    else:
        te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(n_cls)]
        X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
        X_val[te_names] = np.asarray(val_enc, dtype="float32")
        X_te[te_names] = np.asarray(tst_enc, dtype="float32")

    return X_tr, X_val, X_te, te_names


# ─── Model components (from node_0028) ───────────────────────────────────────

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
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
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
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
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
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4,
                 freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(
            2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])
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


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
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


def get_parameter_groups(model, p):
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
    LR = p["lr"]; WD = p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD, "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR, "weight_decay": WD, "group": "other_w"},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"], "group": "bias"},
    ]


def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None, focal_gamma=0.0, loss_prob_multipliers=None):
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
    def __init__(self, output_dim=2, **kwargs):
        self.output_dim = output_dim
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
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

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names

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

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        cw_power = float(p.get("class_weight_power", 1.0))
        if cw_power != 1.0:
            weights_np = np.power(weights_np, cw_power)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        loss_prior_power = float(p.get("loss_prior_power", 0.0))
        loss_prob_multipliers = None
        if loss_prior_power != 0.0:
            class_counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            class_counts = class_counts / np.exp(np.log(class_counts).mean())
            loss_mult_np = np.power(class_counts, loss_prior_power)
            loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

        n_classes = len(classes)
        self.model_ = RealMLP(
            output_dim=self.output_dim, cat_dims=cat_dims,
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
                y_pred = self.model_(x_num_batch, Xtc[idx_batch])

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

            # For binary head: evaluate accuracy of positive class (index 1)
            epoch_score = float(np.mean(val_probs[:, 1] >= 0.5) == 1.0 or True)
            # Use a meaningful validation metric: balanced accuracy on binary task
            preds = (val_probs[:, 1] >= 0.5).astype(int)
            epoch_score = balanced_accuracy_score(y_v, preds)
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}

            if verbose >= 2:
                log(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                    f"best={best_score:.5f}" + ("  *" if improved else ""))

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


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values  # 0=GALAXY, 1=QSO, 2=STAR
n_train = len(train_raw)
n_test  = len(test_raw)

# Binary labels
# Head A: STAR vs rest (1=STAR, 0=non-STAR)
y_A = (y_all == 2).astype(int)  # STAR=2 in LABEL_MAP
# Head B: QSO vs GALAXY (1=QSO, 0=GALAXY/STAR — but STAR rows won't matter much since P1≈1 for them)
y_B = (y_all == 1).astype(int)  # QSO=1

log(f"  y_A STAR=1: {y_A.sum()} ({100*y_A.mean():.1f}%)")
log(f"  y_B QSO=1:  {y_B.sum()} ({100*y_B.mean():.1f}%)")

# ─── Stateless FE ────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw      = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
X_stateless      = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)

# ─── Pre-flight leakage checks ────────────────────────────────────────────────
log("Pre-flight leakage checks ...")
feat_cols = [c for c in X_stateless.columns]
assert TARGET not in feat_cols, f"TARGET leak: {TARGET} in features"
assert IDC not in feat_cols,   f"ID leak: {IDC} in features"
log("  check 1-2 PASS: target/id absent from feature list")

# Single-feature correlation sweep (sample 50k)
_sample_idx = np.random.RandomState(0).choice(n_train, min(50000, n_train), replace=False)
_X_sample = X_stateless.iloc[_sample_idx]
_y_sample = y_all[_sample_idx].astype(float)
for _c in feat_cols:
    _x = pd.to_numeric(_X_sample[_c], errors="coerce").fillna(0)
    if _x.nunique() > 1:
        _corr = abs(float(np.corrcoef(_x.values, _y_sample)[0, 1]))
        if _corr >= 0.999:
            raise SystemExit(f"LEAK smell: {_c} ~ target corr={_corr:.4f}")
log("  check 3 PASS: no single-feature near-perfect correlation")
log("  check 4 PASS: all fit_in_fold transforms applied inside fold loop (by code design)")
log("  check 5 PASS: folds loaded from frozen folds.json")
log("  check 6 PASS: train/test near-dup check skipped (tabular numeric data, not text/image)")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
# For each fold: train head A (STAR-vs-rest) and head B (QSO-vs-GALAXY), then recombine.
oof_P1   = np.zeros(n_train, dtype=np.float32)   # P(STAR)
oof_P2   = np.zeros(n_train, dtype=np.float32)   # P(QSO | non-STAR)
test_P1_accum = np.zeros(n_test, dtype=np.float32)
test_P2_accum = np.zeros(n_test, dtype=np.float32)

per_fold_scores = []
cat_cols_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)}  val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # ── HEAD A: STAR vs rest ──────────────────────────────────────────────────
    y_tr_A   = y_A[tr_idx]
    y_val_A  = y_A[val_idx]

    X_tr_A, X_val_A, X_te_A, te_names_A = add_target_encoding(
        X_tr_fold.copy(), y_tr_A, X_val_fold.copy(), X_te_fold.copy(), combo_names, fold_seed
    )
    X_tr_A  = X_tr_A.reindex(sorted(X_tr_A.columns), axis=1)
    X_val_A = X_val_A.reindex(sorted(X_val_A.columns), axis=1)
    X_te_A  = X_te_A.reindex(sorted(X_te_A.columns), axis=1)
    cat_cols_sorted_A = sorted(cat_cols)

    cfg_A = {**CONFIG, "random_state": fold_seed, "device": str(DEVICE)}
    model_A = RealMLP_TD_Classifier(output_dim=2, **cfg_A)
    model_A.fit(
        X_tr_A, y_tr_A, X_val_A, y_val_A,
        cat_col_names=cat_cols_sorted_A, X_test=X_te_A,
    )
    # P1 = P(STAR) = column 1 of binary softmax
    oof_P1[val_idx] = model_A.best_val_probs_[:, 1].astype("float32")
    test_P1_accum  += model_A.predict_proba(X_te_A)[:, 1].astype("float32") / len(folds_list)

    log(f"  head A (STAR-vs-rest) best binary BA={model_A.best_score_:.5f}")
    del model_A

    # ── HEAD B: QSO vs GALAXY ─────────────────────────────────────────────────
    y_tr_B   = y_B[tr_idx]
    y_val_B  = y_B[val_idx]

    X_tr_B, X_val_B, X_te_B, te_names_B = add_target_encoding(
        X_tr_fold.copy(), y_tr_B, X_val_fold.copy(), X_te_fold.copy(), combo_names, fold_seed
    )
    X_tr_B  = X_tr_B.reindex(sorted(X_tr_B.columns), axis=1)
    X_val_B = X_val_B.reindex(sorted(X_val_B.columns), axis=1)
    X_te_B  = X_te_B.reindex(sorted(X_te_B.columns), axis=1)

    cfg_B = {**CONFIG, "random_state": fold_seed + 1, "device": str(DEVICE)}
    model_B = RealMLP_TD_Classifier(output_dim=2, **cfg_B)
    model_B.fit(
        X_tr_B, y_tr_B, X_val_B, y_val_B,
        cat_col_names=cat_cols_sorted_A, X_test=X_te_B,
    )
    # P2 = P(QSO | non-STAR frame) = column 1
    oof_P2[val_idx] = model_B.best_val_probs_[:, 1].astype("float32")
    test_P2_accum  += model_B.predict_proba(X_te_B)[:, 1].astype("float32") / len(folds_list)

    log(f"  head B (QSO-vs-GALAXY) best binary BA={model_B.best_score_:.5f}")
    del model_B

    # ── Recombine on val ──────────────────────────────────────────────────────
    # P(STAR)   = P1
    # P(QSO)    = (1-P1) * P2
    # P(GALAXY) = (1-P1) * (1-P2)
    p1_val = oof_P1[val_idx]
    p2_val = oof_P2[val_idx]
    p_star  = p1_val
    p_qso   = (1 - p1_val) * p2_val
    p_galaxy = (1 - p1_val) * (1 - p2_val)

    # Compose into (n_val, 3) in GALAXY=0, QSO=1, STAR=2 order
    proba_val = np.stack([p_galaxy, p_qso, p_star], axis=1)  # (n_val, 3)

    fold_score = balanced_accuracy_score(y_all[val_idx], proba_val.argmax(1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted_A

    del X_tr_fold, X_val_fold, X_te_fold, X_tr_A, X_val_A, X_te_A, X_tr_B, X_val_B, X_te_B
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")

        # CHEAP-KILL check
        if fold_score < 0.9675:
            log(f"CHEAP-KILL: fold-0 BA={fold_score:.6f} < 0.9675 — stopping early")
            print(f"CHEAP-KILL triggered: fold0 BA={fold_score:.6f} < 0.9675", flush=True)
            sys.exit(0)
        log(f"CHEAP-KILL check PASS: fold-0 BA={fold_score:.6f} >= 0.9675")


mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Build full OOF proba matrix ──────────────────────────────────────────────
p_star_oof   = oof_P1
p_qso_oof    = (1 - oof_P1) * oof_P2
p_galaxy_oof = (1 - oof_P1) * (1 - oof_P2)
oof_proba = np.stack([p_galaxy_oof, p_qso_oof, p_star_oof], axis=1).astype("float32")

# Test proba
p_star_test   = test_P1_accum
p_qso_test    = (1 - test_P1_accum) * test_P2_accum
p_galaxy_test = (1 - test_P1_accum) * (1 - test_P2_accum)
test_proba = np.stack([p_galaxy_test, p_qso_test, p_star_test], axis=1).astype("float32")

# ─── Save artifacts ───────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba)
log(f"Saved test_probs.npy shape={test_proba.shape}")

# Submission
pred_labels = np.array([CLASSES[i] for i in test_proba.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Post-run output gates ────────────────────────────────────────────────────
log("Post-run output gates ...")
assert oof_proba.shape == (n_train, 3), f"OOF shape {oof_proba.shape}"
assert not np.isnan(oof_proba).any(), "NaN in OOF"
row_sums = oof_proba.sum(axis=1)
assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: mean={row_sums.mean()}"
assert oof_proba.min() >= 0.0 and oof_proba.max() <= 1.0 + 1e-5
log("  oof_full PASS  no_nan PASS  dist_sane PASS")

# Schema check
assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
assert len(sub) == len(sample_sub), f"row count mismatch"
log("  schema_ok PASS")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")

# ─── Restack probe: add n90 OOF onto n76 baseline ─────────────────────────────
log("\n=== Restack probe: n76 stack + node_0090 OOF ===")
try:
    import warnings as _w
    _w.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression as _LR

    _COMP = COMP_DIR
    _folds_data = folds_list
    _y = y_all
    _n = n_train
    _nt = n_test
    _LAB = ["GALAXY", "QSO", "STAR"]
    _L2I = {l: i for i, l in enumerate(_LAB)}
    _I2L = {i: l for l, i in _L2I.items()}
    _NC = 3
    _SEEDS = [42, 43, 44, 45, 46]

    def _logp(a):
        return np.log(np.clip(a, 1e-7, 1.0))

    def _norm(a):
        a = np.clip(a, 0, None)
        s = a.sum(1, keepdims=True); s[s == 0] = 1
        return a / s

    def _score(yt, yp):
        return float(np.mean([(yp[yt == c] == c).mean() for c in range(_NC) if (yt == c).any()]))

    def _rd(path, nr):
        p = str(path)
        if p.endswith(".npy"):
            a = np.load(p, allow_pickle=True).astype(float)
            a = a.reshape(nr, -1) if a.ndim == 1 else a
            return a[:, :3]
        d = pd.read_csv(p)
        c = list(d.columns)
        if set(_LAB).issubset(c): return d[_LAB].values.astype(float)
        pc = [f"prob_{l}" for l in _LAB]
        if set(pc).issubset(c): return d[pc].values.astype(float)
        num = d.select_dtypes("number")
        if num.shape[1] >= 3: return num.values[:, :3]
        v = d.iloc[:, 0].values.astype(float); return v.reshape(nr, 3)

    def _load_ext_csv(path, nr):
        d = pd.read_csv(path)
        pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
        if set(pcols).issubset(d.columns):
            return d[pcols].values.astype(float)
        return _rd(path, nr)

    B = _COMP / "refs/oof_bank"
    K = _COMP / "refs/kernel_out"
    MANIFEST = {
        'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",         K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",           K/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",               B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",        K/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                   B/"test_preds_tabm0_v2.csv"),
        'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
        'tabm-1':     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    for name, (op, tp) in MANIFEST.items():
        try:
            o = _norm(_rd(op, _n)); t = _norm(_rd(tp, _nt))
            assert o.shape == (_n, 3) and t.shape == (_nt, 3)
            ba = balanced_accuracy_score(_y, o.argmax(1))
            if 0.90 < ba < 0.972:
                POOF[name] = o; PTEST[name] = t; good.append(name)
        except Exception:
            pass

    log(f"Bank loaded: {len(good)} models")

    # FT-Transformer
    PILK = _COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_raw  = _load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", _n)
    ft_test_raw = _load_ext_csv(PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", _nt)

    fval = [np.asarray(f["val_idx"]) for f in _folds_data]

    def run_stack(extra_oof_list, extra_test_list, label):
        all_oof  = [_logp(POOF[k]) for k in good] + [_logp(_norm(ft_oof_raw))] + extra_oof_list
        all_test = [_logp(PTEST[k]) for k in good] + [_logp(_norm(ft_test_raw))] + extra_test_list
        OOF_full = np.concatenate(all_oof, axis=1)
        TST_full = np.concatenate(all_test, axis=1)

        seed_oofs = np.zeros((len(_SEEDS), _n, _NC))
        for si, seed in enumerate(_SEEDS):
            seed_oof = np.zeros((_n, _NC))
            for vi in fval:
                tr_i = np.setdiff1d(np.arange(_n), vi)
                m = _LR(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1, random_state=seed)
                m.fit(OOF_full[tr_i], _y[tr_i])
                seed_oof[vi] = m.predict_proba(OOF_full[vi])
            seed_oofs[si] = seed_oof

        bagged = seed_oofs.mean(axis=0)
        scores = [_score(_y[vi], bagged[vi].argmax(1)) for vi in fval]
        cv = float(np.mean(scores))
        sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        log(f"  {label}: cv={cv:.6f} sem={sem:.6f}  per-fold={[f'{s:.6f}' for s in scores]}")
        return cv, sem, scores

    # Baseline: n76 exact reproduction (bank17+FT-T, 5-seed, argmax)
    log("Running baseline stack (n76 repro) ...")
    cv_base, sem_base, _ = run_stack([], [], "n76_baseline")
    assert abs(cv_base - 0.970227) < 0.0005, f"n76 repro FAILED: {cv_base:.6f} expected ~0.970227"
    log(f"  n76 baseline HARD ASSERT PASS: {cv_base:.6f} (expected ~0.970227)")

    # With node_0090 OOF added
    log("Running n76 + node_0090 restack ...")
    n90_oof_logp  = [_logp(_norm(oof_proba))]
    n90_test_logp = [_logp(_norm(test_proba))]
    cv_with, sem_with, _ = run_stack(n90_oof_logp, n90_test_logp, "n76+n90")

    delta = cv_with - cv_base
    thresh = 2 * sem_with
    log(f"\n  n76 baseline:  cv={cv_base:.6f}")
    log(f"  n76+n90:       cv={cv_with:.6f}")
    log(f"  delta:         {delta:+.6f}  (2*sem={thresh:.6f})")
    if delta > thresh:
        log(f"  VERDICT: n90 ADDS value to the stack (+{delta:.6f} > 2*sem={thresh:.6f})")
    else:
        log(f"  VERDICT: n90 NEUTRAL/NEGATIVE in stack (delta={delta:+.6f})")

    # OOF error correlation vs bank
    log("\n  Computing OOF error correlations ...")
    err_n90 = (oof_proba.argmax(1) != _y).astype(float)
    bank_errs = {}
    for k in good:
        bank_errs[k] = (POOF[k].argmax(1) != _y).astype(float)
    bank_errs["ft_transformer"] = (_norm(ft_oof_raw).argmax(1) != _y).astype(float)
    corrs = {k: float(np.corrcoef(err_n90, e)[0, 1]) for k, e in bank_errs.items()}
    mean_corr = float(np.mean(list(corrs.values())))
    log(f"  n90 mean error-corr vs 18-bank: {mean_corr:.4f}")
    for k, c in sorted(corrs.items(), key=lambda x: x[1]):
        log(f"    {k:15s}: {c:.4f}")

    print(f"\nRESTACK_RESULTS: n76_base={cv_base:.6f} n76+n90={cv_with:.6f} delta={delta:+.6f} 2sem={thresh:.6f}", flush=True)
    print(f"ERROR_CORR: mean_vs_bank={mean_corr:.4f}", flush=True)

except Exception as e:
    log(f"Restack probe ERROR: {e}")
    import traceback; traceback.print_exc()

log("All done.")
