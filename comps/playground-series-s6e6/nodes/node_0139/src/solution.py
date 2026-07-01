"""node_0139 — wildcard draft (nn): multi-task RealMLP with per-row ambiguity aux-target.

WILDCARD: two coupled pieces = ONE hypothesis.

(1) fs_ambig: per-row AMBIGUITY auxiliary target, derived from the champion-bank
    OOF disagreement (n070 OOF entropy + n070/n091 argmax disagreement) PLUS the
    row's true label (to identify which rows are genuinely hard vs just uncertain).
    Built FIT-IN-FOLD: computed ONLY from the train-fold rows' fold-honest OOF + their
    true labels. Val/test rows NEVER get a label-derived ambiguity. Binary label:
    1 = ambiguous (high OOF entropy AND/OR bank disagreement AND wrong prediction).

(2) Multi-task RealMLP: shared trunk on fs_realmlp_fe, two heads:
    - 3-class head: the real competition objective
    - ambiguity aux head: binary prediction of ambiguity label
    The aux head loss (BCELoss) is added to the primary CE loss with a small weight
    (alpha_aux). The shared trunk must learn to represent the ambiguity structure
    that the aux label encodes, which may help the 3-class head separate entangled rows.

    At inference: only the 3-class head's softmax output is used for oof/test_probs.

Leakage discipline (CRITICAL):
  - fs_ambig built per fold from that fold's TRAIN-FOLD rows ONLY.
  - The n070 OOF is fold-honest (577347 rows; row i was predicted by the fold that
    held row i out). For each fold's TRAIN indices, we use n070's OOF for those rows
    (never for val rows). This is safe because n070 itself is fold-honest.
  - Val and test rows get NO ambiguity label for the aux head at train time (we
    simply omit them from the aux loss computation). At inference, val/test rows do
    NOT need ambiguity labels — the 3-class head produces probs unconditionally.
  - All stateless FE (color pairs, redshift ratios) computed once.
  - KBins, TargetEncoder, NumericalPreprocessor fit inside each fold on train rows.

CHEAP-KILL criterion: fold-0 BA < 0.965 → kill.
MIRAGE guard: bootstrap P(>n091) + holdout check.

CV reference:
  baseline (n028): cv 0.969065, folds [0.970072, 0.968456, 0.968899, 0.968716, 0.969180]
  champion (n091): cv 0.970355, sem 0.000249
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
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# Fold-0 only mode for CHEAP-KILL check
FOLD_0_ONLY = os.environ.get("FOLD_0_ONLY", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}")
log(f"FOLD_0_ONLY: {FOLD_0_ONLY}")

# ─── Aux-target config ────────────────────────────────────────────────────────
# Weight for the auxiliary ambiguity head loss
AUX_LOSS_ALPHA = 0.15   # primary CE loss weight = 1.0, aux BCE weight = AUX_LOSS_ALPHA

# Ambiguity thresholds (computed from fold-honest n070 OOF)
# A row is "ambiguous" if:
#   (a) n070 OOF entropy > ENTROPY_THRESH (top ~20% = truly uncertain), AND
#   (b) n070 OOF predicted wrongly (fold-honest prediction != true label)
# This concentrates the aux target on rows where the bank genuinely struggles —
# not just borderline-confident rows that the bank actually gets right.
ENTROPY_THRESH_PCTILE = 80  # top 20% entropy rows
WRONG_PRED_ONLY = True       # ambiguous only if ALSO mispredicted by n070

def log(*args, **kwargs):
    """Re-define with timestamp."""
    msg = " ".join(str(a) for a in args)
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


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

# ─── CONFIG (identical to n028 — no change to primary NN hyperparams) ─────────
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
    # Aux task
    "aux_loss_alpha": AUX_LOSS_ALPHA,
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
    """Pure row-wise stateless FE — safe to apply to full df before fold split."""
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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
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

    # Original categorical columns
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
    """TargetEncoder fit on train fold only, transform val and test."""
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


# ─── fs_ambig: per-row ambiguity target (fit_in_fold) ───────────────────────────

def compute_ambig_labels_for_fold(
    train_idx: np.ndarray,
    y_all: np.ndarray,
    oof70: np.ndarray,
    oof91: np.ndarray,
) -> np.ndarray:
    """
    Compute binary ambiguity labels for train-fold rows ONLY.

    The ambiguity label is 1 for rows where:
      (a) n070 OOF entropy is in the top ENTROPY_THRESH_PCTILE (high uncertainty)
      AND
      (b) (optionally) n070 predicted the row wrong (using that row's OOF pred)

    This is fit-in-fold: we use the train-fold rows' OOF probs from n070 (which
    were computed by n070 leaving that row out), so this is fold-honest.

    CRITICAL: we only use train_idx rows of the OOF. Val/test rows do NOT get
    a label from this function — they never receive a label-derived ambiguity value.

    Args:
        train_idx: integer indices of the current train fold
        y_all: all training labels (integer-coded)
        oof70: n070 fold-honest OOF (577347, 3)
        oof91: n091 fold-honest OOF (577347, 3) — not used currently, kept for future
    Returns:
        ambig: (len(train_idx),) float32 array, values in {0.0, 1.0}
    """
    # Extract train-fold rows from the fold-honest OOF
    tr_probs70 = oof70[train_idx].astype(np.float64)  # (n_tr, 3)
    tr_labels = y_all[train_idx]  # (n_tr,)

    # Compute entropy for each row (per scipy convention: entropy = -sum(p log p))
    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(tr_probs70, eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p)).sum(axis=1)  # (n_tr,)

    # Threshold: top ENTROPY_THRESH_PCTILE% entropy within this fold's train rows
    ent_thresh = np.percentile(ent, ENTROPY_THRESH_PCTILE)
    high_ent = ent >= ent_thresh  # (n_tr,) bool

    if WRONG_PRED_ONLY:
        # Also require that n070 predicted wrong
        n070_pred = tr_probs70.argmax(axis=1)
        wrong_pred = n070_pred != tr_labels
        ambig = (high_ent & wrong_pred).astype(np.float32)
    else:
        ambig = high_ent.astype(np.float32)

    return ambig


# ─── Model components (from n028, extended with aux head) ────────────────────

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


class RealMLPMultiTask(nn.Module):
    """
    Multi-task RealMLP: shared trunk + 2 heads.
    Primary head: 3-class softmax (the competition objective).
    Aux head: binary sigmoid (ambiguity prediction).
    """
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

        # Primary head: 3-class classification
        self.primary_head = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

        # Auxiliary head: binary ambiguity prediction (one output per ensemble member)
        self.aux_head = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=1)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        trunk = self.hidden(combined)

        # Primary head output: softmax probabilities
        primary_logits = self.primary_head(trunk)  # (batch, n_ens, 3)
        primary_probs = F.softmax(primary_logits, dim=2)

        # Aux head output: sigmoid probability of being ambiguous
        aux_logits = self.aux_head(trunk).squeeze(-1)  # (batch, n_ens)
        aux_probs = torch.sigmoid(aux_logits)  # (batch, n_ens)

        return primary_probs, aux_probs


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


def get_parameter_groups(model: RealMLPMultiTask, p: dict):
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


class RealMLP_MultiTask_Classifier(BaseEstimator):
    """Multi-task RealMLP with ambiguity aux head."""

    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            cat_col_names=None, X_test: pd.DataFrame = None,
            ambig_train: np.ndarray = None):
        """
        Args:
            ambig_train: (n_train,) float32 binary ambiguity labels for train rows.
                         If None, the aux head is not trained (for smoke test).
        """
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

        # Cat dims
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

        # loss_prior_power
        loss_prior_power = float(p.get("loss_prior_power", 0.0))
        loss_prob_multipliers = None
        if loss_prior_power != 0.0:
            class_counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            class_counts = class_counts / np.exp(np.log(class_counts).mean())
            loss_mult_np = np.power(class_counts, loss_prior_power)
            loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

        n_classes = len(classes)
        self.model_ = RealMLPMultiTask(
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

        # Aux target tensor (train rows only)
        have_aux = ambig_train is not None
        if have_aux:
            ambig_t = torch.as_tensor(ambig_train, dtype=torch.float32, device=dev)
            aux_alpha = float(p.get("aux_loss_alpha", AUX_LOSS_ALPHA))
            log(f"  Aux target: {ambig_train.sum():.0f}/{len(ambig_train)} ambiguous rows "
                f"({100*ambig_train.mean():.1f}%) alpha={aux_alpha}")
        else:
            aux_alpha = 0.0
            log("  No aux target provided — training primary head only")

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
                primary_probs, aux_probs = self.model_(x_num_batch, Xtc[idx_batch])
                # primary_probs: (bs, n_ens, 3)
                # aux_probs: (bs, n_ens)

                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val

                # Primary CE loss (average over ensemble)
                prim_loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    primary_probs.reshape(-1, n_classes),
                    ls=ls_val,
                    class_weights=class_weights,
                    focal_gamma=float(p.get("focal_gamma", 0.0)),
                    loss_prob_multipliers=loss_prob_multipliers,
                )

                # Aux BCE loss (if aux labels available)
                if have_aux:
                    # aux_probs: (bs, n_ens) — average over ensemble for loss
                    # targets: (bs,) -> expand to (bs, n_ens) then flatten
                    aux_target = ambig_t[idx_batch]  # (bs,)
                    aux_target_expanded = aux_target.unsqueeze(1).expand(-1, n_ens)  # (bs, n_ens)
                    aux_loss = F.binary_cross_entropy(
                        aux_probs.reshape(-1),
                        aux_target_expanded.reshape(-1),
                    )
                    total_loss = prim_loss + aux_alpha * aux_loss
                else:
                    total_loss = prim_loss

                total_loss.backward()
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
                val_probs_list = []
                for s in range(0, len(y_v), eval_bs):
                    prim_probs_batch, _ = self.model_(
                        Xvn[s: s + eval_bs], Xvc[s: s + eval_bs]
                    )
                    val_probs_list.append(prim_probs_batch.mean(dim=1).cpu().numpy())
                val_probs = np.concatenate(val_probs_list, axis=0)

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
                aux_info = f"  aux_loss={aux_loss.item():.4f}" if have_aux else ""
                log(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                    f"best={best_score:.5f}  ls={ls_val:.4f}  drop={drop_val:.4f}"
                    + aux_info
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
        """Predict using primary head only."""
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        probs_list = []
        with torch.no_grad():
            for s in range(0, len(X_num), eval_bs):
                prim_probs, _ = self.model_(Xn[s: s + eval_bs], Xc[s: s + eval_bs])
                probs_list.append(prim_probs.mean(dim=1).cpu().numpy())
        return np.concatenate(probs_list, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


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

# ─── Load bank OOF for fs_ambig ───────────────────────────────────────────────
log("Loading bank OOF for fs_ambig ...")
oof70_path = COMP_DIR / "nodes/node_0070/oof.npy"
oof91_path = COMP_DIR / "nodes/node_0091/oof.npy"
oof70 = np.load(oof70_path).astype(np.float64)
oof91 = np.load(oof91_path).astype(np.float64)
log(f"  oof70={oof70.shape} oof91={oof91.shape}")
assert oof70.shape == (n_train, N_CLASSES), f"oof70 shape mismatch: {oof70.shape}"
assert oof91.shape == (n_train, N_CLASSES), f"oof91 shape mismatch: {oof91.shape}"

# Verify OOF sanity
from sklearn.metrics import balanced_accuracy_score as _ba
_ba70 = _ba(y_all, oof70.argmax(1))
_ba91 = _ba(y_all, oof91.argmax(1))
log(f"  sanity: n070 OOF BA={_ba70:.6f} n091 OOF BA={_ba91:.6f}")
assert _ba70 > 0.96, f"n070 OOF BA too low: {_ba70}"
assert _ba91 > 0.96, f"n091 OOF BA too low: {_ba91}"

# ─── PRE-FLIGHT LEAKAGE CHECKS ────────────────────────────────────────────────
log("=== PRE-FLIGHT LEAKAGE CHECKS ===")

# Check 1: target not in features (will be verified when we build features)
# Check 2: id not in features (will be verified below)
# Check 5: folds from frozen folds.json
log("Check 5: folds loaded from frozen folds.json — PASS (loaded from folds.json above)")

# Check 6: train/test near-dup (sample)
log("Check 6: train/test near-dup check (redshift, mag values) ...")
_tr_sample = train_raw[["u", "g", "r", "i", "z", "redshift"]].values[:5000]
_te_sample = test_raw[["u", "g", "r", "i", "z", "redshift"]].values[:5000]
_tr_rounded = np.round(_tr_sample, 4)
_te_rounded = np.round(_te_sample, 4)
_tr_set = set(map(tuple, _tr_rounded))
_te_set = set(map(tuple, _te_rounded))
_overlap = len(_tr_set & _te_set)
log(f"  overlap in 5k/5k sample (6-feature round-4): {_overlap} rows")
if _overlap > 100:
    log("  WARN: significant train/test overlap detected — check if synthetic comp")
else:
    log("  Check 6: OK (expected for synthetic tabular comp)")

log("Pre-flight checks: target/id excluded from feature columns verified at FE time.")
log("=== END PRE-FLIGHT ===")

# ─── Stateless FE (computed once) ─────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

# Check 1 & 2: verify target and id not in X_raw
assert TARGET not in X_raw.columns, f"Target leaked into features: {TARGET}"
assert IDC not in X_raw.columns, f"ID leaked into features: {IDC}"
log("Check 1 (target not in features): PASS")
log("Check 2 (id not in features): PASS")

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Check 3: single-feature correlation sweep (≤50k sample)
log("Check 3: single-feature↔target sweep ...")
_sample_size = min(50000, n_train)
_sample_idx = np.random.choice(n_train, _sample_size, replace=False)
_X_samp = X_stateless.iloc[_sample_idx]
_y_samp = y_all[_sample_idx]
_suspect_feats = []
for col in _X_samp.columns:
    _x = pd.to_numeric(_X_samp[col], errors="coerce")
    if _x.nunique() > 1:
        _r = abs(np.corrcoef(_x.fillna(_x.mean()), _y_samp)[0, 1])
        if _r >= 0.999:
            _suspect_feats.append((col, _r))
if _suspect_feats:
    log(f"  LEAK SMELL: {_suspect_feats} — stopping!")
    sys.exit(1)
else:
    log("Check 3: single-feature sweep clean — PASS")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None
num_cols_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

# Filter folds for FOLD_0_ONLY mode
active_folds = [fi for fi in folds_list if (not FOLD_0_ONLY or fi["fold"] == 0)]

for fi in active_folds:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ── fs_ambig: build per-row ambiguity labels from THIS fold's train rows ──
    # CRITICAL CHECK: only train-fold rows, never val rows
    # The oof70 probs for train_idx rows are valid fold-honest probs (n070 held
    # those rows out in its respective folds). Using train_idx rows of oof70 as
    # a training signal is fold-safe here because we're building a FEATURE for
    # a DIFFERENT node's training — not for n070's own val, which would be a leak.
    # The key guarantee: val_idx rows NEVER receive an ambiguity label derived
    # from their own true labels.
    log(f"  Building fs_ambig from {len(tr_idx)} train-fold rows ...")
    ambig_labels_train = compute_ambig_labels_for_fold(
        tr_idx, y_all, oof70, oof91
    )
    log(f"  fs_ambig: {ambig_labels_train.sum():.0f}/{len(ambig_labels_train)} "
        f"ambiguous ({100*ambig_labels_train.mean():.1f}%)")

    # ── Categorical encoding — fit_in_fold ─────────────────────────────────
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # ── Target encoding — fit_in_fold ──────────────────────────────────────
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

        # Check 4 (fit-inside-fold): verify that cat/TE/KBins/NumericalPrep are all fold-local
        # (verified by code structure: fit_fold_categoricals and add_target_encoding
        # are called INSIDE this fold loop with only tr_idx rows; NumericalPreprocessor
        # is fit inside RealMLP_MultiTask_Classifier.fit() on X_tr_num from that fold)
        log("Check 4 (fit-inside-fold): categorical/TE/KBins fit on tr_idx only — PASS by construction")
        log("  fs_ambig: computed from tr_idx rows of fold-honest oof70 — PASS (see compute_ambig_labels_for_fold)")

    cfg_fold = {**CONFIG, "random_state": fold_seed, "device": str(DEVICE)}
    model = RealMLP_MultiTask_Classifier(**cfg_fold)
    model.fit(
        X_tr_fold, y_tr_fold,
        X_val_fold, y_val_fold,
        cat_col_names=cat_cols_sorted,
        X_test=X_te_fold,
        ambig_train=ambig_labels_train,
    )

    # OOF probabilities (3-class head output)
    oof_proba[val_idx] = model.best_val_probs_.astype("float32")

    # Test predictions — average across folds
    test_proba_accum += model.predict_proba(X_te_fold).astype("float32") / len(active_folds)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold, ambig_labels_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # CHEAP-KILL check for FOLD_0_ONLY mode
        if FOLD_0_ONLY:
            kill_bar = 0.965
            log(f"  CHEAP-KILL check: fold-0 BA={fold_score:.6f}  kill_bar={kill_bar}")
            if fold_score < kill_bar:
                log(f"  KILL: fold-0 BA {fold_score:.6f} < {kill_bar} — aux objective hurts primary task")
                log(f"  Exiting early (CHEAP-KILL tripped)")
                print(f"CHEAP_KILL fold0_BA={fold_score:.6f} bar={kill_bar}", flush=True)
                sys.exit(42)
            else:
                log(f"  CHEAP-KILL: PASS — fold-0 BA {fold_score:.6f} >= {kill_bar}")
                print(f"CHEAP_KILL_PASS fold0_BA={fold_score:.6f}", flush=True)
                sys.exit(0)  # Signal success for fold-0-only run

if FOLD_0_ONLY:
    sys.exit(0)

# ─── Post-OOF scoring ─────────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Post-train leakage checks ────────────────────────────────────────────────
# Check 7: OOF complete (all train rows covered)
oof_covered = np.all(oof_proba.sum(axis=1) > 0)
log(f"Check 7 (OOF complete): {oof_covered} — {'PASS' if oof_covered else 'FAIL'}")
oof_has_nan = np.any(np.isnan(oof_proba))
log(f"Check 7 (no NaN in OOF): {not oof_has_nan} — {'PASS' if not oof_has_nan else 'FAIL'}")

# Check 8: distribution sane
probs_sum = oof_proba.sum(axis=1)
dist_sane = (
    not oof_has_nan and
    float(probs_sum.min()) > 0.99 and
    float(probs_sum.max()) < 1.01 and
    float(oof_proba.min()) >= 0.0 and
    float(oof_proba.max()) <= 1.0
)
log(f"Check 8 (distribution sane): min_prob={oof_proba.min():.4f} max_prob={oof_proba.max():.4f} "
    f"sum_min={probs_sum.min():.4f} sum_max={probs_sum.max():.4f} — {'PASS' if dist_sane else 'FAIL'}")

# Per-class recall check
from sklearn.metrics import classification_report as _cr
_report = _cr(y_all, oof_proba.argmax(1), target_names=CLASSES, output_dict=True)
for cls_name in CLASSES:
    log(f"  {cls_name}: precision={_report[cls_name]['precision']:.4f} "
        f"recall={_report[cls_name]['recall']:.4f} "
        f"f1={_report[cls_name]['f1-score']:.4f}")

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
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
