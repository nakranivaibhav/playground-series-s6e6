"""node_0110 — TabM multi-task with redshift auxiliary head.

WILDCARD (coupled change):
  1. DROP raw redshift AND its direct derivatives (log1p_redshift, _g_div_redshift,
     _i_div_redshift, redshift_cat_) from the INPUT feature set.
  2. Add a second linear head on the TabM trunk outputting a scalar redshift estimate.
     Total loss = balanced-CE(class) + lambda * Huber(redshift_true_std, redshift_pred).
     Redshift standardized using TRAIN-FOLD-ONLY mean/std (fit_in_fold).
  3. At inference, ONLY the class head is used.

Gate order:
  - Fold-0 only first: sweep lambda in {0.1, 0.5}; record solo BA + err-corr vs node_0070.
  - Stop if best-lambda fold-0 BA < 0.962 OR fold-0 err-corr >= 0.70.
  - Else run all 5 folds at best lambda.

FE pipeline: byte-identical to node_0033 EXCEPT redshift and its direct derivatives
are excluded from the input features (they are still available internally for the
aux target, but not passed to the model as inputs).
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
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  tabm={tabm.__version__}")

SMOKE = os.environ.get("TABM_SMOKE") == "1"

# TabM hyperparameters
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# node_0070 OOF path — for err-corr gate check
OOF_BANK_PATH = COMP_DIR / "nodes/node_0070/oof.npy"

# Lambda sweep at fold-0
LAMBDA_SWEEP = [0.1, 0.5]
# Kill thresholds
BA_KILL = 0.962
ERRCORR_KILL = 0.70

# Columns that are DIRECT functions of redshift — drop from inputs
# (redshift itself, log1p_redshift, g/redshift, i/redshift, and the int-floor cat for redshift)
REDSHIFT_DERIVED_COLS = {"_log1p_redshift", "_g_div_redshift", "_i_div_redshift"}
REDSHIFT_INPUT_BASE_COL = "redshift"  # raw redshift column — dropped from inputs
REDSHIFT_CAT_COL = "redshift_cat_"   # int-floor categorical view — also dropped from inputs


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_everything(SEED)

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
    """
    Stateless FE — safe to apply once. Redshift derivatives ARE computed but
    will be EXCLUDED from input features downstream (used only for aux target).
    We still compute them here so the dataframe schema is complete; the drop
    happens when building the model input arrays.
    """
    df = df.copy()

    # Redshift ratios (computed but excluded from model inputs)
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

    # Log1p of shifted redshift (computed but excluded from model inputs)
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")

    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Categorical encoding fit on train-fold only (fit_in_fold).
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    NOTE: redshift_cat_ will be built (for consistency) but excluded at model input time.
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


# ─── TabM with dual heads ─────────────────────────────────────────────────────

class TabMMultiTask(nn.Module):
    """
    Thin wrapper around tabm.TabM that adds an auxiliary scalar regression head
    for redshift prediction. The TabM trunk outputs (B, k, d_block)-ish features
    which are averaged across k before being passed to a linear head.

    We patch the forward: the trunk's last layer maps to class logits (d_out=3).
    For the aux head we tap the backbone's hidden rep before the output layer.

    Strategy: Use tabm.TabM.make(d_out=3) as class head. Add a separate nn.Linear
    mapping the TabM's penultimate d_block dimension to 1 for redshift regression.
    We achieve this by hooking into the backbone.
    """
    def __init__(self, base_model: tabm.TabM, d_block: int = 512, k: int = 32):
        super().__init__()
        self.base = base_model
        self.k = k
        # Aux head: from d_block → 1 scalar
        self.aux_head = nn.Linear(d_block, 1)
        nn.init.xavier_uniform_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)
        self._hook_hidden: torch.Tensor | None = None
        self._register_hook()

    def _register_hook(self):
        """Register a forward hook on the last backbone block to capture hidden rep."""
        # tabm.TabM architecture: backbone → output layer
        # We need to capture the output of the last block (before the final linear)
        # The last block is in model.backbone.blocks[-1] or similar
        # Let's introspect the model structure
        self._hook_handle = None
        # We'll capture hidden reps by hooking the module before the final linear
        # tabm internals: the model has a backbone with blocks and then a head
        # Find the last backbone block
        try:
            last_block = list(self.base.backbone.blocks)[-1]
            self._hook_handle = last_block.register_forward_hook(self._hook_fn)
            log("  Registered hook on base.backbone.blocks[-1]")
        except Exception as e:
            log(f"  WARNING: Could not register hook ({e}) — aux head will use zero tensor")

    def _hook_fn(self, module, input, output):
        # output may be (B, k, d_block) or (B, d_block)
        self._hook_hidden = output

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None):
        """Returns (class_logits (B,k,3), redshift_pred (B,1))."""
        self._hook_hidden = None
        class_logits = self.base(x_num, x_cat)  # (B, k, 3)

        if self._hook_hidden is not None:
            h = self._hook_hidden  # (B, k, d_block) or (B, d_block)
            if h.dim() == 3:
                h = h.mean(dim=1)  # (B, d_block) — average across ensemble
            # Ensure float32
            h = h.float()
            rs_pred = self.aux_head(h)  # (B, 1)
        else:
            # Fallback: zero prediction (shouldn't happen)
            B = class_logits.shape[0]
            rs_pred = torch.zeros(B, 1, device=class_logits.device)

        return class_logits, rs_pred

    def predict_class_proba(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        """Inference: returns (B, 3) class probabilities using class head only."""
        with torch.no_grad():
            class_logits, _ = self.forward(x_num, x_cat)
            return torch.softmax(class_logits.float(), dim=-1).mean(dim=1)


def build_tabm_multitask(n_num: int, cat_cards: list[int], bins: list) -> TabMMultiTask:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    base = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=N_CLASSES,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    # Introspect d_block from last backbone block's linear out_features
    try:
        last_blk = list(base.backbone.blocks)[-1]
        last_linear = list(last_blk.children())[0]
        d_block = last_linear.out_features
    except Exception:
        d_block = 512  # safe default
    model = TabMMultiTask(base, d_block=d_block, k=K_ENS)
    return model.to(DEVICE)


def predict_proba_batch(model: TabMMultiTask, Xn: np.ndarray, Xc: np.ndarray | None,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            probs = model.predict_class_proba(xn, xc)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm_multitask(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    rs_tr: np.ndarray,        # redshift values (standardized) for aux target
    cat_cards: list[int],
    fold_seed: int,
    lam: float,               # aux loss weight
) -> tuple[TabMMultiTask, np.ndarray]:
    """
    Train TabM with dual heads.
    rs_tr: standardized redshift (train fold only mean/std — fit_in_fold).
    lam: weight for Huber aux loss.
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    # PLR bins — fit on train portion only (fit-in-fold)
    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_multitask(Xn_tr.shape[1], cat_cards, bins)

    # Class weights (balanced)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_w)
    huber_loss_fn = nn.HuberLoss(delta=1.0)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # GPU tensors
    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    rs_t = torch.as_tensor(rs_tr[ti], dtype=torch.float32, device=DEVICE).unsqueeze(1)
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
            rs_b = rs_t[idx]

            opt.zero_grad()
            logits, rs_pred = model(xn_b, xc_b)  # (B, k, 3), (B, 1)

            # Classification loss (over k ensemble)
            b, k, c = logits.shape
            ce_loss = ce_loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))

            # Aux regression loss
            aux_loss = huber_loss_fn(rs_pred, rs_b)

            loss = ce_loss + lam * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Early-stop on internal val (class metric only)
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
    log(f"    TabM multitask: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}  lam={lam}")
    return model, bins


def compute_err_corr(oof_a: np.ndarray, oof_b: np.ndarray, y_true: np.ndarray) -> float:
    """
    Per-class error correlation between two OOF arrays vs node_0070 bank.
    Returns mean correlation of (1 - correct_class_prob) errors across classes.
    """
    # Use error = 1 - p_true_class as error signal
    err_a = np.array([1.0 - oof_a[i, y_true[i]] for i in range(len(y_true))])
    err_b = np.array([1.0 - oof_b[i, y_true[i]] for i in range(len(y_true))])
    corr = np.corrcoef(err_a, err_b)[0, 1]
    return float(corr)


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())
folds_list = folds_data["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
rs_all = train_raw["redshift"].values.astype(np.float32)  # aux target (from train only)
n_train = len(train_raw)
n_test = len(test_raw)

# Load node_0070 OOF for err-corr gate
oof_bank = np.load(OOF_BANK_PATH).astype(np.float32)
log(f"  oof_bank shape={oof_bank.shape}")

if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE (computed once) ─────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Columns to EXCLUDE from model inputs (redshift and its direct derivatives)
# These will be filtered out after FE
REDSHIFT_EXCLUDE_INPUT = {
    REDSHIFT_INPUT_BASE_COL,    # "redshift"
    REDSHIFT_CAT_COL,           # "redshift_cat_"
} | REDSHIFT_DERIVED_COLS       # "_log1p_redshift", "_g_div_redshift", "_i_div_redshift"

log(f"  Redshift columns excluded from inputs: {sorted(REDSHIFT_EXCLUDE_INPUT)}")

# ─── Pre-flight leakage checks ────────────────────────────────────────────────
log("PRE-FLIGHT leakage checks ...")
# 1. Target not in features
sample_feat_cols = [c for c in X_stateless.columns if c not in REDSHIFT_EXCLUDE_INPUT and c != TARGET and c != IDC]
assert TARGET not in sample_feat_cols, f"LEAK: target in features"
assert IDC not in sample_feat_cols, f"LEAK: id in features"
log("  [1] target/id not in features: OK")

# 2. Single-feature sweep on sample
samp = X_stateless.sample(min(50000, len(X_stateless)), random_state=0)
ys = pd.Series(y_all).iloc[samp.index].values
for c in sample_feat_cols[:30]:  # check first 30 non-redshift numeric cols
    x = pd.to_numeric(samp[c], errors="coerce")
    if x.nunique() > 1:
        xf = x.fillna(x.mean())
        corr_val = abs(np.corrcoef(xf.values, ys)[0, 1])
        if corr_val >= 0.999:
            raise SystemExit(f"LEAK smell: {c} ~ target corr={corr_val:.4f}")
log("  [2] single-feature sweep: clean")

# 3. Folds frozen (from folds.json — not recomputed)
log("  [3] folds: frozen from folds.json")

# 4. Redshift aux target: true redshift is a train feature, NOT the class label
assert "redshift" in train_raw.columns, "redshift col not found"
log("  [4] aux target = true redshift (legitimate — not class label)")
log("PRE-FLIGHT checks done.")

# ─── Phase 1: Fold-0 only, sweep lambda ───────────────────────────────────────
log("=" * 60)
log("PHASE 1: fold-0 lambda sweep")

fold0_data = folds_list[0]
assert fold0_data["fold"] == 0
val_idx_0 = np.asarray(fold0_data["val_idx"])
tr_idx_0 = np.setdiff1d(np.arange(n_train), val_idx_0)
fold_seed_0 = SEED + 100
seed_everything(fold_seed_0)

log(f"  Fold 0: train={len(tr_idx_0)} val={len(val_idx_0)}")

# Categorical encoding for fold-0 (fit_in_fold)
X_tr_f0, X_val_f0, X_te_f0, cat_cols_f0, combo_names_f0, lmap_f0 = fit_fold_categoricals(
    X_stateless.iloc[tr_idx_0].reset_index(drop=True),
    X_stateless.iloc[val_idx_0].reset_index(drop=True),
    X_test_stateless.copy(),
)

y_tr_f0 = y_all[tr_idx_0]
y_val_f0 = y_all[val_idx_0]

# Target encoding (fit_in_fold)
X_tr_f0, X_val_f0, X_te_f0, te_names_f0 = add_target_encoding(
    X_tr_f0, y_tr_f0, X_val_f0, X_te_f0, combo_names_f0, fold_seed_0
)

# Sort columns consistently
X_tr_f0 = X_tr_f0.reindex(sorted(X_tr_f0.columns), axis=1)
X_val_f0 = X_val_f0.reindex(sorted(X_val_f0.columns), axis=1)
X_te_f0 = X_te_f0.reindex(sorted(X_te_f0.columns), axis=1)

cat_cols_sorted_f0 = sorted(cat_cols_f0)

# EXCLUDE redshift and its direct derivatives from input features
def filter_cols_for_model(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in REDSHIFT_EXCLUDE_INPUT]

TABM_CAT_COLS_F0 = [c for c in cat_cols_sorted_f0 if c in BASE_CAT_COLS]
all_cols_sorted_f0 = sorted(X_tr_f0.columns)
# Filter out redshift-derived cols from numerical features
num_for_tabm_f0 = [c for c in all_cols_sorted_f0
                   if c not in TABM_CAT_COLS_F0 and c not in REDSHIFT_EXCLUDE_INPUT]

log(f"  n_all_cols={len(all_cols_sorted_f0)}  n_excluded_rs={len(REDSHIFT_EXCLUDE_INPUT)}")
log(f"  tabm_cat={len(TABM_CAT_COLS_F0)}  tabm_num={len(num_for_tabm_f0)}")

# Verify no redshift columns leak in
for c in REDSHIFT_EXCLUDE_INPUT:
    if c in num_for_tabm_f0 or c in TABM_CAT_COLS_F0:
        raise SystemExit(f"LEAK: {c} still in model inputs!")
log("  Verified: no redshift-derived cols in model input feature lists")

Xn_tr_f0 = X_tr_f0[num_for_tabm_f0].values.astype(np.float32)
Xn_va_f0 = X_val_f0[num_for_tabm_f0].values.astype(np.float32)
Xn_te_f0 = X_te_f0[num_for_tabm_f0].values.astype(np.float32)

if TABM_CAT_COLS_F0:
    Xc_tr_f0 = X_tr_f0[TABM_CAT_COLS_F0].values.astype(np.int64)
    Xc_va_f0 = X_val_f0[TABM_CAT_COLS_F0].values.astype(np.int64)
    Xc_te_f0 = X_te_f0[TABM_CAT_COLS_F0].values.astype(np.int64)
    cat_cards_f0 = (Xc_tr_f0.max(axis=0) + 2).tolist()
    card_arr = np.array(cat_cards_f0) - 1
    Xc_tr_f0 = np.clip(Xc_tr_f0, 0, card_arr)
    Xc_va_f0 = np.clip(Xc_va_f0, 0, card_arr)
    Xc_te_f0 = np.clip(Xc_te_f0, 0, card_arr)
else:
    Xc_tr_f0 = Xc_va_f0 = Xc_te_f0 = None
    cat_cards_f0 = []

# Standardize numerical features (fit on train fold only — fit_in_fold)
mu_f0 = Xn_tr_f0.mean(0)
sd_f0 = Xn_tr_f0.std(0) + 1e-8
Xn_tr_f0_std = (Xn_tr_f0 - mu_f0) / sd_f0
Xn_va_f0_std = (Xn_va_f0 - mu_f0) / sd_f0
Xn_te_f0_std = (Xn_te_f0 - mu_f0) / sd_f0

# Standardize redshift aux target — TRAIN FOLD ONLY (fit_in_fold)
rs_tr_f0 = rs_all[tr_idx_0]
rs_mu_f0 = rs_tr_f0.mean()
rs_sd_f0 = rs_tr_f0.std() + 1e-8
rs_tr_f0_std = (rs_tr_f0 - rs_mu_f0) / rs_sd_f0
log(f"  Redshift aux target: train-fold mean={rs_mu_f0:.4f}  std={rs_sd_f0:.4f}")

# Lambda sweep
best_lam = None
best_ba_f0 = -1.0
best_errcorr_f0 = None
fold0_results = {}

for lam in LAMBDA_SWEEP:
    log(f"\n  --- Lambda={lam} ---")
    seed_everything(fold_seed_0)
    model_f0, bins_f0 = train_tabm_multitask(
        Xn_tr_f0_std, Xc_tr_f0, y_tr_f0, rs_tr_f0_std,
        cat_cards_f0, fold_seed_0, lam=lam
    )

    val_probs_f0 = predict_proba_batch(model_f0, Xn_va_f0_std, Xc_va_f0)
    ba_f0 = balanced_accuracy_score(y_val_f0, val_probs_f0.argmax(1))

    # Err-corr vs node_0070 bank (fold-0 val rows only)
    oof_bank_f0 = oof_bank[val_idx_0]
    errcorr_f0 = compute_err_corr(val_probs_f0, oof_bank_f0, y_val_f0)

    log(f"  Lambda={lam}: fold-0 BA={ba_f0:.6f}  err-corr_vs_n070={errcorr_f0:.4f}")
    fold0_results[lam] = {"ba": ba_f0, "errcorr": errcorr_f0, "probs": val_probs_f0}

    if ba_f0 > best_ba_f0:
        best_ba_f0 = ba_f0
        best_lam = lam
        best_errcorr_f0 = errcorr_f0

    del model_f0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

log(f"\nLambda sweep complete: best_lam={best_lam}  best_BA={best_ba_f0:.6f}  err-corr={best_errcorr_f0:.4f}")

# ─── Gate checks ──────────────────────────────────────────────────────────────
gate_failed = None
if best_ba_f0 < BA_KILL:
    gate_failed = f"fold-0 BA {best_ba_f0:.4f} < {BA_KILL} (strength kill)"
    log(f"KILL: {gate_failed}")
elif best_errcorr_f0 >= ERRCORR_KILL:
    gate_failed = f"fold-0 err-corr {best_errcorr_f0:.4f} >= {ERRCORR_KILL} (no decorrelation)"
    log(f"KILL: {gate_failed}")

if gate_failed:
    log(f"STOPPING: {gate_failed}")
    # Write a partial node.md update
    print(f"GATE_FAILED: {gate_failed}", flush=True)
    print(f"best_lam={best_lam}  fold0_BA={best_ba_f0:.6f}  fold0_errcorr={best_errcorr_f0:.4f}", flush=True)
    print(f"cv=null", flush=True)
    sys.exit(0)

log(f"Gates passed! Proceeding with lambda={best_lam} for all 5 folds.")
log("=" * 60)

# ─── Phase 2: Full 5-fold OOF at best lambda ──────────────────────────────────
log("PHASE 2: Full OOF at best lambda")

# Re-use fold-0 val probs from best lambda (already computed above)
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None
num_cols_final = None

# Store fold-0 results (already computed)
oof_proba[val_idx_0] = fold0_results[best_lam]["probs"]
fold0_score = fold0_results[best_lam]["ba"]
per_fold_scores.append(fold0_score)
log(f"  Fold 0 (reused): BA={fold0_score:.6f}")
print(f"fold0_score={fold0_score:.6f}", flush=True)

# For test predictions at fold-0, we need to retrain fold-0 (could reuse but retrain for consistency)
# Actually let's retrain fold-0 to also get test predictions
log("  Retraining fold-0 to get test predictions ...")
seed_everything(fold_seed_0)
model_f0_best, bins_f0_best = train_tabm_multitask(
    Xn_tr_f0_std, Xc_tr_f0, y_tr_f0, rs_tr_f0_std,
    cat_cards_f0, fold_seed_0, lam=best_lam
)
test_probs_f0 = predict_proba_batch(model_f0_best, Xn_te_f0_std, Xc_te_f0)
test_proba_accum += test_probs_f0.astype(np.float32) / len(folds_list)

del model_f0_best, X_tr_f0, X_val_f0, X_te_f0, Xn_tr_f0, Xn_va_f0, Xn_te_f0
del Xn_tr_f0_std, Xn_va_f0_std, Xn_te_f0_std
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

fold_t0 = time.perf_counter()

# Remaining folds
for fi in folds_list[1:]:
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
    num_for_tabm = [c for c in all_cols_sorted
                    if c not in TABM_CAT_COLS and c not in REDSHIFT_EXCLUDE_INPUT]

    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = num_for_tabm
        log(f"  n_features={X_tr_fold.shape[1]}  tabm_cat={len(TABM_CAT_COLS)}  tabm_num={len(num_for_tabm)}")

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

    # Standardize — fit on train fold only
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Redshift aux target — standardize using train fold only
    rs_tr_fold = rs_all[tr_idx]
    rs_mu = rs_tr_fold.mean()
    rs_sd = rs_tr_fold.std() + 1e-8
    rs_tr_std = (rs_tr_fold - rs_mu) / rs_sd

    model, bins = train_tabm_multitask(
        Xn_tr, Xc_tr, y_tr_fold, rs_tr_std,
        cat_cards, fold_seed, lam=best_lam
    )

    val_probs = predict_proba_batch(model, Xn_va, Xc_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Full err-corr on OOF ─────────────────────────────────────────────────────
full_errcorr = compute_err_corr(oof_proba, oof_bank, y_all)
log(f"Full OOF err-corr vs node_0070: {full_errcorr:.4f}")
print(f"full_errcorr={full_errcorr:.4f}", flush=True)

# ─── Save artifacts ───────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# Features file
tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
