"""node_0141 — TabM dual-head (CE + pairwise-margin ranking) on fs_realmlp_fe.

THE ONE ATOMIC CHANGE vs node_0033:
  Add a pairwise-margin / contrastive ranking head alongside the standard CE head.
  The CE head outputs 3-class posteriors; the ranking head outputs a scalar score per
  sample used only during training. A pairwise-margin loss (torch.nn.functional.
  margin_ranking_loss) pushes the score of the correctly-labelled class above the
  confusable partner. Two losses summed: L = CE + alpha * ranking_loss.
  OOF/test output is the CE head's 3-class posterior (unchanged format).

WILDCARD hypothesis: the ranking objective explicitly maximises separation margin on
confusable pairs (GAL/QSO, GAL/STAR) shaping the trunk representation differently
from pure CE; this may produce a decorrelated error structure.

FE pipeline: byte-identical to node_0033 (fs_realmlp_fe).
Leakage: pair sampling draws ONLY from the train-fold batch; val/test rows never
         paired during training. OOF = CE head's held-out fold predictions.
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
import torch.nn.functional as F
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
FOLD0_ONLY = os.environ.get("TABM_FOLD0_ONLY") == "1"

# TabM hyperparameters (byte-identical to node_0033)
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# Ranking loss weight — small so CE dominates; ranking shapes the trunk
RANK_ALPHA = 0.1   # L = CE + 0.1 * ranking_loss
RANK_MARGIN = 1.0  # margin for margin_ranking_loss


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

# ─── Feature engineering (byte-identical to node_0033) ───────────────────────
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


def fit_fold_categoricals(df_tr, df_val, df_te):
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


# ─── Dual-head TabM model ─────────────────────────────────────────────────────

class TabMDualHead(nn.Module):
    """TabM trunk with CE head + ranking head.

    The trunk is a standard TabM (k=32 PLR). Two heads share the trunk:
      - CE head (base.output, LinearEnsemble): (B, k, 3) logits for classification
      - Rank head: a new LinearEnsemble that reads the same (B, k, d_block) trunk output

    TabM architecture: num_emb -> ensemble_view -> backbone (MLPBackboneBatchEnsemble)
                        -> output (LinearEnsemble(k, d_block, d_out=3))
    The output LinearEnsemble input is (B, k, d_block=512).
    We hook on base.output to capture the (B, k, 512) trunk representation,
    then feed it to a parallel rank_head (a small linear outputting (B, k) scalars).
    """

    def __init__(self, base_model: tabm.TabM):
        super().__init__()
        self.base = base_model
        # Inspect the output LinearEnsemble to get d_block
        # output.weight shape: (k, d_block, d_out)
        out_w = base_model.output.weight   # (k, d_block, 3)
        k, d_block, d_out = out_w.shape
        self._k = k
        self._d_block = d_block
        dev = next(base_model.parameters()).device
        # Ranking head: maps (B, k, d_block) -> (B, k, 1) then squeeze to (B, k)
        # Implemented as a parameter tensor + manual bmm to mirror LinearEnsemble
        # weight: (k, d_block, 1)
        self.rank_weight = nn.Parameter(torch.empty(k, d_block, 1, device=dev))
        self.rank_bias = nn.Parameter(torch.zeros(k, device=dev))
        nn.init.normal_(self.rank_weight, 0.0, 0.01)

    def _run_with_trunk_hook(self, x_num, x_cat):
        """Run base forward, capturing trunk output before the CE head."""
        trunk_holder = []

        def hook(module, inp, out):
            # inp[0]: (B, k, d_block)
            trunk_holder.append(inp[0])

        h = self.base.output.register_forward_hook(hook)
        ce_logits = self.base(x_num, x_cat)   # (B, k, 3)
        h.remove()
        trunk = trunk_holder[0]   # (B, k, d_block)
        return ce_logits, trunk

    def forward(self, x_num, x_cat):
        """Returns (ce_logits, rank_scores).
        ce_logits: (B, k, 3)
        rank_scores: (B, k) — one scalar per ensemble member per sample
        """
        ce_logits, trunk = self._run_with_trunk_hook(x_num, x_cat)
        # trunk: (B, k, d_block)
        # rank_weight: (k, d_block, 1) => per-ensemble-member linear
        # einsum: (B, k, d_block) x (k, d_block, 1) -> (B, k, 1)
        rank_raw = torch.einsum("bkd,kdj->bkj", trunk, self.rank_weight)  # (B, k, 1)
        rank_raw = rank_raw.squeeze(-1) + self.rank_bias.unsqueeze(0)     # (B, k) + (1, k)
        return ce_logits, rank_raw


def pairwise_ranking_loss(rank_scores: torch.Tensor, labels: torch.Tensor,
                          margin: float = RANK_MARGIN) -> torch.Tensor:
    """Pairwise margin ranking loss on confusable class pairs.

    For each pair of samples (i, j) in the batch:
      - If label[i] != label[j], we want rank_score[i] > rank_score[j]
        when label[i] is the 'harder' class (STAR=2 or QSO=1 vs GALAXY=0).
    We prioritise minority-class margin: STAR > GALAXY, QSO > GALAXY.
    Uses torch.nn.functional.margin_ranking_loss (library-first per CLAUDE.md rule 8).

    rank_scores: (B, k) — average over k for a scalar per sample
    labels: (B,) long
    """
    # Average over ensemble members for a single score per sample
    s = rank_scores.mean(dim=1)   # (B,)

    # Build pairs: positive = minority class sample, negative = GALAXY sample
    # Classes: GALAXY=0, QSO=1, STAR=2
    # We want STAR and QSO ranked above GALAXY (the confusable boundary).
    galaxy_mask = (labels == 0)
    minority_mask = (labels > 0)   # QSO or STAR

    s_pos = s[minority_mask]   # minority-class scores
    s_neg = s[galaxy_mask]     # galaxy scores

    if s_pos.numel() == 0 or s_neg.numel() == 0:
        return torch.tensor(0.0, device=rank_scores.device, dtype=rank_scores.dtype)

    # Broadcast: (n_pos, n_neg) pairs
    # Limit to at most 512 pos and 512 neg to avoid quadratic explosion
    max_pairs = 512
    if s_pos.numel() > max_pairs:
        idx = torch.randperm(s_pos.numel(), device=s_pos.device)[:max_pairs]
        s_pos = s_pos[idx]
    if s_neg.numel() > max_pairs:
        idx = torch.randperm(s_neg.numel(), device=s_neg.device)[:max_pairs]
        s_neg = s_neg[idx]

    # (n_pos, n_neg) broadcast
    s1 = s_pos.unsqueeze(1).expand(-1, s_neg.numel())   # (n_pos, n_neg)
    s2 = s_neg.unsqueeze(0).expand(s_pos.numel(), -1)   # (n_pos, n_neg)
    tgt = torch.ones_like(s1)   # we want s1 > s2

    loss = F.margin_ranking_loss(s1.reshape(-1), s2.reshape(-1),
                                 tgt.reshape(-1), margin=margin)
    return loss


def build_tabm_dual(n_num: int, cat_cards: list[int], bins: list) -> TabMDualHead:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    base = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=N_CLASSES,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    base = base.to(DEVICE)
    model = TabMDualHead(base)
    model = model.to(DEVICE)
    return model


def predict_proba_batch(model: TabMDualHead, Xn: np.ndarray, Xc: np.ndarray | None,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            ce_logits, _ = model(xn, xc)    # (B, k, 3)
            probs = torch.softmax(ce_logits.float(), dim=-1).mean(dim=1)  # (B, 3)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm_dual(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
) -> tuple[TabMDualHead, list]:
    """Train dual-head TabM. PLR bins fit on train-fold ti subset (fit_in_fold).
    Returns (best_model, bins).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    # PLR bins — fit on TRAIN portion only (fit_in_fold, target-aware)
    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_dual(Xn_tr.shape[1], cat_cards, bins)

    # Class weights (balanced) for CE loss
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_w)

    # Optimize all parameters together (trunk + CE head + rank head)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # Move train data to GPU
    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
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
        ep_ce_loss = 0.0
        ep_rank_loss = 0.0
        n_batches = 0

        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_t[idx] if Xc_t is not None else None
            y_b = y_t[idx]

            opt.zero_grad()
            ce_logits, rank_scores = model(xn_b, xc_b)  # (B, k, 3), (B, k)

            # CE loss over all k ensemble members
            b, k, c = ce_logits.shape
            ce_loss = ce_loss_fn(ce_logits.reshape(b * k, c), y_b.repeat_interleave(k))

            # Pairwise ranking loss — pairs drawn from THIS train-fold batch only
            # rank_scores: (B, k); labels: (B,)
            rank_loss = pairwise_ranking_loss(rank_scores, y_b, margin=RANK_MARGIN)

            loss = ce_loss + RANK_ALPHA * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_ce_loss += ce_loss.item()
            ep_rank_loss += rank_loss.item()
            n_batches += 1

        scheduler.step()

        # Early-stop on internal val (CE head only, no label leakage)
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

        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"    ep={ep+1:3d}  ce={ep_ce_loss/n_batches:.4f}  "
                f"rank={ep_rank_loss/n_batches:.4f}  int_ba={ba:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    TabM dual-head early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model, bins


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

if SMOKE:
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# Fold-0-only mode for cheap-kill check
if FOLD0_ONLY and not SMOKE:
    log("FOLD0_ONLY mode: running fold 0 only for cheap-kill check")
    folds_list = [folds_list[0]]

# ─── Pre-flight leakage checks 1-3 ───────────────────────────────────────────
log("Pre-flight leakage checks ...")
assert TARGET not in train_raw.drop(columns=[TARGET]).columns, "LEAK: target in features"
assert IDC not in train_raw.drop(columns=[IDC, TARGET]).columns, "LEAK: id in features"
# single-feature↔target sweep on a 50k sample
_sample_size = min(50000, n_train)
_s = train_raw.sample(_sample_size, random_state=0)
_ys = _s[TARGET].map(LABEL_MAP).values.astype(float)
_feature_cols = [c for c in train_raw.columns if c not in [TARGET, IDC]]
for _c in _feature_cols:
    _x = pd.to_numeric(_s[_c], errors="coerce").fillna(0).values.astype(float)
    if len(np.unique(_x)) > 1:
        corr = abs(np.corrcoef(_x, _ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {_c} ~ target corr={corr:.4f}")
log("  Pre-flight checks OK (no target/id leak, no near-perfect corr)")

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None
num_cols_final = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

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
    TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
    all_cols_sorted = sorted(X_tr_fold.columns)
    num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

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

    # Standardize numerical features — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Train dual-head TabM
    model, bins = train_tabm_dual(Xn_tr, Xc_tr, y_tr_fold, cat_cards, fold_seed)

    # OOF predictions (CE head only)
    val_probs = predict_proba_batch(model, Xn_va, Xc_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions — average across folds
    test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
    n_folds_total = len(json.loads((COMP_DIR / "folds.json").read_text())["folds"])
    test_proba_accum += test_probs_fold.astype(np.float32) / n_folds_total

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

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * n_folds_total
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # ─── CHEAP-KILL CHECK (fold-0 only) ──────────────────────────────────
        fold0_ba = fold_score
        log(f"CHEAP-KILL CHECK: fold-0 BA={fold0_ba:.6f}")

        # Load node_0070 OOF for error correlation check
        n70_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
        if n70_oof_path.exists():
            n70_oof = np.load(str(n70_oof_path))
            # err-corr on fold-0 val rows
            n70_preds_val = n70_oof[val_idx].argmax(1)
            n141_preds_val = oof_proba[val_idx].argmax(1)
            # Error correlation: correlation of binary error indicators
            n70_err = (n70_preds_val != y_val_fold).astype(float)
            n141_err = (n141_preds_val != y_val_fold).astype(float)
            if n70_err.std() > 0 and n141_err.std() > 0:
                err_corr = float(np.corrcoef(n70_err, n141_err)[0, 1])
            else:
                err_corr = 1.0
            log(f"CHEAP-KILL: err-corr vs n070 on fold-0 val = {err_corr:.4f}")
            print(f"fold0_err_corr_vs_n070={err_corr:.4f}", flush=True)
        else:
            log("WARNING: node_0070/oof.npy not found — skipping err-corr check")
            err_corr = 0.0  # assume OK

        # Kill criterion: err_corr >= 0.65 OR BA < 0.960
        kill = (err_corr >= 0.65) or (fold0_ba < 0.960)
        log(f"CHEAP-KILL decision: err_corr={err_corr:.4f} BA={fold0_ba:.6f}  kill={kill}")
        print(f"cheap_kill={kill}  err_corr={err_corr:.4f}  fold0_ba={fold0_ba:.6f}", flush=True)

        if kill:
            log("CHEAP-KILL TRIGGERED — stopping after fold 0. Node will be marked dead.")
            print(f"cv={fold0_ba:.6f}", flush=True)
            log(f"Total elapsed: {time.perf_counter() - T0:.1f}s")
            log("Exiting after cheap-kill.")
            sys.exit(0)
        else:
            log("CHEAP-KILL PASSED — continuing to full 5-fold.")

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
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
tabm_cat_in_file = [c for c in (cat_cols_final or []) if c in BASE_CAT_COLS]
all_features = sorted((num_cols_final or []) + tabm_cat_in_file)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
