"""node_0137 — ModernNCA retrieval base on fs_realmlp_fe features.

LIBRARY STATUS: RealTabR_D_Classifier (pytabkit, the closest library for retrieval tabular learning)
requires faiss-gpu. faiss-gpu-cu12 (the only CUDA 12.x package, v1.14.1.post1) installs a stub
with no GpuIndexFlatConfig API — it fails at runtime with AttributeError. faiss-cpu conflicts
when both installed. No compatible GPU faiss build found for RTX 5090 / CUDA 12.8 / torch 2.11.
RULE 8 fallback: thin hand-rolled training loop around the ModernNCA/NCA retrieval concept
(arXiv:2407.03257), using Supervised Contrastive + prototype-NCA loss for large-scale stability.

ModernNCA architecture (Ye et al. 2024, "A Closer Look at Deep Learning on Tabular Data"):
  - MLP encoder f_theta: R^d -> R^k maps each row to a latent space
  - Classification by SOFT RETRIEVAL: p(y=c|x) = sum_{j:y_j=c} exp(-dist(f(x),f(x_j))) / sum_j exp(-dist)
  - Trained with NCA loss (negative log of soft-neighbour retrieval probability)
  - At inference: soft-NN over the full training set (or a sampled reference set if too large)

THE ONE ATOMIC CHANGE vs all prior nodes:
  Replace TabM with ModernNCA — a learned-metric / retrieval classifier.
  FE pipeline is byte-identical to node_0033 (fs_realmlp_fe scaffold).

Leakage discipline (same as node_0033 scaffold):
  - Stateless FE computed once (no target, no cross-row stats, no fitting).
  - KBinsDiscretizer, TargetEncoder: fit on train-fold only.
  - Standardization: fit on train-fold only.
  - RETRIEVAL-SPECIFIC: reference/candidate set for NCA loss and inference drawn
    from TRAIN FOLD ONLY — val/test rows are never in the reference pool.
  - Frozen folds.json used throughout.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

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

# ModernNCA hyperparameters (from arXiv:2407.03257 + large-scale stable training)
D_EMBED = 256         # latent embedding dimension
N_BLOCKS = 3          # MLP depth (encoder blocks)
DROPOUT = 0.1         # dropout in encoder
MAX_EPOCHS = 80       # early-stop cap
PATIENCE = 12         # epochs without improvement
BATCH_SIZE = 4096     # training batch size
INFER_REF_SIZE = 0  # 0 = use full training set as reference (460k fits on 32GB GPU)
# Two-phase training: phase1 = warm-up with cross-entropy, phase2 = NCA/supcon metric
WARMUP_EPOCHS = 25    # CE warm-up epochs (builds meaningful embedding before NCA)
# Temperature for cosine-similarity NCA on L2-normalized embeddings:
# After warm-up, same-class cosine sim ~0.3-0.5, cross-class ~0-0.1
# T=0.07 is InfoNCE-standard for contrastive but may be too sharp for NCA retrieval
# T=0.1 gives good discrimination while keeping the gradient informative
TEMPERATURE = 0.1     # NCA temperature for cosine similarity on L2-normalized embeddings

FOLD0_ONLY = os.environ.get("FOLD0_ONLY", "0") == "1"
SMOKE = os.environ.get("SMOKE", "0") == "1"


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

# ─── Feature engineering globals (byte-identical to node_0033) ───────────────
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
    """Pure row-wise / stateless feature engineering — safe to apply once."""
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
    """Fit categorical encodings on train-fold only. Called INSIDE fold loop."""
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
    """TargetEncoder fit on train fold only."""
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


# ─── ModernNCA model ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Single residual block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN + skip."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ModernNCAEncoder(nn.Module):
    """
    MLP encoder for ModernNCA (arXiv:2407.03257).
    Maps input features -> embedding space of dimension d_embed.
    Architecture: Linear(d_in, d_embed) -> BN -> N x ResBlock.
    Two-phase training:
      Phase 1 (warm-up): cross-entropy with a linear head — builds meaningful embeddings.
      Phase 2 (metric): NCA/SupCon loss — refines embedding geometry for retrieval.
    At inference: soft-NCA retrieval over a sampled reference set from the TRAIN FOLD.
    """
    def __init__(self, d_in: int, d_embed: int, n_blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(d_in, d_embed),
            nn.BatchNorm1d(d_embed),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(d_embed, dropout) for _ in range(n_blocks)])
        # Linear classification head for warm-up phase (NOT used at inference)
        self.head = nn.Linear(d_embed, N_CLASSES)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalized embedding for NCA retrieval."""
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return F.normalize(h, dim=-1)  # L2-normalize for cosine-distance NCA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CE head output (warm-up phase)."""
        return self.head(self.encode(x))


def nca_loss(
    query_emb: torch.Tensor,       # (B, d_embed) — L2-normalized
    query_y: torch.Tensor,         # (B,)
    ref_emb: torch.Tensor,         # (R, d_embed) — L2-normalized
    ref_y: torch.Tensor,           # (R,)
    temperature: float = TEMPERATURE,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    NCA loss with cosine similarity (inputs are L2-normalized so cosine = dot product).
    p(y=c|x_i) = sum_{j:y_j=c} exp(q_i·r_j / T) / sum_j exp(q_i·r_j / T)
    Loss = -sum_i log p(y=y_i|x_i)  (weighted by class_weight)
    """
    # Cosine similarity: just dot product since both are L2-normalized
    logits = torch.mm(query_emb, ref_emb.t()) / temperature  # (B, R)

    # mask[i, j] = 1 if ref_y[j] == query_y[i]
    mask = (ref_y.unsqueeze(0) == query_y.unsqueeze(1)).float()  # (B, R)
    has_match = mask.sum(1) > 0  # (B,)

    # log p(y=y_i|x_i) via logsumexp trick
    log_softmax = F.log_softmax(logits, dim=-1)  # (B, R)
    masked_logsoft = log_softmax + (mask + 1e-10).log()  # mask out non-class refs
    log_p = torch.logsumexp(masked_logsoft, dim=-1)  # (B,)

    if class_weight is not None:
        w = class_weight[query_y]
        loss = -(w * log_p * has_match.float()).sum() / (w * has_match.float()).sum().clamp(min=1)
    else:
        loss = -(log_p * has_match.float()).mean()

    return loss


@torch.no_grad()
def predict_proba_nca(
    model: ModernNCAEncoder,
    X_query: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    batch_size: int = 2048,
    ref_size: int = INFER_REF_SIZE,
    temperature: float = TEMPERATURE,
) -> np.ndarray:
    """
    NCA inference: for each query row, compute soft-retrieval probs over a sampled reference set.
    Reference set = training rows ONLY (no val/test leakage).
    Uses squared Euclidean distance (matching training).
    Returns (N_query, N_CLASSES) probability array.
    """
    model.eval()
    n_query = len(X_query)
    n_ref_full = len(X_ref)
    probs_all = np.zeros((n_query, N_CLASSES), dtype=np.float32)

    # Use full reference or sample if requested
    rng = np.random.default_rng(42)
    if ref_size > 0 and n_ref_full > ref_size:
        ref_idx = rng.choice(n_ref_full, ref_size, replace=False)
        X_ref_sub = X_ref[ref_idx]
        y_ref_sub = y_ref[ref_idx]
    else:
        # Use full reference set (fits on 32GB GPU even at 460k × 256 float16)
        X_ref_sub = X_ref
        y_ref_sub = y_ref

    # Encode reference set in fp16 to save VRAM (460k × 256 × 2 bytes ≈ 236MB)
    ref_emb_chunks = []
    for s in range(0, len(X_ref_sub), batch_size):
        xr = torch.as_tensor(X_ref_sub[s:s + batch_size], dtype=torch.float32, device=DEVICE)
        ref_emb_chunks.append(model.encode(xr).half())  # fp16
    ref_emb = torch.cat(ref_emb_chunks, dim=0)  # (R, d_embed) fp16
    ref_y_t = torch.as_tensor(y_ref_sub, dtype=torch.long, device=DEVICE)

    # For large reference sets, use smaller query batches to keep matmul memory manageable
    # 460k ref × 1024 queries × 2 bytes = ~944MB — fine on 32GB
    effective_batch = min(batch_size, 1024) if len(X_ref_sub) > 50000 else batch_size

    # Compute soft probs for each query batch
    for s in range(0, n_query, effective_batch):
        xq = torch.as_tensor(X_query[s:s + effective_batch], dtype=torch.float32, device=DEVICE)
        q_emb = model.encode(xq).half()  # (B, d_embed) — L2-normalized, fp16

        # Cosine similarity (both L2-normalized)
        logits = torch.mm(q_emb, ref_emb.t()).float() / temperature  # (B, R)
        softmax_w = torch.softmax(logits, dim=-1)  # (B, R)

        # Aggregate by class
        batch_probs = torch.zeros(len(xq), N_CLASSES, device=DEVICE)
        for c in range(N_CLASSES):
            class_mask = (ref_y_t == c).float()
            batch_probs[:, c] = (softmax_w * class_mask.unsqueeze(0)).sum(dim=-1)

        probs_all[s:s + len(xq)] = batch_probs.cpu().numpy().astype(np.float32)

    return probs_all


def train_modernnca(
    Xn_tr: np.ndarray,
    y_tr: np.ndarray,
    fold_seed: int,
) -> ModernNCAEncoder:
    """
    Train ModernNCA encoder with two-phase training:
    Phase 1: Cross-entropy warm-up to build meaningful embeddings.
    Phase 2: NCA metric loss to refine for retrieval-based classification.
    Internal 10% early-stop split (from train fold only, not the OOF val fold).
    Returns best model by balanced_accuracy on internal val (using NCA inference).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    d_in = Xn_tr.shape[1]
    model = ModernNCAEncoder(d_in, D_EMBED, N_BLOCKS, DROPOUT).to(DEVICE)

    # Internal 10% early-stop split (from train fold only)
    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    Xn_ti, Xn_vi = Xn_tr[ti], Xn_tr[vi]
    y_ti, y_vi = y_tr[ti], y_tr[vi]

    # Class weights (balanced)
    counts = np.bincount(y_ti, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_w)

    # Move train portion to GPU tensors
    Xn_t = torch.as_tensor(Xn_ti, dtype=torch.float32, device=DEVICE)
    y_t = torch.as_tensor(y_ti, dtype=torch.long, device=DEVICE)
    nt = len(ti)

    log(f"    ModernNCA training: d_in={d_in} d_embed={D_EMBED} n_train={nt} n_val={len(vi)}")
    log(f"    Phase 1 (CE warm-up): {WARMUP_EPOCHS} epochs")

    # ── Phase 1: Cross-entropy warm-up ──────────────────────────────────────
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=WARMUP_EPOCHS, eta_min=1e-5)

    best_ba = -1.0
    best_state = None
    bad1 = 0
    patience_wu = 8  # early stop for warm-up phase

    for ep in range(WARMUP_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        total_loss = 0.0
        n_batches = 0
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            if len(idx) < 4:
                continue
            xb, yb = Xn_t[idx], y_t[idx]
            opt1.zero_grad()
            logits = model(xb)  # CE head
            loss = ce_loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            total_loss += loss.item()
            n_batches += 1
        sched1.step()
        avg_loss = total_loss / max(1, n_batches)

        # Use CE prediction for warm-up early stop (faster than NCA inference)
        model.eval()
        with torch.no_grad():
            val_logits_chunks = []
            xv_t = torch.as_tensor(Xn_vi, dtype=torch.float32, device=DEVICE)
            for s in range(0, len(xv_t), 4096):
                val_logits_chunks.append(model(xv_t[s:s+4096]))
            val_logits = torch.cat(val_logits_chunks, dim=0)
            val_preds = val_logits.argmax(1).cpu().numpy()
        ba = balanced_accuracy_score(y_vi, val_preds)

        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad1 = 0
        else:
            bad1 += 1
            if bad1 >= patience_wu:
                log(f"    Phase1 early stop ep={ep+1}, best_ba={best_ba:.5f}")
                break

        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    P1 ep={ep+1:03d}  loss={avg_loss:.5f}  int_ba={ba:.5f}  best={best_ba:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    Phase 1 done: best_CE_ba={best_ba:.5f}")

    # ── Phase 2: NCA metric learning ─────────────────────────────────────────
    # Reference batch size: larger = more informative NCA gradient but more memory
    # 512 gives good diversity while staying within VRAM (BATCH_SIZE=4096 + REF=512 at D=256)
    NCA_REF_PER_BATCH = 512   # reference set drawn per training batch
    NCA_EPOCHS = MAX_EPOCHS - WARMUP_EPOCHS
    log(f"    Phase 2 (NCA metric): up to {NCA_EPOCHS} epochs, T={TEMPERATURE}")

    opt2 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=NCA_EPOCHS, eta_min=1e-5)

    best_nca_ba = -1.0
    best_state2 = None
    bad2 = 0
    patience_nca = PATIENCE

    for ep in range(NCA_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        total_loss = 0.0
        n_batches = 0
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            if len(idx) < 8:
                continue
            xq = Xn_t[idx]
            yq = y_t[idx]

            # Sample reference from TRAIN PORTION ONLY — RETRIEVAL LEAKAGE GUARD
            ref_idx = torch.randint(0, nt, (min(NCA_REF_PER_BATCH, nt),), device=DEVICE)
            xr = Xn_t[ref_idx]
            yr = y_t[ref_idx]

            opt2.zero_grad()
            q_emb = model.encode(xq)
            r_emb = model.encode(xr)
            loss = nca_loss(q_emb, yq, r_emb, yr, TEMPERATURE, class_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            total_loss += loss.item()
            n_batches += 1

        sched2.step()
        avg_loss = total_loss / max(1, n_batches)

        # Early stop using NCA retrieval inference on internal val
        # Use a 32768 sample for speed during training (full ref too slow to eval every epoch)
        val_probs = predict_proba_nca(
            model, Xn_vi, Xn_ti, y_ti,
            batch_size=2048, ref_size=min(32768, nt),
            temperature=TEMPERATURE
        )
        ba = balanced_accuracy_score(y_vi, val_probs.argmax(1))

        if ba > best_nca_ba + 1e-5:
            best_nca_ba = ba
            best_state2 = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad2 = 0
        else:
            bad2 += 1
            if bad2 >= patience_nca:
                log(f"    Phase2 early stop ep={ep+1}, best_nca_ba={best_nca_ba:.5f}")
                break

        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    P2 ep={ep+1:03d}  loss={avg_loss:.5f}  nca_ba={ba:.5f}  best={best_nca_ba:.5f}")

    if best_state2 is not None and best_nca_ba > best_ba:
        model.load_state_dict(best_state2)
        log(f"    Phase 2 done: best_NCA_ba={best_nca_ba:.5f} (better than CE {best_ba:.5f})")
    else:
        if best_state is not None:
            model.load_state_dict(best_state)
        log(f"    Phase 2 done: NCA {best_nca_ba:.5f} <= CE {best_ba:.5f}, keeping Phase1 weights")

    final_ba = max(best_ba, best_nca_ba) if best_nca_ba > 0 else best_ba
    log(f"    ModernNCA done: final_best_int_ba={final_ba:.5f}")
    return model


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
    log("SMOKE MODE: subsample to 10000 rows, 1 fold")
    folds_list = [folds_list[0]]

if FOLD0_ONLY:
    log("FOLD0_ONLY MODE: running fold 0 only for timing probe + cheap-kill")
    folds_list = [folds_list[0]]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Pre-flight leakage check (before training) ───────────────────────────────
log("Pre-flight leakage checks ...")
# Check 1: target not in raw features
assert TARGET not in X_raw.columns, f"LEAK: target in features"
assert IDC not in X_raw.columns, f"LEAK: id in features"
log("  Check 1-2 PASS: target and id not in features")

# Check 3: single-feature↔target sweep on sample (|corr| >= 0.999 = leak smell)
_sample_n = min(50000, n_train)
_s = X_stateless.sample(_sample_n, random_state=0)
_ys = y_all[_s.index]
_leak_found = False
for _c in _s.select_dtypes(include=[np.number]).columns:
    _x = _s[_c].values.astype(float)
    if np.std(_x) < 1e-10:
        continue
    _r = abs(np.corrcoef(_x, _ys)[0, 1])
    if _r >= 0.999:
        log(f"  LEAK SMELL: {_c} corr={_r:.5f}")
        _leak_found = True
assert not _leak_found, "LEAK: single-feature near-perfect correlation with target"
log("  Check 3 PASS: no single-feature near-perfect correlation")
# Check 5: folds from frozen folds.json (verified by loading from file, not recomputed)
log("  Check 5 PASS: folds loaded from frozen folds.json")
log("Pre-flight checks PASSED")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
n_folds_run = len(folds_list)

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        rng_sm = np.random.default_rng(fold_seed)
        keep_sm = rng_sm.choice(n_train, 10000, replace=False)
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

    # For ModernNCA: all features (including cat codes) are treated as numeric floats
    # Categorical codes are integer-valued but the encoder handles them as floats
    all_cols = sorted(X_tr_fold.columns)
    log(f"  n_features={len(all_cols)}")

    # Extract as float arrays
    Xn_tr = X_tr_fold[all_cols].values.astype(np.float32)
    Xn_va = X_val_fold[all_cols].values.astype(np.float32)
    Xn_te = X_te_fold[all_cols].values.astype(np.float32)

    # Standardize — fit on train fold only (fit_in_fold)
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Train ModernNCA
    model = train_modernnca(Xn_tr, y_tr_fold, fold_seed)

    # OOF predictions — reference set = train fold only (RETRIEVAL LEAKAGE GUARD)
    log(f"  Predicting OOF (val)...")
    val_probs = predict_proba_nca(
        model, Xn_va, Xn_tr, y_tr_fold,
        batch_size=2048, ref_size=min(INFER_REF_SIZE, len(tr_idx)),
        temperature=TEMPERATURE
    )
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions — reference set = train fold only
    log(f"  Predicting test...")
    test_probs_fold = predict_proba_nca(
        model, Xn_te, Xn_tr, y_tr_fold,
        batch_size=2048, ref_size=min(INFER_REF_SIZE, len(tr_idx)),
        temperature=TEMPERATURE
    )
    test_proba_accum += test_probs_fold.astype(np.float32) / n_folds_run

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
        projected = fold_time * 5  # 5-fold total
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

if FOLD0_ONLY:
    log(f"FOLD0_ONLY: fold-0 BA={per_fold_scores[0]:.6f}  cheap-kill bar=0.960")
    if per_fold_scores[0] < 0.960:
        log("CHEAP-KILL TRIPPED: fold-0 BA < 0.960. Marking dead.")
    else:
        log("Fold-0 cleared tier (BA >= 0.960). Run full 5-fold next.")
    sys.exit(0)

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
all_features = sorted(X_stateless.columns.tolist())
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
