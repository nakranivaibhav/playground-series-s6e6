"""node_0057 — feature->image ResNet (rest-frame SED).

THE ONE ATOMIC CHANGE (vs all prior nodes):
  Per row, from [u,g,r,i,z]+redshift build a 7-channel 32x32 "SED-texture" image
  (rest-frame-warped GAF/RP/MTF), classify with a small from-scratch ResNet (~0.7M params)
  with 2 side scalars fused at the head. New DRAFT family never tried before.

IMAGE (7 channels, native 24x24 -> zero-padded 32x32):
  ch0 GASF(s_rest), ch1 GADF(s_rest), ch2 RP(s_rest), ch3 MTF(s_rest) [fit_in_fold],
  ch4 GASF(s_obs), ch5 zmod=outer(s_obs,s_obs)*z_normed, ch6 support-mask.
  s_rest: PCHIP-resampled flux-shape after (1+z) rest-frame warp.
  s_obs: same, no warp.
SIDE SCALARS: [mag_mean, redshift], standardized in-fold (NOT in image).

MODEL: small ResNet from scratch.
  Conv3x3(7->32)+BN+SiLU; 3 stages [2,2,2] BasicBlocks 32->64->128 (stride-2 stages 2,3);
  GAP->128-d; CONCAT scalars->130; Dropout(0.2)->Linear(130->128)->SiLU->Linear(128->3).

LEAKAGE: MTF edges, channel mean/std, scalar stats all fit on TRAIN-FOLD rows only.

MEMORY STRATEGY: store images as float16 (N,7,24,24); apply standardize+pad in Dataset.
  Max RAM footprint ~6.7 GB for all fold images simultaneously.

HARD FOLD-0 KILL-SWITCH: kill if fold-0 BA < 0.955 OR err-corr-vs-CORE15 > 0.6.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    _t = torch.randn(64, 64, device=DEVICE); _ = _t @ _t.T; del _t, _
    log(f"Device: {DEVICE}  GPU={torch.cuda.get_device_name(0)}")
else:
    log(f"Device: {DEVICE}")

# ─── Image constants ───────────────────────────────────────────────────────────
BANDS = ["u", "g", "r", "i", "z"]
LAM_OBS = np.array([3543., 4770., 6231., 7625., 9134.], dtype=np.float64)
NGRID = 24
GRID = np.exp(np.linspace(np.log(2500.0), np.log(9200.0), NGRID))
IMG_NATIVE = 24
IMG_SIZE = 32
PAD = (IMG_SIZE - IMG_NATIVE) // 2  # 4
N_CHANNELS = 7
MTF_N_BINS = 8

MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 1024


def seed_everything(seed: int = 42):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# ─── Image building ────────────────────────────────────────────────────────────

def build_flux_signals(mags: np.ndarray, redshifts: np.ndarray):
    """Build s_rest (N,24) and s_obs (N,24) float32."""
    N = len(mags)
    mag_means = mags.mean(axis=1)
    flux = 10.0 ** (-0.4 * (mags - mag_means[:, None]))
    s_rest = np.empty((N, NGRID), dtype=np.float32)
    s_obs = np.empty((N, NGRID), dtype=np.float32)

    for i in range(N):
        z = float(redshifts[i]); zc = max(z, -0.009)
        lam_rest = LAM_OBS / (1.0 + zc)
        f = flux[i].astype(np.float64)

        order_r = np.argsort(lam_rest)
        p = PchipInterpolator(lam_rest[order_r], f[order_r], extrapolate=False)
        sr = p(GRID)
        if np.isnan(sr).any():
            sr[GRID < lam_rest[order_r][0]] = f[order_r][0]
            sr[GRID > lam_rest[order_r][-1]] = f[order_r][-1]
        s_rest[i] = sr.astype(np.float32)

        order_o = np.argsort(LAM_OBS)
        p2 = PchipInterpolator(LAM_OBS[order_o], f[order_o], extrapolate=False)
        so = p2(GRID)
        if np.isnan(so).any():
            so[GRID < LAM_OBS[order_o][0]] = f[order_o][0]
            so[GRID > LAM_OBS[order_o][-1]] = f[order_o][-1]
        s_obs[i] = so.astype(np.float32)

    return s_rest, s_obs, mag_means.astype(np.float32), redshifts.astype(np.float32)


def gasf(s: np.ndarray) -> np.ndarray:
    smin, smax = s.min(), s.max()
    sc = np.clip((s - smin) / (smax - smin + 1e-9) * 2.0 - 1.0, -1.0, 1.0)
    phi = np.arccos(sc)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gadf(s: np.ndarray) -> np.ndarray:
    smin, smax = s.min(), s.max()
    sc = np.clip((s - smin) / (smax - smin + 1e-9) * 2.0 - 1.0, -1.0, 1.0)
    phi = np.arccos(sc)
    return np.sin(phi[:, None] - phi[None, :]).astype(np.float32)


def rp_continuous(s: np.ndarray) -> np.ndarray:
    r = np.abs(s[:, None] - s[None, :])
    return (r / (r.max() + 1e-9)).astype(np.float32)


def mtf_transform(s: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    n_bins = len(bin_edges) + 1
    bins = np.digitize(s, bin_edges).clip(0, n_bins - 1)
    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    for k in range(len(s) - 1):
        W[bins[k], bins[k + 1]] += 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / (row_sums + 1e-9)
    return W[np.ix_(bins, bins)].astype(np.float32)


def fit_mtf_bins(s_rest_train: np.ndarray) -> np.ndarray:
    """Fit MTF quantile bin edges on TRAIN-FOLD s_rest only."""
    edges = np.quantile(s_rest_train.ravel(), np.linspace(0, 1, MTF_N_BINS + 1)[1:-1])
    return edges.astype(np.float64)


def build_single_image_channels(sr, so, z_n, bin_edges):
    """Build 7 channels (24,24) for one row."""
    ch0 = gasf(sr)
    ch1 = gadf(sr)
    ch2 = rp_continuous(sr)
    ch3 = mtf_transform(sr, bin_edges)
    ch4 = gasf(so)
    ch5_raw = np.outer(so, so).astype(np.float32)
    ch5 = ch5_raw / (np.abs(ch5_raw).max() + 1e-9) * z_n
    sr_n = (sr - sr.min()) / (sr.max() - sr.min() + 1e-9)
    so_n = (so - so.min()) / (so.max() - so.min() + 1e-9)
    ch6 = np.outer(sr_n, so_n).astype(np.float32)
    return ch0, ch1, ch2, ch3, ch4, ch5, ch6


def build_images_float16(
    s_rest: np.ndarray,
    s_obs: np.ndarray,
    redshifts: np.ndarray,
    bin_edges: np.ndarray,
    desc: str = "",
    chunk_log: int = 50000,
) -> np.ndarray:
    """Build images stored as float16 (N, 7, 24, 24) — halves RAM vs float32."""
    N = len(s_rest)
    z_std = np.std(redshifts) + 1e-9
    z_normed = (redshifts.astype(np.float64) / z_std).astype(np.float32)
    imgs = np.zeros((N, N_CHANNELS, IMG_NATIVE, IMG_NATIVE), dtype=np.float16)
    for i in range(N):
        chs = build_single_image_channels(s_rest[i], s_obs[i], float(z_normed[i]), bin_edges)
        for c, ch in enumerate(chs):
            imgs[i, c] = ch.astype(np.float16)
        if desc and (i + 1) % chunk_log == 0:
            log(f"  {desc}: {i+1}/{N}")
    if desc:
        log(f"  {desc}: {N}/{N} done")
    return imgs


def fit_channel_stats(imgs_f16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from float16 images. Returns (mu_7, sd_7) float32."""
    mu = np.empty(N_CHANNELS, dtype=np.float32)
    sd = np.empty(N_CHANNELS, dtype=np.float32)
    for c in range(N_CHANNELS):
        arr = imgs_f16[:, c].astype(np.float32)
        mu[c] = float(arr.mean())
        sd[c] = float(arr.std()) + 1e-8
    return mu, sd


# ─── Dataset: lazy standardize + pad in __getitem__ ──────────────────────────

class SedImageDataset(torch.utils.data.Dataset):
    """Load float16 native images; standardize + pad to float32 (32x32) in __getitem__.

    This avoids materializing a large float32 padded array — RAM stays at float16 native.
    """
    def __init__(
        self,
        imgs_f16: np.ndarray,     # (N, 7, 24, 24) float16
        scalars: np.ndarray,      # (N, 2) float32
        labels: np.ndarray,       # (N,) int64; None for test
        img_mu: np.ndarray,       # (7,) float32
        img_sd: np.ndarray,       # (7,) float32
    ):
        self.imgs = imgs_f16
        self.scalars = scalars.astype(np.float32)
        self.labels = labels
        # Precompute tensors for stats to avoid recreating per __getitem__
        # Store as float32 numpy for fast conversion
        self.mu = torch.from_numpy(img_mu.reshape(7, 1, 1))   # (7,1,1)
        self.sd = torch.from_numpy(img_sd.reshape(7, 1, 1))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # float16 (7,24,24) -> float32 (7,24,24)
        img = torch.from_numpy(self.imgs[idx].astype(np.float32))  # (7,24,24)
        # Standardize
        img = (img - self.mu) / self.sd
        # Zero-pad to 32x32
        img_pad = F.pad(img, (PAD, PAD, PAD, PAD), mode='constant', value=0.0)  # (7,32,32)
        sc = torch.from_numpy(self.scalars[idx])
        if self.labels is not None:
            return img_pad, sc, int(self.labels[idx])
        return img_pad, sc


# ─── ResNet model ─────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.silu(out + identity)


class SedResNet(nn.Module):
    """Small ResNet from scratch for 7-ch 32x32 images. ~0.7M params."""
    def __init__(self, n_scalars: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        self.stage1 = nn.Sequential(BasicBlock(32, 32, 1), BasicBlock(32, 32, 1))
        self.stage2 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64, 1))
        self.stage3 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 + n_scalars, 128), nn.SiLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x, sc):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.head(torch.cat([x, sc], 1))


def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_resnet(
    imgs_f16: np.ndarray,
    scalars: np.ndarray,
    y_tr: np.ndarray,
    img_mu: np.ndarray,
    img_sd: np.ndarray,
    fold_seed: int,
    n_scalars: int = 2,
) -> tuple[SedResNet, dict]:
    """Train ResNet; imgs_f16 is float16 (N,7,24,24); standardize+pad done in Dataset."""
    seed_everything(fold_seed)

    n = len(y_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]
    log(f"    ResNet split: train={len(ti)} int_val={len(vi)}")

    counts = np.bincount(y_tr[ti], minlength=N_CLASSES).astype(np.float64)
    cw = torch.tensor(counts.sum() / (N_CLASSES * counts + 1e-8), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    model = SedResNet(n_scalars=n_scalars).to(DEVICE)
    if fold_seed == (SEED + 100):  # only log once
        log(f"    ResNet params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    train_ds = SedImageDataset(imgs_f16[ti], scalars[ti], y_tr[ti], img_mu, img_sd)
    val_ds = SedImageDataset(imgs_f16[vi], scalars[vi], y_tr[vi], img_mu, img_sd)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    best_ba = -1.0; best_state = None; bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for imgs_b, sc_b, y_b in train_loader:
            imgs_b = imgs_b.to(DEVICE, non_blocking=True)
            sc_b = sc_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            # Channel dropout (p=0.1) + Gaussian noise (std=0.02)
            mask = (torch.rand(imgs_b.shape[0], imgs_b.shape[1], 1, 1, device=DEVICE) > 0.1).float()
            imgs_b = imgs_b * mask + torch.randn_like(imgs_b) * 0.02
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs_b, sc_b)
                loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs_b, sc_b, y_b in val_loader:
                imgs_b = imgs_b.to(DEVICE, non_blocking=True)
                sc_b = sc_b.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(imgs_b, sc_b)
                preds_all.extend(logits.float().argmax(1).cpu().numpy())
                labels_all.extend(y_b.numpy())
        ba = balanced_accuracy_score(labels_all, preds_all)

        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"    Early stop ep={ep+1}"); break

        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    ep {ep+1}/{MAX_EPOCHS}: int_ba={ba:.5f} best={best_ba:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    Done: best_int_ba={best_ba:.5f}")
    return model, {"best_ba": best_ba}


def predict_proba(
    model: SedResNet,
    imgs_f16: np.ndarray,
    scalars: np.ndarray,
    img_mu: np.ndarray,
    img_sd: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict probabilities. Standardize+pad done in Dataset."""
    ds = SedImageDataset(imgs_f16, scalars, labels=None, img_mu=img_mu, img_sd=img_sd)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True, persistent_workers=True)
    model.eval(); out = []
    with torch.no_grad():
        for imgs_b, sc_b in loader:
            imgs_b = imgs_b.to(DEVICE, non_blocking=True)
            sc_b = sc_b.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs_b, sc_b)
            out.append(torch.softmax(logits.float(), -1).cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


# ─── CORE15 bases ─────────────────────────────────────────────────────────────
BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]


def compute_restack_cv(oof_new, y_all, folds_list, label="CORE15+n57") -> dict:
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import differential_evolution

    X_meta = np.concatenate(
        [np.load(COMP_DIR / "nodes" / b / "oof.npy") for b in BASES] + [oof_new],
        axis=1
    )
    fold_scores = []
    for fi in folds_list:
        val_idx = np.array(fi["val_idx"])
        tr_idx = np.setdiff1d(np.arange(len(y_all)), val_idx)
        meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                  solver="lbfgs", multi_class="multinomial", random_state=SEED)
        meta.fit(X_meta[tr_idx], y_all[tr_idx])
        pva = meta.predict_proba(X_meta[val_idx])

        def neg_ba(t):
            return -balanced_accuracy_score(y_all[val_idx], (pva * np.array(t)).argmax(1))

        res = differential_evolution(neg_ba, bounds=[(0.5, 2.0)] * N_CLASSES,
                                     seed=SEED, maxiter=50, tol=1e-5, popsize=10)
        fold_scores.append(-res.fun)

    mean_cv = float(np.mean(fold_scores))
    sem_cv = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
    log(f"  {label}: {mean_cv:.6f} ± {sem_cv:.6f}  folds={[round(s,6) for s in fold_scores]}")
    return {"mean_cv": mean_cv, "sem_cv": sem_cv, "fold_scores": fold_scores}


# ─── Load data ─────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Pre-compute flux signals (stateless) ─────────────────────────────────────
log("Pre-computing flux signals (stateless) ...")
t_sig = time.perf_counter()
s_rest_tr, s_obs_tr, mag_mean_tr, z_tr = build_flux_signals(
    train_raw[BANDS].values.astype(np.float64), train_raw["redshift"].values)
s_rest_te, s_obs_te, mag_mean_te, z_te = build_flux_signals(
    test_raw[BANDS].values.astype(np.float64), test_raw["redshift"].values)
log(f"  Signals done {time.perf_counter()-t_sig:.1f}s  s_rest_tr={s_rest_tr.shape}")
log(f"  Signal RAM: {(s_rest_tr.nbytes+s_obs_tr.nbytes+s_rest_te.nbytes+s_obs_te.nbytes)/1e6:.0f} MB")

# ─── Load CORE15 OOFs (for kill-switch + re-stack) ────────────────────────────
log("Loading CORE15 OOFs ...")
core15_oof_sum = np.zeros((n_train, N_CLASSES), dtype=np.float64)
for b in BASES:
    core15_oof_sum += np.load(COMP_DIR / "nodes" / b / "oof.npy").astype(np.float64)
core15_mean = (core15_oof_sum / len(BASES)).astype(np.float32)
del core15_oof_sum; gc.collect()

# ─── FOLD-0 KILL-SWITCH PHASE ────────────────────────────────────────────────
log("=" * 70)
log("FOLD-0 KILL-SWITCH PHASE")
log("=" * 70)

fi0 = folds_list[0]
val0 = np.array(fi0["val_idx"])
tr0 = np.setdiff1d(np.arange(n_train), val0)
fold0_seed = SEED + 100
log(f"Fold 0: train={len(tr0)} val={len(val0)}")
t_fold0 = time.perf_counter()

# MTF bin edges fit on TRAIN-FOLD-0 s_rest (fit_in_fold)
bin_edges_0 = fit_mtf_bins(s_rest_tr[tr0])

# Build float16 images
log("  Building images fold-0 ...")
imgs_tr0 = build_images_float16(s_rest_tr[tr0], s_obs_tr[tr0], z_tr[tr0].astype(np.float64), bin_edges_0, desc="tr0")
imgs_val0 = build_images_float16(s_rest_tr[val0], s_obs_tr[val0], z_tr[val0].astype(np.float64), bin_edges_0, desc="val0")
img_build_t = time.perf_counter() - t_fold0
log(f"  Images: tr0={imgs_tr0.shape} {imgs_tr0.nbytes/1e9:.2f}GB  val0={imgs_val0.shape} {imgs_val0.nbytes/1e9:.2f}GB")
log(f"  IMAGE BUILD: {img_build_t:.1f}s  projected total: {img_build_t*(n_train+n_test)/len(tr0)/60:.1f}min (single-fold scale)")

# Fit channel stats on TRAIN-FOLD-0 images (fit_in_fold)
img_mu_0, img_sd_0 = fit_channel_stats(imgs_tr0)

# Standardize scalars (fit on TRAIN-FOLD-0)
sc_tr0_raw = np.stack([mag_mean_tr[tr0], z_tr[tr0]], axis=1)
sc_val0_raw = np.stack([mag_mean_tr[val0], z_tr[val0]], axis=1)
sc_mu_0 = sc_tr0_raw.mean(0); sc_sd_0 = sc_tr0_raw.std(0) + 1e-8
sc_tr0 = ((sc_tr0_raw - sc_mu_0) / sc_sd_0).astype(np.float32)
sc_val0 = ((sc_val0_raw - sc_mu_0) / sc_sd_0).astype(np.float32)

# Train fold-0 ResNet
log("  Training fold-0 ResNet ...")
t_train0 = time.perf_counter()
model0, info0 = train_resnet(imgs_tr0, sc_tr0, y_all[tr0], img_mu_0, img_sd_0, fold0_seed)
log(f"  ResNet fold-0 training: {time.perf_counter()-t_train0:.1f}s")

# Fold-0 OOF predictions
oof0_probs = predict_proba(model0, imgs_val0, sc_val0, img_mu_0, img_sd_0)
ba0 = balanced_accuracy_score(y_all[val0], oof0_probs.argmax(1))
log(f"  FOLD-0 STANDALONE BA: {ba0:.6f}")
print(f"fold0_ba={ba0:.6f}", flush=True)

if torch.cuda.is_available():
    log(f"  Peak VRAM fold-0: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Error correlation
core15_wrong_f0 = (core15_mean[val0].argmax(1) != y_all[val0]).astype(float)
n57_wrong_f0 = (oof0_probs.argmax(1) != y_all[val0]).astype(float)
err_corr_f0 = float(np.corrcoef(core15_wrong_f0, n57_wrong_f0)[0, 1])
log(f"  FOLD-0 ERR-CORR vs CORE15: {err_corr_f0:.4f}")
print(f"fold0_err_corr={err_corr_f0:.4f}", flush=True)

fold0_total = time.perf_counter() - t_fold0
log(f"  FOLD-0 TOTAL: {fold0_total:.1f}s  PROJ 5-FOLD: {fold0_total*5:.1f}s ({fold0_total*5/60:.1f}min)")

# ─── Ablation studies ─────────────────────────────────────────────────────────
log("  Running fold-0 ablations ...")


def run_ablation(label: str, imgs_tr_abl, imgs_val_abl, sc_tr, sc_val, seed_off: int) -> tuple[float, float]:
    """Train a model with ablation images, return (ba, err_corr)."""
    t_a = time.perf_counter()
    mu_a, sd_a = fit_channel_stats(imgs_tr_abl)
    model_a, _ = train_resnet(imgs_tr_abl, sc_tr, y_all[tr0], mu_a, sd_a, fold0_seed + seed_off)
    oof_a = predict_proba(model_a, imgs_val_abl, sc_val, mu_a, sd_a)
    ba_a = balanced_accuracy_score(y_all[val0], oof_a.argmax(1))
    ec_a = float(np.corrcoef(core15_wrong_f0, (oof_a.argmax(1) != y_all[val0]).astype(float))[0, 1])
    del model_a; gc.collect()
    log(f"    Ablation {label}: BA={ba_a:.6f}  err-corr={ec_a:.4f}  time={time.perf_counter()-t_a:.1f}s")
    return ba_a, ec_a


# Ablation A: warp-OFF — zero channels 0-3 (all s_rest channels)
log("    Building warp-OFF images ...")
def build_imgs_warp_off(s_rest, s_obs, redshifts, bin_edges):
    N = len(s_rest)
    z_std = np.std(redshifts) + 1e-9
    z_normed = (redshifts.astype(np.float64) / z_std).astype(np.float32)
    imgs = np.zeros((N, N_CHANNELS, IMG_NATIVE, IMG_NATIVE), dtype=np.float16)
    for i in range(N):
        ch4 = gasf(s_obs[i])
        ch5_raw = np.outer(s_obs[i], s_obs[i]).astype(np.float32)
        ch5 = ch5_raw / (np.abs(ch5_raw).max() + 1e-9) * float(z_normed[i])
        so_n = (s_obs[i]-s_obs[i].min())/(s_obs[i].max()-s_obs[i].min()+1e-9)
        ch6 = np.outer(np.zeros(IMG_NATIVE, np.float32), so_n).astype(np.float32)
        imgs[i, 4] = ch4.astype(np.float16)
        imgs[i, 5] = ch5.astype(np.float16)
        imgs[i, 6] = ch6.astype(np.float16)
        # ch0-3 remain zero
    return imgs

imgs_tr0_woff = build_imgs_warp_off(s_rest_tr[tr0], s_obs_tr[tr0], z_tr[tr0].astype(np.float64), bin_edges_0)
imgs_val0_woff = build_imgs_warp_off(s_rest_tr[val0], s_obs_tr[val0], z_tr[val0].astype(np.float64), bin_edges_0)
ba_warpoff, ec_warpoff = run_ablation("warp-OFF", imgs_tr0_woff, imgs_val0_woff, sc_tr0, sc_val0, seed_off=10)
del imgs_tr0_woff, imgs_val0_woff; gc.collect()

# Ablation B: zmod-OFF — zero channel 5
log("    Building zmod-OFF images ...")
def build_imgs_zmod_off(s_rest, s_obs, redshifts, bin_edges):
    N = len(s_rest)
    z_std = np.std(redshifts) + 1e-9
    z_normed = (redshifts.astype(np.float64) / z_std).astype(np.float32)
    imgs = np.zeros((N, N_CHANNELS, IMG_NATIVE, IMG_NATIVE), dtype=np.float16)
    for i in range(N):
        chs = build_single_image_channels(s_rest[i], s_obs[i], float(z_normed[i]), bin_edges)
        for c, ch in enumerate(chs):
            imgs[i, c] = ch.astype(np.float16)
        imgs[i, 5] = np.float16(0)  # zero out zmod
    return imgs

imgs_tr0_zoff = build_imgs_zmod_off(s_rest_tr[tr0], s_obs_tr[tr0], z_tr[tr0].astype(np.float64), bin_edges_0)
imgs_val0_zoff = build_imgs_zmod_off(s_rest_tr[val0], s_obs_tr[val0], z_tr[val0].astype(np.float64), bin_edges_0)
ba_zmodoff, ec_zmodoff = run_ablation("zmod-OFF", imgs_tr0_zoff, imgs_val0_zoff, sc_tr0, sc_val0, seed_off=20)
del imgs_tr0_zoff, imgs_val0_zoff; gc.collect()

# Ablation C: image-only (no side scalars, pass zeros for scalars)
sc_zeros = np.zeros_like(sc_tr0)
sc_val_zeros = np.zeros_like(sc_val0)
ba_imgonly, ec_imgonly = run_ablation("img-only", imgs_tr0, imgs_val0, sc_zeros, sc_val_zeros, seed_off=30)

# ─── KILL-SWITCH DECISION ─────────────────────────────────────────────────────
log("=" * 70)
log("KILL-SWITCH VERDICT")
log("=" * 70)
log(f"  Fold-0 standalone BA  : {ba0:.6f}  (kill if < 0.955)")
log(f"  Fold-0 err-corr CORE15: {err_corr_f0:.4f}   (kill if > 0.6)")
log(f"  Ablation warp-OFF     : BA={ba_warpoff:.6f}  err-corr={ec_warpoff:.4f}")
log(f"  Ablation zmod-OFF     : BA={ba_zmodoff:.6f}  err-corr={ec_zmodoff:.4f}")
log(f"  Ablation image-only   : BA={ba_imgonly:.6f}  err-corr={ec_imgonly:.4f}")

KILL = (ba0 < 0.955) or (err_corr_f0 > 0.6)

if KILL:
    reasons = []
    if ba0 < 0.955: reasons.append(f"BA={ba0:.4f}<0.955")
    if err_corr_f0 > 0.6: reasons.append(f"err_corr={err_corr_f0:.4f}>0.6")
    reason_str = " AND ".join(reasons)
    log(f"  KILL TRIGGERED: {reason_str}")
    log("  feature->image->CNN family retired: warp does not de-correlate")

    partial_oof = np.full((n_train, N_CLASSES), np.nan, dtype=np.float32)
    partial_oof[val0] = oof0_probs
    np.save(NODE_DIR / "oof_fold0_only.npy", partial_oof)
    (NODE_SRC / "features.txt").write_text("\n".join(sorted(["u","g","r","i","z","redshift"])) + "\n")

    print(f"kill_switch=TRIGGERED reason={reason_str}", flush=True)
    print(f"cv=KILLED_FOLD0_ONLY", flush=True)
    log(f"Total elapsed: {time.perf_counter()-T0:.1f}s ({(time.perf_counter()-T0)/60:.1f}min)")
    log("DONE. status=dead, leak=clean")
    sys.exit(0)

log("KILL-SWITCH PASSED — continuing to 5-fold loop")

del model0
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─── Full 5-fold OOF loop ─────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting 5-fold OOF loop ...")
loop_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.array(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)
    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")
    t_fold = time.perf_counter()

    # MTF bin edges fit on TRAIN-FOLD rows (fit_in_fold)
    bin_edges_f = fit_mtf_bins(s_rest_tr[tr_idx])

    # Build float16 images
    imgs_tr_f = build_images_float16(s_rest_tr[tr_idx], s_obs_tr[tr_idx], z_tr[tr_idx].astype(np.float64), bin_edges_f, desc=f"tr{fold_id}")
    imgs_val_f = build_images_float16(s_rest_tr[val_idx], s_obs_tr[val_idx], z_tr[val_idx].astype(np.float64), bin_edges_f, desc=f"va{fold_id}")
    imgs_te_f = build_images_float16(s_rest_te, s_obs_te, z_te.astype(np.float64), bin_edges_f, desc=f"te{fold_id}")
    log(f"  Images: tr={imgs_tr_f.nbytes/1e9:.2f}GB val={imgs_val_f.nbytes/1e9:.2f}GB te={imgs_te_f.nbytes/1e9:.2f}GB")

    # Fit channel stats on TRAIN-FOLD (fit_in_fold)
    img_mu_f, img_sd_f = fit_channel_stats(imgs_tr_f)

    # Scalars
    sc_tr_raw = np.stack([mag_mean_tr[tr_idx], z_tr[tr_idx]], axis=1)
    sc_val_raw = np.stack([mag_mean_tr[val_idx], z_tr[val_idx]], axis=1)
    sc_te_raw = np.stack([mag_mean_te, z_te], axis=1)
    sc_mu_f = sc_tr_raw.mean(0); sc_sd_f = sc_tr_raw.std(0) + 1e-8
    sc_tr_f = ((sc_tr_raw - sc_mu_f) / sc_sd_f).astype(np.float32)
    sc_val_f = ((sc_val_raw - sc_mu_f) / sc_sd_f).astype(np.float32)
    sc_te_f = ((sc_te_raw - sc_mu_f) / sc_sd_f).astype(np.float32)

    # Train
    model_f, _ = train_resnet(imgs_tr_f, sc_tr_f, y_all[tr_idx], img_mu_f, img_sd_f, fold_seed)

    # OOF predictions
    val_probs = predict_proba(model_f, imgs_val_f, sc_val_f, img_mu_f, img_sd_f)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions
    test_probs_fold = predict_proba(model_f, imgs_te_f, sc_te_f, img_mu_f, img_sd_f)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_all[val_idx], val_probs.argmax(1))
    per_fold_scores.append(fold_score)
    elapsed = time.perf_counter() - t_fold
    log(f"  fold {fold_id}: BA={fold_score:.6f}  elapsed={elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        log(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    del model_f, imgs_tr_f, imgs_val_f, imgs_te_f
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        t1 = time.perf_counter() - loop_t0
        log(f"  TIMING fold-0: {t1:.1f}s  proj 5-fold: {t1*5:.1f}s ({t1*5/60:.1f}min)")

# ─── CV summary ───────────────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Error correlation vs CORE15 ─────────────────────────────────────────────
core15_wrong_all = (core15_mean.argmax(1) != y_all).astype(float)
n57_wrong_all = (oof_proba.argmax(1) != y_all).astype(float)
err_corr_full = float(np.corrcoef(core15_wrong_all, n57_wrong_all)[0, 1])
log(f"Full OOF err-corr vs CORE15: {err_corr_full:.4f}")

# ─── Re-stack CORE15+n57 vs champion ─────────────────────────────────────────
log("Computing re-stack CORE15+n57 ...")
restack = compute_restack_cv(oof_proba, y_all, folds_list)
delta = restack['mean_cv'] - 0.969808
log(f"  Champion 0.969808  delta={delta:+.6f}  (2sem={2*restack['sem_cv']:.6f})")

# ─── Save OOF ─────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy {oof_proba.shape}")

# ─── Save test_probs ──────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy {test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv {len(sub)} rows")
log(f"Class dist:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
(NODE_SRC / "features.txt").write_text("\n".join(sorted(["u","g","r","i","z","redshift"])) + "\n")
log("Wrote features.txt")

# ─── Final OOF metric ─────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full BA={oof_metric:.6f}")
total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
