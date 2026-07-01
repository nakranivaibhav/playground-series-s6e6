"""node_0058 — YOLO26n-cls + MuSGD augmentation ablation (fold-0 kill-switch).

THE ONE ATOMIC CHANGE (vs node_0057):
  Swap the ResNet-from-scratch trainer with YOLO26n-cls pretrained backbone +
  MuSGD optimizer (ultralytics 8.4+). Only 2 epochs (Muon converges fast).

INPUT: 3-channel RGB PNGs written from rest-frame SED:
  R = (GASF + 1) / 2,  G = (GADF + 1) / 2,  B = RP  (all in [0,1] → uint8)
  Written as 24x24 uint8 PNGs; ultralytics upscales to imgsz=64.

AUGMENTATION CONFIGS (fold-0 ablation, then kill-switch):
  C0 = none  (new reference, measures backbone/optimizer vs node_0057 0.9401)
  C1 = Tier A  (input-space: mag/z jitter, band-dropout → 1 pre-rendered copy)
  C2 = Tier B  (image regularizers: erasing + pixel noise)
  C3 = Tier C  (geometric: hflip, vflip, rotate, shear, scale)
  C4 = A + B
  C5 = A + B + C

KILL-SWITCH: proceed to 5-fold only if some config clears
  fold-0 BA >= 0.955  AND  err-corr vs CORE15 <= 0.6;
  otherwise record results and exit as dead.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET   = "class"
IDC      = "id"
SEED     = 42
N_CLASSES = 3
CLASSES  = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log(f"Device: {DEVICE}  GPU={torch.cuda.get_device_name(0)}")
else:
    log(f"Device: {DEVICE}")

import torch as _torch
log(f"torch={_torch.__version__}  cuda={_torch.cuda.is_available()}")

# ─── Image / backbone constants ───────────────────────────────────────────────
BANDS   = ["u", "g", "r", "i", "z"]
LAM_OBS = np.array([3543., 4770., 6231., 7625., 9134.], dtype=np.float64)
NGRID   = 24
GRID    = np.exp(np.linspace(np.log(2500.0), np.log(9200.0), NGRID))
IMGSZ   = 64

BACKBONE  = "yolo26n-cls.pt"
OPTIMIZER = "MuSGD"
N_EPOCHS  = 2

BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]


def seed_everything(seed: int = 42):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ─── Rest-frame signal building ───────────────────────────────────────────────

def build_s_rest(mags: np.ndarray, redshifts: np.ndarray) -> np.ndarray:
    """Build s_rest (N, NGRID) float32. Stateless row-wise PCHIP."""
    N = len(mags)
    mag_means = mags.mean(axis=1)
    flux = 10.0 ** (-0.4 * (mags - mag_means[:, None]))
    s_rest = np.empty((N, NGRID), dtype=np.float32)
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
    return s_rest


def build_s_rest_tier_a(
    mags: np.ndarray,
    redshifts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build s_rest with Tier-A input-space augmentation (TRAIN only).

    Augmentations applied BEFORE PCHIP / GAF:
      1. Per-band mag jitter N(0, 0.03)
      2. Flux-zeropoint (common offset) jitter N(0, 0.03)
      3. Random band-dropout p=0.1 → linear interp from neighbors
      4. Redshift jitter N(0, max(0.002, 0.005*(1+|z|)))
    """
    N = len(mags)
    m = mags.copy().astype(np.float64)
    # 1. per-band jitter
    m += rng.normal(0.0, 0.03, size=m.shape)
    # 2. zeropoint jitter (common offset per row)
    m += rng.normal(0.0, 0.03, size=(N, 1))
    # 3. band-dropout
    drop_mask = rng.random(N) < 0.1
    if drop_mask.any():
        bands_drop = rng.integers(0, 5, size=int(drop_mask.sum()))
        rows_drop  = np.where(drop_mask)[0]
        for row, band in zip(rows_drop, bands_drop):
            if band == 0:
                m[row, 0] = m[row, 1]
            elif band == 4:
                m[row, 4] = m[row, 3]
            else:
                m[row, band] = 0.5 * (m[row, band - 1] + m[row, band + 1])
    # 4. redshift jitter
    z_std = np.maximum(0.002, 0.005 * (1.0 + np.abs(redshifts)))
    z_aug = redshifts.astype(np.float64) + rng.normal(0.0, 1.0, size=N) * z_std

    mag_means = m.mean(axis=1)
    flux = 10.0 ** (-0.4 * (m - mag_means[:, None]))
    s_rest = np.empty((N, NGRID), dtype=np.float32)
    for i in range(N):
        z = float(z_aug[i]); zc = max(z, -0.009)
        lam_rest = LAM_OBS / (1.0 + zc)
        f = flux[i]
        order_r = np.argsort(lam_rest)
        p = PchipInterpolator(lam_rest[order_r], f[order_r], extrapolate=False)
        sr = p(GRID)
        if np.isnan(sr).any():
            sr[GRID < lam_rest[order_r][0]] = f[order_r][0]
            sr[GRID > lam_rest[order_r][-1]] = f[order_r][-1]
        s_rest[i] = sr.astype(np.float32)
    return s_rest


# ─── Vectorized GAF/RP ────────────────────────────────────────────────────────

def gasf_batch(S: np.ndarray) -> np.ndarray:
    """(N,L) → (N,L,L) GASF float32."""
    smin = S.min(axis=1, keepdims=True)
    smax = S.max(axis=1, keepdims=True)
    sc = np.clip((S - smin) / (smax - smin + 1e-9) * 2.0 - 1.0, -1.0, 1.0)
    phi = np.arccos(sc)
    return np.cos(phi[:, :, None] + phi[:, None, :]).astype(np.float32)


def gadf_batch(S: np.ndarray) -> np.ndarray:
    """(N,L) → (N,L,L) GADF float32."""
    smin = S.min(axis=1, keepdims=True)
    smax = S.max(axis=1, keepdims=True)
    sc = np.clip((S - smin) / (smax - smin + 1e-9) * 2.0 - 1.0, -1.0, 1.0)
    phi = np.arccos(sc)
    return np.sin(phi[:, :, None] - phi[:, None, :]).astype(np.float32)


def rp_batch(S: np.ndarray) -> np.ndarray:
    """(N,L) → (N,L,L) RP float32 in [0,1]."""
    diff = np.abs(S[:, :, None] - S[:, None, :])
    d_max = diff.max(axis=(1, 2), keepdims=True)
    return (diff / (d_max + 1e-9)).astype(np.float32)


# ─── PNG writing helpers ──────────────────────────────────────────────────────

def s_rest_to_rgb_u8(s_rest: np.ndarray) -> np.ndarray:
    """(N, 24) → (N, 24, 24, 3) uint8  [R=GASF, G=GADF, B=RP]."""
    gasf_c = gasf_batch(s_rest)
    gadf_c = gadf_batch(s_rest)
    rp_c   = rp_batch(s_rest)
    R = ((gasf_c + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    G = ((gadf_c + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    B = (rp_c * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)  # (N, 24, 24, 3)


def write_pngs(
    s_rest: np.ndarray,        # (N, 24)
    labels: np.ndarray | None, # (N,) int label indices or None for test
    out_dir: Path,             # e.g. png_root/train
    idx_offset: int = 0,       # filename offset for extra copies
    chunk_size: int = 5000,
    desc: str = "",
) -> None:
    """Write uint8 PNGs into out_dir/{class_name}/{idx_offset+i}.png."""
    from PIL import Image

    N = len(s_rest)
    is_test = labels is None
    if is_test:
        (out_dir / "_test_").mkdir(parents=True, exist_ok=True)
    else:
        for cls in CLASSES:
            (out_dir / cls).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_chunks = (N + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        sl_s = ci * chunk_size
        sl_e = min(sl_s + chunk_size, N)
        rgb = s_rest_to_rgb_u8(s_rest[sl_s:sl_e])  # (B, 24, 24, 3)
        for bi in range(sl_e - sl_s):
            global_i = sl_s + bi
            img = Image.fromarray(rgb[bi], mode="RGB")
            if is_test:
                img.save(out_dir / "_test_" / f"{idx_offset + global_i}.png")
            else:
                cls_name = CLASSES[int(labels[global_i])]
                img.save(out_dir / cls_name / f"{idx_offset + global_i}.png")

        if desc and (ci + 1) % max(1, n_chunks // 4) == 0:
            pct = (ci + 1) / n_chunks * 100
            log(f"  {desc}: {pct:.0f}% ({sl_e}/{N})  {time.perf_counter()-t0:.1f}s")

    if desc:
        log(f"  {desc}: 100%  {time.perf_counter()-t0:.1f}s")


def build_png_dataset(
    s_rest_train: np.ndarray,
    y_train: np.ndarray,
    s_rest_val: np.ndarray,
    y_val: np.ndarray,
    out_root: Path,
    s_rest_train_extra: np.ndarray | None = None,
    y_train_extra: np.ndarray | None = None,
    desc: str = "",
) -> Path:
    """Write train/val PNGs in ImageFolder format under out_root.

    Optional extra train PNGs (for Tier-A jittered copy) written with offset.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    write_pngs(s_rest_train, y_train, out_root / "train",
               idx_offset=0, desc=f"train_{desc}")
    write_pngs(s_rest_val,   y_val,   out_root / "val",
               idx_offset=0, desc=f"val_{desc}")

    if s_rest_train_extra is not None:
        write_pngs(s_rest_train_extra, y_train_extra, out_root / "train",
                   idx_offset=len(s_rest_train), desc=f"extra_{desc}")

    return out_root


# ─── YOLO training ────────────────────────────────────────────────────────────

YOLO_PROJECT = "/tmp/yolo_n0058"


def _train_yolo(
    data_dir: Path,
    run_name: str,
    epochs: int,
    seed: int,
    erasing: float = 0.0,
    fliplr: float = 0.0,
    flipud: float = 0.0,
    degrees: float = 0.0,
    shear: float = 0.0,
    scale: float = 0.0,
    hsv_h: float = 0.0,
    hsv_s: float = 0.0,
    hsv_v: float = 0.0,
    pixel_noise_std: float = 0.0,
    batch: int = 256,
):
    """Return the trained YOLO model (best weights loaded)."""
    from ultralytics import YOLO

    seed_everything(seed)
    model = YOLO(BACKBONE)

    if pixel_noise_std > 0:
        def on_train_batch_start(trainer):
            batch_data = getattr(trainer, "batch", None)
            if batch_data is not None:
                imgs = batch_data.get("img")
                if imgs is not None:
                    noise = torch.randn_like(imgs) * pixel_noise_std
                    trainer.batch["img"] = (imgs + noise).clamp(0.0, 1.0)
        model.add_callback("on_train_batch_start", on_train_batch_start)

    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=IMGSZ,
        optimizer=OPTIMIZER,
        batch=batch,
        device="0" if torch.cuda.is_available() else "cpu",
        workers=4,
        project=YOLO_PROJECT,
        name=run_name,
        exist_ok=True,
        verbose=True,   # keep verbose so we can see epoch BA in log
        # Augmentations
        fliplr=fliplr,
        flipud=flipud,
        degrees=degrees,
        shear=shear,
        scale=scale,
        erasing=erasing,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        mosaic=0.0,
        close_mosaic=0,
        auto_augment=None,
        # Other
        seed=seed,
        save_period=-1,
        warmup_epochs=0,
        label_smoothing=0.05,
        patience=0,        # no early stopping (we only run 2 epochs)
    )

    # Load best weights
    best_pt = Path(YOLO_PROJECT) / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(YOLO_PROJECT) / run_name / "weights" / "last.pt"
    log(f"  Loading weights: {best_pt}")
    return YOLO(str(best_pt))


def predict_on_png_dir(model, val_dir: Path, y_true: np.ndarray, batch: int = 512):
    """Predict on val_dir/{class_name}/*.png; returns (probs N×3, BA, sorted y_true).

    Files sorted by integer stem within each class folder;
    class order follows CLASSES list so label assignment is consistent.
    """
    all_paths, all_labels = [], []
    for cls_name in CLASSES:
        cls_dir = val_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir(), key=lambda x: int(x.stem)):
            all_paths.append(str(p))
            all_labels.append(LABEL_MAP[cls_name])

    if not all_paths:
        raise RuntimeError(f"No PNGs in {val_dir}")

    all_probs = []
    for i in range(0, len(all_paths), batch):
        results = model.predict(all_paths[i:i+batch], verbose=False, imgsz=IMGSZ)
        for r in results:
            all_probs.append(r.probs.data.cpu().float().numpy())

    probs = np.stack(all_probs, axis=0)        # (N, 3)
    labels = np.array(all_labels, dtype=np.int64)
    ba = balanced_accuracy_score(labels, probs.argmax(1))
    return probs, ba, labels


# ─── Ablation runner ──────────────────────────────────────────────────────────

def run_ablation(
    config_name: str,
    data_dir: Path,
    core15_wrong_val: np.ndarray,
    val_true: np.ndarray,
    seed: int = SEED,
    erasing: float = 0.0,
    fliplr: float = 0.0,
    flipud: float = 0.0,
    degrees: float = 0.0,
    shear: float = 0.0,
    scale: float = 0.0,
    pixel_noise_std: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    log(f"\n{'='*60}")
    log(f"Config {config_name}: erasing={erasing} fliplr={fliplr} flipud={flipud} "
        f"degrees={degrees} shear={shear} scale={scale} pnoise={pixel_noise_std}")
    t0 = time.perf_counter()

    model = _train_yolo(
        data_dir=data_dir,
        run_name=config_name,
        epochs=N_EPOCHS,
        seed=seed,
        erasing=erasing,
        fliplr=fliplr,
        flipud=flipud,
        degrees=degrees,
        shear=shear,
        scale=scale,
        pixel_noise_std=pixel_noise_std,
    )

    val_probs, ba, labels = predict_on_png_dir(model, data_dir / "val", val_true)
    wrong = (val_probs.argmax(1) != labels).astype(float)
    err_corr = float(np.corrcoef(core15_wrong_val, wrong)[0, 1])

    elapsed = time.perf_counter() - t0
    log(f"  {config_name}: BA={ba:.6f}  err_corr={err_corr:.4f}  time={elapsed:.1f}s")
    print(f"config_{config_name}_BA={ba:.6f}  err_corr={err_corr:.4f}", flush=True)

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ba, err_corr, val_probs


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw  = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw   = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# ─── Pre-compute rest-frame signals (stateless) ───────────────────────────────
log("Pre-computing rest-frame flux signals (stateless) ...")
t_sig = time.perf_counter()
s_rest_tr = build_s_rest(
    train_raw[BANDS].values.astype(np.float64),
    train_raw["redshift"].values,
)
s_rest_te = build_s_rest(
    test_raw[BANDS].values.astype(np.float64),
    test_raw["redshift"].values,
)
log(f"  Done {time.perf_counter()-t_sig:.1f}s  s_rest_tr={s_rest_tr.shape}")

# ─── Load CORE15 OOFs ─────────────────────────────────────────────────────────
log("Loading CORE15 OOFs ...")
core15_oof_sum = np.zeros((n_train, N_CLASSES), dtype=np.float64)
for b in BASES:
    core15_oof_sum += np.load(COMP_DIR / "nodes" / b / "oof.npy").astype(np.float64)
core15_mean = (core15_oof_sum / len(BASES)).astype(np.float32)
del core15_oof_sum; gc.collect()

# ─── Fold-0 setup ─────────────────────────────────────────────────────────────
log("=" * 70)
log("FOLD-0 ABLATION TOURNAMENT")
log("=" * 70)

fi0  = folds_list[0]
val0 = np.array(fi0["val_idx"])
tr0  = np.setdiff1d(np.arange(n_train), val0)
log(f"Fold 0: train={len(tr0)} val={len(val0)}")

core15_wrong_f0 = (core15_mean[val0].argmax(1) != y_all[val0]).astype(float)

# ─── TIMING PROBE ─────────────────────────────────────────────────────────────
PNG_TMP = Path("/tmp/s6e6_n0058_pngs")
shutil.rmtree(PNG_TMP, ignore_errors=True)
PNG_TMP.mkdir(parents=True, exist_ok=True)

PROBE_N = 2000
log(f"\nTIMING PROBE: rendering {PROBE_N} PNGs ...")
t_probe = time.perf_counter()
probe_dir = PNG_TMP / "probe"
write_pngs(s_rest_tr[tr0[:PROBE_N]], y_all[tr0[:PROBE_N]],
           probe_dir / "train", chunk_size=2000, desc="probe")
probe_t = time.perf_counter() - t_probe
rate = PROBE_N / probe_t
log(f"  {PROBE_N} PNGs in {probe_t:.2f}s = {rate:.0f} PNGs/s")
est_fold0_render = (len(tr0) + len(val0)) / rate
log(f"  Projected fold-0 render: {est_fold0_render:.0f}s ({est_fold0_render/60:.1f}min)")
if est_fold0_render > 1800:
    log("  WARNING: render > 30min. This is a bottleneck.")
shutil.rmtree(probe_dir, ignore_errors=True)

# ─── Build base fold-0 PNG dataset ────────────────────────────────────────────
log("\nBuilding base fold-0 PNG dataset ...")
t_base = time.perf_counter()
BASE_DIR = PNG_TMP / "fold0_base"
build_png_dataset(
    s_rest_tr[tr0], y_all[tr0],
    s_rest_tr[val0], y_all[val0],
    BASE_DIR, desc="base",
)
log(f"  Base done: {time.perf_counter()-t_base:.1f}s")

# ─── Build Tier-A fold-0 PNG dataset (original + 1 jittered copy) ─────────────
log("\nBuilding Tier-A fold-0 PNG dataset (1 jittered copy) ...")
t_ta = time.perf_counter()
rng_ta = np.random.default_rng(SEED + 200)
s_rest_tr0_jitter = build_s_rest_tier_a(
    train_raw[BANDS].values[tr0].astype(np.float64),
    train_raw["redshift"].values[tr0].astype(np.float64),
    rng_ta,
)
TIER_A_DIR = PNG_TMP / "fold0_tier_a"
build_png_dataset(
    s_rest_tr[tr0], y_all[tr0],
    s_rest_tr[val0], y_all[val0],
    TIER_A_DIR,
    s_rest_train_extra=s_rest_tr0_jitter,
    y_train_extra=y_all[tr0],
    desc="tier_a",
)
del s_rest_tr0_jitter; gc.collect()
log(f"  Tier-A done: {time.perf_counter()-t_ta:.1f}s")

# Count PNGs
n_base_tr  = sum(1 for _ in (BASE_DIR / "train").glob("*/*.png"))
n_ta_tr    = sum(1 for _ in (TIER_A_DIR / "train").glob("*/*.png"))
n_val_pngs = sum(1 for _ in (BASE_DIR / "val").glob("*/*.png"))
log(f"  Base train: {n_base_tr}  Tier-A train: {n_ta_tr}  val: {n_val_pngs}")

# ─── C0: no augmentation ──────────────────────────────────────────────────────
log("\nC0: no augmentation (reference)")
t_c0_start = time.perf_counter()
ba_c0, ec_c0, probs_c0 = run_ablation(
    "C0", BASE_DIR, core15_wrong_f0, y_all[val0], seed=SEED,
)
t_c0 = time.perf_counter() - t_c0_start
log(f"\nC0 TIMING: {t_c0:.1f}s ({t_c0/60:.1f}min)")
log(f"Projected 6 configs: {t_c0*6:.1f}s ({t_c0*6/60:.1f}min)")
log(f"C0 vs node_0057 0.9401: {ba_c0:.6f} (delta={ba_c0-0.9401:+.4f})")

# ─── C1: Tier A only ──────────────────────────────────────────────────────────
ba_c1, ec_c1, probs_c1 = run_ablation(
    "C1", TIER_A_DIR, core15_wrong_f0, y_all[val0], seed=SEED + 1,
)

# ─── C2: Tier B only (erasing + pixel noise) ──────────────────────────────────
ba_c2, ec_c2, probs_c2 = run_ablation(
    "C2", BASE_DIR, core15_wrong_f0, y_all[val0], seed=SEED + 2,
    erasing=0.4,
    pixel_noise_std=0.02,
)

# ─── C3: Tier C only (geometric) ─────────────────────────────────────────────
ba_c3, ec_c3, probs_c3 = run_ablation(
    "C3", BASE_DIR, core15_wrong_f0, y_all[val0], seed=SEED + 3,
    fliplr=0.5,
    flipud=0.5,
    degrees=15.0,
    shear=10.0,
    scale=0.2,
)

# ─── C4: A + B ────────────────────────────────────────────────────────────────
ba_c4, ec_c4, probs_c4 = run_ablation(
    "C4", TIER_A_DIR, core15_wrong_f0, y_all[val0], seed=SEED + 4,
    erasing=0.4,
    pixel_noise_std=0.02,
)

# ─── C5: A + B + C ────────────────────────────────────────────────────────────
ba_c5, ec_c5, probs_c5 = run_ablation(
    "C5", TIER_A_DIR, core15_wrong_f0, y_all[val0], seed=SEED + 5,
    erasing=0.4,
    pixel_noise_std=0.02,
    fliplr=0.5,
    flipud=0.5,
    degrees=15.0,
    shear=10.0,
    scale=0.2,
)

# ─── Results table ────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log(f"ABLATION RESULTS  backbone={BACKBONE}  optimizer={OPTIMIZER}  epochs={N_EPOCHS}")
log("=" * 70)
log(f"{'Config':<8} {'Tiers':<14} {'Fold-0 BA':<14} {'err_corr':<12} {'vs C0':>8}  Kill-switch")
log("-" * 70)
configs = [
    ("C0", "none",  ba_c0, ec_c0),
    ("C1", "A",     ba_c1, ec_c1),
    ("C2", "B",     ba_c2, ec_c2),
    ("C3", "C",     ba_c3, ec_c3),
    ("C4", "A+B",   ba_c4, ec_c4),
    ("C5", "A+B+C", ba_c5, ec_c5),
]
for name, tiers, ba, ec in configs:
    delta  = ba - ba_c0
    passes = "PASS" if ba >= 0.955 and ec <= 0.6 else "fail"
    log(f"{name:<8} {tiers:<14} {ba:<14.6f} {ec:<12.4f} {delta:>+8.4f}  {passes}")
log("=" * 70)

best_ba_all = max(c[2] for c in configs)
best_idx    = [c[2] for c in configs].index(best_ba_all)
best_config = configs[best_idx]
log(f"BEST: {best_config[0]} ({best_config[1]}) BA={best_ba_all:.6f} ec={best_config[3]:.4f}")

print(f"fold0_C0_BA={ba_c0:.6f}  fold0_C0_ec={ec_c0:.4f}", flush=True)
print(f"fold0_best={best_config[0]}  best_BA={best_ba_all:.6f}", flush=True)

# Features file (always written)
(NODE_SRC / "features.txt").write_text(
    "\n".join(sorted(["u", "g", "r", "i", "z", "redshift"])) + "\n"
)

# ─── Kill-switch ──────────────────────────────────────────────────────────────
KILL_BA = 0.955
KILL_EC = 0.6

winners = [(c[0], c[1], c[2], c[3]) for c in configs
           if c[2] >= KILL_BA and c[3] <= KILL_EC]

probs_map = {
    "C0": probs_c0, "C1": probs_c1, "C2": probs_c2,
    "C3": probs_c3, "C4": probs_c4, "C5": probs_c5,
}
best_probs = probs_map[best_config[0]]

if not winners:
    log("\n" + "=" * 70)
    log("VERDICT: DEAD — no config cleared BA>=0.955 AND err_corr<=0.6")
    log(f"Best BA: {best_ba_all:.6f}  (threshold 0.955)")
    log("=" * 70)

    partial_oof = np.full((n_train, N_CLASSES), np.nan, dtype=np.float32)
    partial_oof[val0] = best_probs
    np.save(NODE_DIR / "oof_fold0_only.npy", partial_oof)
    log(f"Saved oof_fold0_only.npy  (config {best_config[0]})")

    print(f"cv=FOLD0_ONLY_BEST={best_ba_all:.6f}", flush=True)
    print("status=dead  leak=clean", flush=True)

    # Clean up
    shutil.rmtree(PNG_TMP, ignore_errors=True)

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Backbone: {BACKBONE}  Optimizer: {OPTIMIZER}  Epochs: {N_EPOCHS}")
    for name, tiers, ba, ec in configs:
        log(f"  {name} ({tiers:8s}): fold-0 BA={ba:.6f}  err_corr={ec:.4f}")
    total_elapsed = time.perf_counter() - T0
    log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    log("Done. node_0058 ablation — dead path.")
    sys.exit(0)

# ─── PROCEED: full 5-fold ─────────────────────────────────────────────────────
best_winner = max(winners, key=lambda x: x[2])
log("\n" + "=" * 70)
log(f"PROCEED: {len(winners)} config(s) cleared kill-switch")
log(f"Selected: {best_winner[0]} ({best_winner[1]})  BA={best_winner[2]:.6f}")
log("=" * 70)

# Map config to augmentation params
AUG_PARAMS = {
    "C0": dict(use_ta=False, erasing=0.0, fliplr=0.0, flipud=0.0,
               degrees=0.0, shear=0.0, scale=0.0, pixel_noise_std=0.0),
    "C1": dict(use_ta=True,  erasing=0.0, fliplr=0.0, flipud=0.0,
               degrees=0.0, shear=0.0, scale=0.0, pixel_noise_std=0.0),
    "C2": dict(use_ta=False, erasing=0.4, fliplr=0.0, flipud=0.0,
               degrees=0.0, shear=0.0, scale=0.0, pixel_noise_std=0.02),
    "C3": dict(use_ta=False, erasing=0.0, fliplr=0.5, flipud=0.5,
               degrees=15.0, shear=10.0, scale=0.2, pixel_noise_std=0.0),
    "C4": dict(use_ta=True,  erasing=0.4, fliplr=0.0, flipud=0.0,
               degrees=0.0, shear=0.0, scale=0.0, pixel_noise_std=0.02),
    "C5": dict(use_ta=True,  erasing=0.4, fliplr=0.5, flipud=0.5,
               degrees=15.0, shear=10.0, scale=0.2, pixel_noise_std=0.02),
}
wp = AUG_PARAMS[best_winner[0]]

oof_proba        = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_scores  = []

log("\nStarting FULL 5-FOLD OOF LOOP ...")

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.array(fi["val_idx"])
    tr_idx  = np.setdiff1d(np.arange(n_train), val_idx)
    f_seed  = SEED + (fold_id + 1) * 100
    seed_everything(f_seed)
    log(f"\nFold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")
    t_fold = time.perf_counter()

    # Build fold PNG dataset
    fold_dir = PNG_TMP / f"fold_{fold_id}"
    if wp["use_ta"]:
        rng_fold = np.random.default_rng(f_seed + 200)
        s_rest_jitter_f = build_s_rest_tier_a(
            train_raw[BANDS].values[tr_idx].astype(np.float64),
            train_raw["redshift"].values[tr_idx].astype(np.float64),
            rng_fold,
        )
        build_png_dataset(
            s_rest_tr[tr_idx], y_all[tr_idx],
            s_rest_tr[val_idx], y_all[val_idx],
            fold_dir,
            s_rest_train_extra=s_rest_jitter_f,
            y_train_extra=y_all[tr_idx],
            desc=f"f{fold_id}",
        )
        del s_rest_jitter_f; gc.collect()
    else:
        build_png_dataset(
            s_rest_tr[tr_idx], y_all[tr_idx],
            s_rest_tr[val_idx], y_all[val_idx],
            fold_dir, desc=f"f{fold_id}",
        )

    model_f = _train_yolo(
        data_dir=fold_dir,
        run_name=f"{best_winner[0]}_f{fold_id}",
        epochs=N_EPOCHS,
        seed=f_seed,
        erasing=wp["erasing"],
        fliplr=wp["fliplr"],
        flipud=wp["flipud"],
        degrees=wp["degrees"],
        shear=wp["shear"],
        scale=wp["scale"],
        pixel_noise_std=wp["pixel_noise_std"],
    )

    # Val predictions (val always clean)
    val_probs, fold_ba, _ = predict_on_png_dir(
        model_f, fold_dir / "val", y_all[val_idx],
    )
    oof_proba[val_idx] = val_probs.astype(np.float32)
    per_fold_scores.append(fold_ba)
    log(f"  fold {fold_id}: BA={fold_ba:.6f}  elapsed={time.perf_counter()-t_fold:.1f}s")
    print(f"fold{fold_id}_score={fold_ba:.6f}", flush=True)

    # Test predictions
    test_dir = PNG_TMP / f"test_{fold_id}"
    write_pngs(s_rest_te, None, test_dir, chunk_size=5000, desc=f"te{fold_id}")
    test_paths = sorted(
        (test_dir / "_test_").iterdir(), key=lambda x: int(x.stem)
    )
    test_probs_fold = []
    BS = 512
    for i in range(0, len(test_paths), BS):
        batch_p = [str(p) for p in test_paths[i:i+BS]]
        results = model_f.predict(batch_p, verbose=False, imgsz=IMGSZ)
        for r in results:
            test_probs_fold.append(r.probs.data.cpu().float().numpy())
    test_probs_fold = np.stack(test_probs_fold, axis=0)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    del model_f; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(fold_dir,  ignore_errors=True)
    shutil.rmtree(test_dir,  ignore_errors=True)

mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"\nFull OOF CV: {mean_cv:.6f} ± {sem_cv:.6f}")
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
print(f"cv={mean_cv:.6f}", flush=True)

# Error correlation (full OOF)
core15_wrong_all = (core15_mean.argmax(1) != y_all).astype(float)
n58_wrong_all    = (oof_proba.argmax(1) != y_all).astype(float)
err_corr_full    = float(np.corrcoef(core15_wrong_all, n58_wrong_all)[0, 1])
log(f"Full OOF err-corr vs CORE15: {err_corr_full:.4f}")

# Save artifacts
np.save(NODE_DIR / "oof.npy", oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log("Saved oof.npy  test_probs.npy")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv  {len(sub)} rows")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full BA={oof_metric:.6f}")

# Clean up
shutil.rmtree(PNG_TMP, ignore_errors=True)

# ─── Final summary ────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("FINAL SUMMARY")
log("=" * 70)
log(f"Backbone: {BACKBONE}  Optimizer: {OPTIMIZER}  Epochs: {N_EPOCHS}")
for name, tiers, ba, ec in configs:
    log(f"  {name} ({tiers:8s}): fold-0 BA={ba:.6f}  err_corr={ec:.4f}")
total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done. node_0058 yolo26n-cls + MuSGD ablation complete.")
