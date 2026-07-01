"""Reusable, leak-safe cleaning + feature helpers for playground-series-s6e6.

Stellar-class (GALAXY/QSO/STAR) tabular comp. Data is already clean (no NaNs, no
sentinels), so "cleaning" here is mostly type hygiene + physically-motivated,
STATELESS feature engineering (row-wise magnitude differences), which is
fit-inside-fold-safe by construction (it learns nothing from the data).

Every helper takes a DataFrame and returns a transformed COPY. Anything that
learned from data would expose fit()/transform(); none here do, on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "id"
TARGET_COL = "class"
MAGS = ["u", "g", "r", "i", "z"]              # SDSS photometric bands (blue→red)
CAT_COLS = ["spectral_type", "galaxy_population"]
NUM_COLS = ["alpha", "delta", *MAGS, "redshift"]


def cast_categoricals(df: pd.DataFrame, cols: list[str] = CAT_COLS) -> pd.DataFrame:
    """Cast the known low-cardinality string columns to pandas 'category' dtype.

    Stateless: the category set is taken from each frame independently. GBDTs
    (LightGBM/CatBoost) consume 'category' dtype natively; this is a no-op for
    XGBoost paths that one-hot later.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add classic astronomy 'color indices' = adjacent-band magnitude differences.

    Colors (u-g, g-r, r-i, i-z) plus the broad u-z span are the canonical
    discriminators between stars, galaxies and quasars and are invariant to
    overall brightness. Pure row-wise arithmetic -> no fitting, no leakage.
    Only added when all magnitude columns are present; otherwise returns a copy
    unchanged.
    """
    out = df.copy()
    if not all(m in out.columns for m in MAGS):
        return out
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_z"] = out["u"] - out["z"]   # broadest color span
    return out


def add_extended_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add the full pairwise color set + second-order (color-color curvature) terms.

    Beyond the adjacent-band colors, astronomers read object type off positions in
    color-color space, so non-adjacent colors (u-r, u-i, g-i, r-z) and curvature
    terms ((u-g)-(g-r), (g-r)-(r-i)) give trees the exact axes of the stellar locus.
    Pure row-wise arithmetic -> stateless, leak-safe. No-op if mags absent.
    """
    out = df.copy()
    if not all(m in out.columns for m in MAGS):
        return out
    out["u_r"] = out["u"] - out["r"]
    out["u_i"] = out["u"] - out["i"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["c_ug_gr"] = (out["u"] - out["g"]) - (out["g"] - out["r"])   # locus curvature
    out["c_gr_ri"] = (out["g"] - out["r"]) - (out["r"] - out["i"])
    return out


def add_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Redshift transforms: log1p (compress the long QSO tail) + physical-regime flags.

    STAR sits at z≈0, GALAXY low-moderate, QSO high — explicit flags help splits land
    on those regimes. Stateless (fixed physical thresholds, not learned). No-op if
    `redshift` absent.
    """
    out = df.copy()
    if "redshift" not in out.columns:
        return out
    z = out["redshift"]
    out["log1p_redshift"] = np.log1p(z.clip(lower=0))   # clip tiny negatives (noise) to 0
    out["is_star_z"] = (z < 0.0025).astype("int8")       # ~stellar (z≈0)
    out["is_highz"] = (z > 1.0).astype("int8")           # high-z -> strongly QSO
    return out


def add_qso_colorbox(df: pd.DataFrame) -> pd.DataFrame:
    """Literature QSO color-box flags (UV-excess selection) — attacks QSO↔GALAXY/STAR.

    Low-z quasars sit off the stellar locus at u-g < 0.6 and g-r > 0 (UV excess).
    Encodes the astronomer's color cut as binary features. Stateless. Needs u,g,r.
    """
    out = df.copy()
    if not all(m in out.columns for m in ("u", "g", "r")):
        return out
    ug = out["u"] - out["g"]
    gr = out["g"] - out["r"]
    out["qso_box"] = ((ug < 0.6) & (gr > 0)).astype("int8")
    out["uv_excess"] = (ug < 0.4).astype("int8")
    return out


# J2000 north galactic pole / ascending node constants (deg) for the RA/Dec->(l,b) rotation
_RA_NGP, _DEC_NGP, _L_NCP = 192.85948, 27.12825, 122.93192


def add_galactic_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Convert equatorial (alpha=RA, delta=Dec, deg) to galactic (l, b, deg).

    Galactic latitude `b` is an extinction proxy (reddening is worst near the plane,
    b≈0); `l` is cheap positional signal. Closed-form J2000 rotation — stateless,
    deterministic, leak-safe. No-op if alpha/delta absent.
    """
    out = df.copy()
    if not all(c in out.columns for c in ("alpha", "delta")):
        return out
    ra = np.radians(out["alpha"].to_numpy())
    dec = np.radians(out["delta"].to_numpy())
    ra_ngp, dec_ngp, l_ncp = map(np.radians, (_RA_NGP, _DEC_NGP, _L_NCP))
    sin_b = np.sin(dec) * np.sin(dec_ngp) + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))
    y = np.cos(dec) * np.sin(ra - ra_ngp)
    x = np.sin(dec) * np.cos(dec_ngp) - np.cos(dec) * np.sin(dec_ngp) * np.cos(ra - ra_ngp)
    l = l_ncp - np.arctan2(y, x)
    out["gal_l"] = np.degrees(l) % 360.0
    out["gal_b"] = np.degrees(b)
    return out


def add_positional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-safe positional features from (alpha=RA, delta=Dec, deg).

    The drop-column study showed `delta` is the single most IRREPLACEABLE feature — the
    synthetic data has strong class-vs-sky-position structure. Raw coords + axis-aligned
    tree splits exploit it poorly, so we expose it better. Everything here is LABEL-FREE and
    row-wise stateless (fit-inside-fold-safe; nothing is learned from the target):
      - sin/cos of RA fix the 0°/360° wrap (so RA 359 and 1 read as neighbours).
      - unit-sphere cartesian (sx,sy,sz) give smooth joint (RA,Dec) coords the tree can
        carve into 2-D regions; sz == sin(dec).
      - delta×redshift interactions surface 'high-z AT this declination' in one split.
      - sky_cell: a coarse (RA 10° × Dec 5°) grid-cell id as a CATEGORY — the GBDT learns
        the per-region class tendency INSIDE the fold (leak-safe; NO manual target encoding).
    `sky_cell`'s category vocabulary is per-frame; the modelling node aligns test→train
    categories (train-defined vocabulary, test-only cells → missing). No-op if alpha/delta absent.
    """
    out = df.copy()
    if not all(c in out.columns for c in ("alpha", "delta")):
        return out
    ra = np.radians(out["alpha"].to_numpy())
    dec = np.radians(out["delta"].to_numpy())
    out["alpha_sin"] = np.sin(ra)
    out["alpha_cos"] = np.cos(ra)
    out["sx"] = np.cos(dec) * np.cos(ra)
    out["sy"] = np.cos(dec) * np.sin(ra)
    out["sz"] = np.sin(dec)
    if "redshift" in out.columns:
        z = out["redshift"].to_numpy()
        d = out["delta"].to_numpy()
        out["delta_x_redshift"] = d * z
        out["delta_x_logz"] = d * np.log1p(np.clip(z, 0, None))
    a_bin = np.floor(out["alpha"].to_numpy() / 10.0).astype(int)   # RA 10° tiles
    d_bin = np.floor(out["delta"].to_numpy() / 5.0).astype(int)    # Dec 5° tiles
    cell = pd.Series(a_bin, index=out.index).astype(str) + "_" + pd.Series(d_bin, index=out.index).astype(str)
    out["sky_cell"] = cell.astype("category")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All modelling features: everything except id and target. Guards id-leakage."""
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
