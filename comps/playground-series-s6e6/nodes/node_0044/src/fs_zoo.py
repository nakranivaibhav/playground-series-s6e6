"""
fs_zoo.py — shared feature pipeline for node_0044/0045/0046/0047.
Port of xgb-v5-for-s6e6.py (cudf/cuml) to CPU pandas + numpy.
All encoders (TargetEncoder, frequency) are fit INSIDE each train fold only.
Original-prior features are computed on sdss17 data only (no train/val labels used).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
SEED = 42
BANDS = ['u', 'g', 'r', 'i', 'z']
RAW_NUM_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
CLASSES = ['GALAXY', 'QSO', 'STAR']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

TE_SMOOTH = 16.0
TE_INNER_SPLITS = 7

ART_COLOR_BIN_SPECS = [
    ('u_g', 2.0, 'half'),
    ('g_r', 2.0, 'half'),
    ('r_i', 2.0, 'half'),
    ('i_z', 2.0, 'half'),
    ('u_r', 1.0, 'one'),
    ('redshift', 10.0, 'tenth'),
    ('alpha', 0.2, 'deg5'),
    ('delta', 0.2, 'deg5'),
]

TOP_FEATURES = ['TE_art_g_r_half_x_redshift_tenth_STAR',
 'TE_art_g_r_half_x_redshift_tenth_QSO',
 'orig_art_g_r_half_x_redshift_tenth_prior_QSO',
 'redshift_u',
 'z_over_redshift',
 'TE_art_u_g_half_x_redshift_tenth_QSO',
 'TE_art_g_r_half_x_redshift_tenth_GALAXY',
 'g_z',
 'g_i',
 'orig_g_qbin64_prior_QSO',
 'redshift_g',
 'orig_g_qbin16_prior_QSO',
 'i_over_redshift',
 'redshift_abs',
 'TE_redshift_qbin16_GALAXY',
 'redshift_log1p_abs',
 'u_over_redshift',
 'TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY',
 'g_i_abs',
 'orig_art_g_floor_prior_QSO',
 'u_i',
 'TE_redshift_qbin64_GALAXY',
 'orig_art_u_floor_x_z_floor_prior_QSO',
 'TE_alpha_qbin64__x__delta_qbin64_STAR',
 'art_redshift_floor_freq',
 'g_over_redshift',
 'u_r_abs',
 'redshift_is_neg',
 'orig_g_qbin256_prior_QSO',
 'TE_art_redshift_tenth_GALAXY',
 'TE_alpha_qbin64__x__delta_qbin64_GALAXY',
 'TE_redshift_qbin64_QSO',
 'art_redshift_floor',
 'orig_art_u_g_half_x_redshift_tenth_prior_QSO',
 'u_r',
 'mag_slope',
 'art_g_r_half_x_redshift_tenth_freq_log1p',
 'r_over_redshift',
 'g_qbin16',
 'TE_art_u_floor_x_z_floor_QSO',
 'TE_u_g_qbin64__x__g_r_qbin64_STAR',
 'orig_redshift_qbin64_prior_STAR',
 'flux_std',
 'redshift',
 'TE_art_u_g_half_x_redshift_tenth_GALAXY',
 'orig_redshift_qbin64_prior_GALAXY',
 'orig_g_qbin16_prior_GALAXY',
 'flux_g',
 'orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY',
 'mag_std',
 'TE_art_u_floor_x_z_floor_GALAXY',
 'redshift_z',
 'orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO',
 'g',
 'orig_redshift_qbin256_prior_GALAXY',
 'orig_art_i_floor_prior_QSO',
 'flux_range',
 'orig_art_redshift_floor_prior_GALAXY',
 'orig_art_r_floor_prior_QSO',
 'orig_alpha_qbin64__x__delta_qbin64_prior_STAR',
 'art_r_floor',
 'TE_redshift_qbin64_STAR',
 'TE_g_qbin64_QSO',
 'art_g_r_half_x_redshift_tenth_freq',
 'orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY',
 'orig_art_i_floor_count',
 'TE_u_r_qbin64_GALAXY',
 'u_g',
 'TE_g_qbin16_QSO',
 'g_r_x_redshift',
 'z',
 'redshift_r',
 'r_z',
 'art_g_r_half_x_redshift_tenth',
 'mag_range',
 'i',
 'TE_art_u_g_half_x_redshift_tenth_STAR',
 'art_g_floor',
 'flux_i',
 'TE_art_redshift_tenth_STAR',
 'orig_r_qbin16_prior_QSO',
 'redshift_i',
 'r_z_x_redshift',
 'flux_z',
 'g_i_x_redshift',
 'orig_art_u_floor_x_z_floor_prior_GALAXY',
 'r',
 'TE_i_qbin64_QSO',
 'r_z_abs',
 'orig_g_qbin64_prior_GALAXY',
 'TE_redshift_qbin64__x__mag_mean_qbin64_QSO',
 'art_redshift_floor_freq_log1p',
 'orig_i_qbin16_prior_QSO',
 'orig_art_u_r_one_prior_QSO',
 'orig_mag_range_qbin16_prior_STAR',
 'orig_art_alpha_deg5_x_delta_deg5_prior_GALAXY',
 'TE_redshift_qbin16_STAR',
 'orig_art_redshift_tenth_prior_GALAXY',
 'TE_art_alpha_deg5_x_delta_deg5_STAR',
 'flux_r',
 'orig_art_z_floor_count',
 'TE_u_r_qbin64_QSO',
 'orig_art_alpha_deg5_x_delta_deg5_prior_STAR',
 'orig_u_qbin16_prior_QSO',
 'r_i_x_redshift',
 'art_i_floor',
 'orig_mag_range_qbin64_prior_STAR',
 'orig_z_qbin16_prior_QSO',
 'orig_redshift_qbin16_prior_GALAXY',
 'art_u_g_half',
 'art_z_floor_freq',
 'TE_art_g_floor_QSO',
 'art_r_i_half_freq_log1p',
 'orig_mag_range_qbin16_prior_GALAXY',
 'mag_max',
 'flux_min',
 'TE_g_qbin64_GALAXY',
 'orig_art_u_g_half_x_redshift_tenth_prior_STAR',
 'redshift_qbin64__x__mag_mean_qbin64',
 'TE_alpha_qbin64__x__delta_qbin64_QSO',
 'orig_i_qbin256_prior_QSO',
 'art_u_g_half_x_redshift_tenth_freq_log1p',
 'orig_art_i_floor_prior_GALAXY',
 'TE_mag_range_qbin64_QSO',
 'orig_art_r_i_half_count',
 'art_z_floor_freq_log1p',
 'color_plane_radius_ug_gr',
 'art_g_r_half',
 'art_r_i_half_freq',
 'TE_g_qbin16_GALAXY',
 'TE_redshift_qbin64__x__mag_mean_qbin64_STAR',
 'TE_art_r_floor_QSO',
 'flux_u',
 'flux_max',
 'orig_i_qbin64_prior_QSO',
 'u',
 'TE_art_alpha_deg5_x_delta_deg5_GALAXY',
 'flux_mean',
 'orig_u_g_qbin16_prior_STAR',
 'u_g_abs',
 'orig_art_u_floor_prior_QSO',
 'mag_mean_qbin16',
 'TE_art_alpha_floor_STAR',
 'art_u_floor_x_z_floor',
 'u_z',
 'redshift_qbin64__x__mag_mean_qbin64_freq_log1p',
 'orig_art_alpha_deg5_x_delta_deg5_prior_QSO',
 'orig_art_g_r_half_x_redshift_tenth_prior_STAR',
 'orig_mag_range_qbin256_prior_QSO',
 'mag_min',
 'orig_u_r_qbin16_prior_QSO',
 'orig_art_redshift_tenth_count',
 'r_i',
 'redshift_qbin16',
 'TE_i_qbin16_QSO',
 'TE_u_g_qbin64_STAR',
 'art_u_g_half_x_redshift_tenth_freq',
 'TE_art_r_floor_GALAXY',
 'redshift_qbin64__x__mag_mean_qbin64_freq',
 'TE_mag_range_qbin64_GALAXY',
 'u_qbin16',
 'art_i_floor_freq',
 'mag_range_qbin256',
 'orig_mag_range_qbin256_prior_STAR',
 'art_i_floor_freq_log1p',
 'art_u_g_half_x_redshift_tenth',
 'orig_z_qbin64_prior_QSO',
 'art_r_floor_freq_log1p',
 'alpha_sin',
 'orig_art_r_floor_prior_GALAXY',
 'TE_r_i_qbin64_QSO',
 'orig_u_qbin16_prior_STAR',
 'color_plane_radius_ri_iz',
 'TE_art_alpha_deg5_x_delta_deg5_QSO',
 'delta_cos',
 'TE_r_i_qbin16_GALAXY',
 'art_g_floor_freq',
 'orig_art_u_g_half_x_redshift_tenth_count',
 'orig_delta_qbin256_prior_STAR',
 'TE_r_i_qbin16_QSO',
 'orig_art_u_floor_x_z_floor_prior_STAR',
 'mag_range_qbin16',
 'art_delta_deg5',
 'TE_u_g_qbin16_STAR',
 'u_r_x_redshift',
 'sky_y',
 'TE_r_i_qbin64_GALAXY',
 'art_g_floor_freq_log1p',
 'orig_art_alpha_floor_prior_STAR',
 'orig_delta_qbin256_prior_GALAXY',
 'u_g_x_redshift',
 'g_r',
 'r_qbin16',
 'TE_z_qbin64_QSO',
 'art_redshift_tenth_freq_log1p',
 'alpha_qbin256',
 'orig_i_qbin16_prior_GALAXY',
 'orig_art_alpha_floor_prior_GALAXY',
 'orig_art_r_i_half_prior_GALAXY',
 'orig_art_g_floor_prior_GALAXY',
 'orig_redshift_qbin256_count',
 'TE_g_qbin64_STAR',
 'orig_u_qbin16_count',
 'art_redshift_tenth_freq',
 'z_qbin16',
 'r_qbin64',
 'TE_u_qbin16_QSO',
 'art_u_r_one_freq_log1p',
 'orig_u_qbin64_prior_QSO',
 'orig_art_alpha_deg5_prior_STAR',
 'g_qbin64',
 'TE_r_qbin16_QSO',
 'i_qbin16',
 'orig_z_qbin256_prior_QSO',
 'orig_art_delta_deg5_prior_GALAXY',
 'orig_g_qbin256_prior_GALAXY',
 'blue_curvature',
 'TE_u_qbin64_QSO',
 'orig_art_u_g_half_x_redshift_tenth_prior_GALAXY',
 'TE_art_redshift_tenth_QSO',
 'alpha',
 'art_u_r_one_freq',
 'orig_mag_range_qbin256_prior_GALAXY',
 'delta_sin',
 'art_redshift_tenth',
 'TE_art_alpha_floor_GALAXY',
 'art_alpha_floor',
 'delta',
 'TE_art_u_r_one_GALAXY',
 'color_plane_angle_ug_gr',
 'orig_art_redshift_tenth_prior_QSO',
 'orig_art_r_i_half_prior_QSO',
 'orig_art_g_floor_prior_STAR',
 'mag_curvature',
 'TE_art_u_floor_x_z_floor_STAR',
 'art_delta_floor',
 'u_qbin16_freq_log1p',
 'TE_g_r_qbin64_GALAXY',
 'TE_r_qbin64_STAR',
 'sky_z',
 'art_u_floor_freq',
 'mag_mean',
 'g_r_abs',
 'TE_z_qbin16_QSO',
 'sky_x',
 'u_qbin16_freq',
 'TE_u_qbin64_STAR',
 'orig_art_u_floor_count',
 'art_alpha_floor_x_delta_floor',
 'art_u_floor_freq_log1p',
 'orig_z_qbin16_count',
 'art_u_r_one',
 'art_u_g_half_freq',
 'TE_redshift_qbin16_QSO',
 'orig_alpha_qbin64__x__delta_qbin64_prior_QSO',
 'z_qbin16_freq',
 'TE_u_g_qbin64_QSO',
 'TE_u_g_qbin64_GALAXY',
 'orig_g_qbin64_prior_STAR',
 'orig_art_g_r_half_prior_GALAXY',
 'TE_g_qbin16_STAR',
 'orig_art_g_r_half_x_redshift_tenth_count',
 'orig_art_delta_floor_prior_QSO',
 'orig_art_u_r_one_count',
 'orig_spectral_x_pop_prior_QSO',
 'redshift_qbin256',
 'orig_art_u_g_half_prior_STAR',
 'TE_g_r_qbin64_STAR',
 'orig_art_u_r_one_prior_STAR',
 'art_delta_floor_freq_log1p',
 'TE_art_alpha_floor_QSO',
 'art_delta_floor_freq',
 'art_delta_deg5_freq',
 'art_u_g_half_freq_log1p',
 'art_delta_deg5_freq_log1p',
 'z_qbin16_freq_log1p',
 'TE_u_r_qbin16_GALAXY',
 'TE_art_i_floor_GALAXY',
 'orig_art_alpha_floor_prior_QSO',
 'orig_art_alpha_deg5_prior_GALAXY',
 'orig_art_u_g_half_count',
 'orig_art_g_floor_count',
 'TE_art_i_floor_QSO',
 'art_alpha_deg5_freq_log1p',
 'art_r_floor_freq',
 'art_alpha_deg5_freq',
 'orig_art_alpha_deg5_count',
 'TE_art_g_floor_GALAXY',
 'orig_u_r_qbin64_prior_QSO',
 'TE_u_g_qbin64__x__g_r_qbin64_QSO',
 'orig_art_u_floor_prior_STAR',
 'art_alpha_deg5_x_delta_deg5',
 'orig_art_delta_floor_prior_GALAXY',
 'art_alpha_deg5',
 'orig_art_delta_deg5_count',
 'orig_u_r_qbin256_prior_QSO',
 'orig_art_alpha_deg5_prior_QSO',
 'orig_art_g_r_half_prior_QSO',
 'TE_art_alpha_deg5_STAR',
 'orig_art_delta_floor_count',
 'TE_g_r_qbin64_QSO',
 'art_alpha_deg5_x_delta_deg5_freq',
 'art_u_floor',
 'art_alpha_deg5_x_delta_deg5_freq_log1p',
 'orig_art_delta_deg5_prior_QSO',
 'art_alpha_floor_freq_log1p',
 'art_r_i_half',
 'orig_art_r_i_half_prior_STAR',
 'orig_u_g_qbin64_prior_STAR',
 'art_alpha_floor_freq',
 'orig_art_g_r_half_x_redshift_tenth_prior_GALAXY',
 'TE_art_alpha_deg5_GALAXY',
 'orig_art_u_g_half_prior_GALAXY',
 'orig_art_u_floor_x_z_floor_count',
 'orig_art_alpha_floor_count',
 'TE_art_delta_floor_QSO',
 'TE_art_u_floor_QSO',
 'TE_art_delta_deg5_GALAXY',
 'orig_art_alpha_deg5_x_delta_deg5_count',
 'orig_art_i_floor_prior_STAR',
 'orig_art_redshift_floor_prior_QSO',
 'TE_art_delta_floor_GALAXY',
 'TE_art_alpha_deg5_QSO',
 'orig_alpha_qbin256_prior_GALAXY',
 'TE_u_g_qbin64__x__g_r_qbin64_GALAXY',
 'orig_art_delta_deg5_prior_STAR',
 'art_u_floor_x_z_floor_freq_log1p',
 'TE_art_r_floor_STAR',
 'orig_art_z_floor_prior_GALAXY',
 'TE_art_u_g_half_GALAXY',
 'TE_art_g_floor_STAR',
 'TE_art_redshift_floor_STAR',
 'orig_art_r_floor_count',
 'TE_art_r_i_half_STAR',
 'art_u_floor_x_z_floor_freq',
 'TE_art_i_floor_STAR',
 'TE_art_delta_deg5_QSO',
 'TE_art_u_floor_STAR',
 'orig_art_delta_floor_prior_STAR',
 'TE_art_r_i_half_GALAXY',
 'TE_art_u_g_half_QSO',
 'orig_art_u_g_half_prior_QSO',
 'art_alpha_floor_x_delta_floor_freq',
 'orig_art_r_floor_prior_STAR',
 'orig_art_i_z_half_prior_QSO',
 'TE_art_delta_deg5_STAR',
 'orig_alpha_qbin256_prior_STAR',
 'art_alpha_floor_x_delta_floor_freq_log1p',
 'TE_art_z_floor_STAR',
 'orig_art_z_floor_prior_QSO',
 'orig_art_u_floor_prior_GALAXY',
 'orig_art_u_r_one_prior_GALAXY',
 'TE_art_g_r_half_QSO',
 'art_g_r_half_freq_log1p',
 'TE_art_u_r_one_QSO',
 'TE_art_u_g_half_STAR',
 'TE_art_r_i_half_QSO',
 'art_z_floor',
 'TE_art_z_floor_GALAXY',
 'TE_art_z_floor_QSO',
 'TE_art_u_floor_GALAXY',
 'TE_art_g_r_half_GALAXY',
 'TE_art_delta_floor_STAR',
 'orig_art_g_r_half_prior_STAR',
 'TE_art_g_r_half_STAR',
 'TE_art_u_r_one_STAR',
 'orig_art_z_floor_prior_STAR',
 'TE_art_redshift_floor_GALAXY',
 'TE_art_redshift_floor_QSO',
 'TE_art_i_z_half_QSO']


def cat_key(s):
    """Convert series to string category key, filling NA with '__NA__'."""
    return s.astype(str).fillna('__NA__')


def spectral_type_from_gr(r_minus_g):
    return pd.cut(r_minus_g, bins=[-np.inf, -1, -0.5, 0, np.inf],
                  labels=['M', 'G/K', 'A/F', 'O/B']).astype(str)


def galaxy_population_from_ur(u_minus_r):
    return pd.cut(u_minus_r, bins=[-np.inf, 2.2, np.inf],
                  labels=['Blue_Cloud', 'Red_Sequence']).astype(str)


def add_public_features(df):
    """
    Add color pairs, band stats, flux features, trig coords, and all 'full' features.
    Works on pandas DataFrame. Returns (df_out, cat_cols).
    """
    out = df.copy()
    for c in RAW_NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')

    # color pairs
    color_pairs = [
        ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
        ("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"),
        ("g", "z"), ("r", "z"),
    ]
    for a, b in color_pairs:
        out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    band_vals = out[BANDS].values.astype(np.float32)
    out["mag_mean"] = band_vals.mean(axis=1).astype("float32")
    out["mag_std"] = band_vals.std(axis=1, ddof=1).astype("float32")
    out["mag_min"] = band_vals.min(axis=1).astype("float32")
    out["mag_max"] = band_vals.max(axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")

    for b in BANDS:
        out[f"redshift_{b}"] = (out["redshift"] * out[b]).astype("float32")

    alpha_rad = out["alpha"].values.astype(np.float32) * np.float32(np.pi / 180.0)
    delta_rad = out["delta"].values.astype(np.float32) * np.float32(np.pi / 180.0)
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")

    spectral_map = {"O/B": 0, "A": 1, "F": 2, "G": 3, "K": 4, "M": 5}
    out["spectral_ord"] = out["spectral_type"].map(spectral_map).fillna(-1).astype("float32")

    # flux features
    flux_arrays = []
    for b in BANDS:
        clipped = np.clip(out[b].values.astype(np.float32), -30, 30)
        flux = np.power(10.0, -0.4 * clipped).astype(np.float32)
        out[f"flux_{b}"] = flux
        flux_arrays.append(flux)
    flux_vals = np.column_stack(flux_arrays)
    out["flux_mean"] = flux_vals.mean(axis=1).astype("float32")
    out["flux_std"] = flux_vals.std(axis=1, ddof=1).astype("float32")
    out["flux_min"] = flux_vals.min(axis=1).astype("float32")
    out["flux_max"] = flux_vals.max(axis=1).astype("float32")
    out["flux_range"] = (out["flux_max"] - out["flux_min"]).astype("float32")

    x = np.arange(len(BANDS), dtype=np.float32)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()
    out["mag_slope"] = ((band_vals - band_vals.mean(axis=1, keepdims=True)) @ x_centered / denom).astype("float32")
    out["mag_curvature"] = (out["u"] - 2 * out["r"] + out["z"]).astype("float32")
    out["blue_curvature"] = (out["u"] - 2 * out["g"] + out["r"]).astype("float32")
    out["red_curvature"] = (out["r"] - 2 * out["i"] + out["z"]).astype("float32")

    for c in ["u_g", "g_r", "r_i", "i_z"]:
        out[f"{c}_per_redshift"] = (out[c] / (out["redshift"].abs() + EPS)).astype("float32")

    # 'full' style features
    out["mag_argmin"] = np.argmin(band_vals, axis=1).astype(np.int16)
    out["mag_argmax"] = np.argmax(band_vals, axis=1).astype(np.int16)

    redshift_arr = out["redshift"].values.astype(np.float32)
    signed_redshift_denom = np.where(
        np.abs(redshift_arr) < EPS,
        np.where(redshift_arr < 0, -EPS, EPS),
        redshift_arr,
    )
    for b in BANDS:
        out[f"{b}_over_redshift"] = (out[b] / (out["redshift"].abs() + EPS)).astype("float32")
        out[f"{b}_over_redshift_signed"] = (out[b].values.astype(np.float32) / signed_redshift_denom).astype("float32")

    out["sky_x"] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype("float32")
    out["sky_y"] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype("float32")
    out["sky_z"] = np.sin(delta_rad).astype("float32")
    out["redshift_abs"] = out["redshift"].abs().astype("float32")
    redshift_abs = out["redshift_abs"].values.astype(np.float32)
    out["redshift_sky_x"] = (redshift_arr * out["sky_x"].values).astype("float32")
    out["redshift_sky_y"] = (redshift_arr * out["sky_y"].values).astype("float32")
    out["redshift_sky_z"] = (redshift_arr * out["sky_z"].values).astype("float32")
    out["redshift_abs_sky_x"] = (redshift_abs * out["sky_x"].values).astype("float32")
    out["redshift_abs_sky_y"] = (redshift_abs * out["sky_y"].values).astype("float32")
    out["redshift_abs_sky_z"] = (redshift_abs * out["sky_z"].values).astype("float32")
    distmod_proxy = (5.0 * np.log10(redshift_abs + EPS)).astype(np.float32)
    out["redshift_distmod_proxy"] = distmod_proxy
    for b in BANDS:
        out[f"{b}_absmag_proxy"] = (out[b].values.astype(np.float32) - distmod_proxy).astype("float32")
    out["mag_mean_absmag_proxy"] = (out["mag_mean"].values - distmod_proxy).astype("float32")
    out["redshift_log1p_abs"] = np.log1p(redshift_abs).astype("float32")
    out["redshift_is_neg"] = (out["redshift"] < 0).astype("int8")

    phys_bins = np.array([-np.inf, 0.05, 0.10, 0.30, 0.60, np.inf], dtype=np.float32)
    phys_codes = np.searchsorted(phys_bins, redshift_arr, side="right") - 1
    phys_codes = np.where(np.isnan(redshift_arr), -1, phys_codes)
    out["redshift_phys_bin"] = phys_codes.astype(np.int8).astype(str)

    gr = out["g_r"].values.astype(np.float32)
    gr_bins = np.array([-np.inf, 0.0, 0.4, 0.8, 1.2, np.inf], dtype=np.float32)
    gr_codes = np.searchsorted(gr_bins, gr, side="right") - 1
    gr_codes = np.where(np.isnan(gr), -1, gr_codes)
    out["g_r_color_bin"] = gr_codes.astype(np.int8).astype(str)
    out["redshift_phys_x_g_r_color"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["g_r_color_bin"])

    out["spectral_type_calc"] = spectral_type_from_gr(out["r"] - out["g"])
    out["galaxy_population_calc"] = galaxy_population_from_ur(out["u"] - out["r"])
    out["spectral_x_pop"] = cat_key(out["spectral_type"]) + "__" + cat_key(out["galaxy_population"])
    out["spectral_calc_x_pop_calc"] = cat_key(out["spectral_type_calc"]) + "__" + cat_key(out["galaxy_population_calc"])
    out["redshift_phys_x_spectral"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["spectral_type"])
    out["redshift_phys_x_pop"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["galaxy_population"])
    out["redshift_phys_x_spectral_pop"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["spectral_x_pop"])

    for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"]:
        out[f"{c}_x_redshift"] = (out[c] * out["redshift"]).astype("float32")
        out[f"{c}_abs"] = out[c].abs().astype("float32")
        out[f"{c}_over_redshift_signed"] = (out[c].values.astype(np.float32) / signed_redshift_denom).astype("float32")

    ug = out["u_g"].values.astype(np.float32)
    grf = out["g_r"].values.astype(np.float32)
    ri = out["r_i"].values.astype(np.float32)
    iz = out["i_z"].values.astype(np.float32)
    out["color_plane_radius_ug_gr"] = np.sqrt(ug**2 + grf**2).astype(np.float32)
    out["color_plane_angle_ug_gr"] = np.arctan2(ug, grf + EPS).astype(np.float32)
    out["color_plane_radius_ri_iz"] = np.sqrt(ri**2 + iz**2).astype(np.float32)
    out["color_plane_angle_ri_iz"] = np.arctan2(ri, iz + EPS).astype(np.float32)

    cat_cols = [
        "spectral_type", "galaxy_population", "spectral_type_calc", "galaxy_population_calc",
        "spectral_x_pop", "spectral_calc_x_pop_calc", "redshift_phys_bin", "g_r_color_bin",
        "redshift_phys_x_g_r_color", "redshift_phys_x_spectral", "redshift_phys_x_pop",
        "redshift_phys_x_spectral_pop",
    ]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, cat_cols


def add_lowfreq_artifact_features(df):
    out = df.copy()
    cat_cols = []
    for c in RAW_NUM_COLS:
        vals = pd.to_numeric(out[c], errors='coerce').values.astype(np.float32)
        finite = ~np.isnan(vals)
        floored = np.floor(np.where(finite, vals, 0.0)).astype(np.int32)
        floored = np.where(finite, floored, -2147483648)
        name = f"art_{c}_floor"
        out[name] = floored.astype(str)
        cat_cols.append(name)
    out["art_alpha_floor_x_delta_floor"] = cat_key(out["art_alpha_floor"]) + "__" + cat_key(out["art_delta_floor"])
    out["art_u_floor_x_z_floor"] = cat_key(out["art_u_floor"]) + "__" + cat_key(out["art_z_floor"])
    cat_cols.extend(["art_alpha_floor_x_delta_floor", "art_u_floor_x_z_floor"])
    return out, cat_cols


def add_color_artifact_features(df):
    out = df.copy()
    cat_cols = []
    for c, scale, tag in ART_COLOR_BIN_SPECS:
        if c not in out.columns:
            continue
        vals = pd.to_numeric(out[c], errors='coerce').values.astype(np.float32)
        finite = ~np.isnan(vals)
        binned = np.floor(np.where(finite, vals * np.float32(scale), 0.0)).astype(np.int32)
        binned = np.where(finite, binned, -2147483648)
        name = f"art_{c}_{tag}"
        out[name] = binned.astype(str)
        cat_cols.append(name)
    if "art_u_g_half" in out.columns and "art_redshift_tenth" in out.columns:
        out["art_u_g_half_x_redshift_tenth"] = cat_key(out["art_u_g_half"]) + "__" + cat_key(out["art_redshift_tenth"])
        cat_cols.append("art_u_g_half_x_redshift_tenth")
    if "art_g_r_half" in out.columns and "art_redshift_tenth" in out.columns:
        out["art_g_r_half_x_redshift_tenth"] = cat_key(out["art_g_r_half"]) + "__" + cat_key(out["art_redshift_tenth"])
        cat_cols.append("art_g_r_half_x_redshift_tenth")
    if "art_alpha_deg5" in out.columns and "art_delta_deg5" in out.columns:
        out["art_alpha_deg5_x_delta_deg5"] = cat_key(out["art_alpha_deg5"]) + "__" + cat_key(out["art_delta_deg5"])
        cat_cols.append("art_alpha_deg5_x_delta_deg5")
    return out, cat_cols


def qcut_codes(values, ref_values, q):
    """Quantile-bin `values` using bins derived from `ref_values`."""
    ref = ref_values[~np.isnan(ref_values)]
    if len(ref) < 2:
        return np.full(len(values), -1, dtype=np.int16)
    probs = np.linspace(0, 1, q + 1)
    bins = np.quantile(ref, probs).astype(np.float32)
    bins = np.unique(bins)
    if len(bins) <= 1:
        return np.full(len(values), -1, dtype=np.int16)
    codes = np.searchsorted(bins, values, side="left") - 1
    codes = np.where(values == bins[0], 0, codes)
    codes = np.where((values < bins[0]) | (values > bins[-1]) | np.isnan(values), -1, codes)
    return np.clip(codes, -1, len(bins) - 2).astype(np.int16)


def add_quantile_bin_features(df, train_test_mask):
    out = df.copy()
    qbin_cols = []
    cols = RAW_NUM_COLS + [c for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean", "mag_range"] if c in out.columns]
    qbins = [16, 64, 256]
    for c in cols:
        s = pd.to_numeric(out[c], errors='coerce').values.astype(np.float32)
        ref = s[train_test_mask]
        for q in qbins:
            name = f"{c}_qbin{q}"
            out[name] = qcut_codes(s, ref, q).astype(str)
            qbin_cols.append(name)
    for a, b in [("alpha_qbin64", "delta_qbin64"), ("u_g_qbin64", "g_r_qbin64"), ("redshift_qbin64", "mag_mean_qbin64")]:
        if a in out.columns and b in out.columns:
            name = f"{a}__x__{b}"
            out[name] = cat_key(out[a]) + "__" + cat_key(out[b])
            qbin_cols.append(name)
    return out, qbin_cols


def select_te_cols(df, cat_cols, max_card=5000):
    cols = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        card = cat_key(df[c]).nunique()
        if card > max_card:
            continue
        cols.append(c)
    return cols


def add_frequency_features(df, cols, fit_mask):
    """Frequency/log1p frequency features fit on fit_mask rows."""
    out = df.copy()
    for c in cols:
        s = cat_key(out[c])
        vc = s[fit_mask].value_counts(dropna=False)
        out[f"{c}_freq"] = s.map(vc).fillna(0).astype("float32")
        out[f"{c}_freq_log1p"] = np.log1p(out[f"{c}_freq"].values).astype("float32")
    return out


def add_original_prior_features(df, cols, orig_mask, orig_y_arr, smooth=0.0):
    """
    Compute P(class|key) from original SDSS17 data only.
    orig_mask: boolean array (len=len(df)) — True for orig rows.
    orig_y_arr: integer labels (0/1/2) for ORIG rows only (len == sum(orig_mask)).
    """
    out = df.copy()
    # orig_y_arr is already only for orig rows (len == sum(orig_mask))
    orig_y = orig_y_arr
    counts_per_class = np.bincount(orig_y.astype(np.int32), minlength=len(CLASSES)).astype(np.float32)
    prior = counts_per_class / max(counts_per_class.sum(), 1.0)
    smooth = float(smooth or 0.0)

    for c in cols:
        key = cat_key(out[c])
        orig_key = key[orig_mask].reset_index(drop=True)
        tmp = pd.DataFrame({'key': orig_key, 'y': orig_y})
        counts = tmp.groupby('key').size()
        out[f"orig_{c}_count"] = key.map(counts).fillna(0).astype("float32")
        for cls_idx, cls_name in INT_TO_CLASS.items():
            hit = (tmp['y'] == cls_idx).astype(np.float32)
            rates = tmp.assign(hit=hit).groupby('key')['hit'].mean()
            out[f"orig_{c}_prior_{cls_name}"] = key.map(rates).fillna(float(prior[cls_idx])).astype("float32")
            if smooth > 0:
                smooth_tag = int(round(smooth))
                smooth_rates = ((rates * counts.astype(np.float32)) + smooth * float(prior[cls_idx])) / (counts.astype(np.float32) + smooth)
                out[f"orig_{c}_smooth{smooth_tag}_prior_{cls_name}"] = key.map(smooth_rates).fillna(float(prior[cls_idx])).astype("float32")
    return out


def build_base_matrix(train_df, test_df, orig_df, orig_y_arr):
    """
    Build the base feature matrix (everything except fold-dependent TE).
    Returns (X, X_test, cat_cols) — pandas DataFrames.
    This is called ONCE before the fold loop.
    """
    train_base = train_df.copy()
    test_base = test_df.copy()
    orig_base = orig_df.copy()
    train_base['_source'] = 'train'
    test_base['_source'] = 'test'
    orig_base['_source'] = 'orig'

    all_df = pd.concat([train_base, test_base, orig_base], axis=0, ignore_index=True)
    all_df, cat_cols = add_public_features(all_df)
    train_test_mask = all_df['_source'].isin(['train', 'test']).values

    all_df, artifact_cols = add_lowfreq_artifact_features(all_df)
    cat_cols += artifact_cols
    all_df, artifact_cols = add_color_artifact_features(all_df)
    cat_cols += artifact_cols
    all_df, qbin_cols = add_quantile_bin_features(all_df, train_test_mask)
    cat_cols += qbin_cols
    cat_cols = list(dict.fromkeys(cat_cols))
    cat_cols = [c for c in cat_cols if c in all_df.columns]

    # frequency features (fit on all rows incl orig — same as notebook)
    freq_cols = select_te_cols(all_df, cat_cols, max_card=5000 * 4)
    all_df = add_frequency_features(all_df, freq_cols, fit_mask=np.ones(len(all_df), dtype=bool))

    # original prior features
    orig_mask = (all_df['_source'] == 'orig').values
    prior_cols = select_te_cols(all_df, cat_cols, max_card=5000 * 2)
    all_df = add_original_prior_features(all_df, prior_cols, orig_mask, orig_y_arr, smooth=0.0)

    all_df['is_orig'] = (all_df['_source'] == 'orig').astype('int8')
    all_df['is_test'] = (all_df['_source'] == 'test').astype('int8')
    all_df = all_df.drop(columns=[c for c in ['id', '_source'] if c in all_df.columns])
    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    n_train = len(train_base)
    n_test = len(test_base)
    X = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:n_train + n_test].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]
    return X, X_test, cat_cols


def sorted_factorize_three(train_s, valid_s, test_s):
    """Label-encode three series with a consistent mapping."""
    combined = pd.concat([cat_key(train_s), cat_key(valid_s), cat_key(test_s)], ignore_index=True)
    cats = combined.drop_duplicates().sort_values(ignore_index=True)
    mapper = {v: i for i, v in enumerate(cats)}
    n_tr, n_va = len(train_s), len(valid_s)
    codes = combined.map(mapper).fillna(-1).astype(np.int32)
    return (
        codes.iloc[:n_tr].values,
        codes.iloc[n_tr:n_tr + n_va].values,
        codes.iloc[n_tr + n_va:].values,
    )


def add_fold_safe_te(X_train, y_train_arr, X_valid, X_test_fold, te_cols):
    """
    In-fold smoothed target encoder with TE_INNER_SPLITS inner folds.
    Fit ONLY on X_train / y_train_arr (no val/test labels ever used).
    Returns (X_train_out, X_valid_out, X_test_out, added_te_cols).
    """
    if not te_cols:
        return X_train, X_valid, X_test_fold, []
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()

    inner_skf = StratifiedKFold(n_splits=TE_INNER_SPLITS, shuffle=True, random_state=SEED + 177)
    fold_ids = np.empty(len(y_train_arr), dtype=np.int32)
    for fid, (_, va_idx) in enumerate(inner_skf.split(np.zeros(len(y_train_arr), dtype=np.int8), y_train_arr)):
        fold_ids[va_idx] = fid

    added = []
    for c in te_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        n_cats = max(tr_codes.max(), va_codes.max(), te_codes.max()) + 2

        for cls_idx, cls_name in INT_TO_CLASS.items():
            y_bin = (y_train_arr == cls_idx).astype(np.float32)
            global_mean = y_bin.mean()

            # OOF encoding for train
            tr_vals = np.full(len(tr_codes), global_mean, dtype=np.float32)
            for fid in range(TE_INNER_SPLITS):
                fit_mask = (fold_ids != fid)
                val_mask = (fold_ids == fid)
                fit_codes = tr_codes[fit_mask]
                fit_y = y_bin[fit_mask]
                # per-category stats from fit portion
                sum_y = np.zeros(n_cats, dtype=np.float32)
                cnt = np.zeros(n_cats, dtype=np.float32)
                np.add.at(sum_y, fit_codes[fit_codes >= 0], fit_y[fit_codes >= 0])
                np.add.at(cnt, fit_codes[fit_codes >= 0], 1.0)
                smoothed = (sum_y + TE_SMOOTH * global_mean) / (cnt + TE_SMOOTH)
                val_codes = tr_codes[val_mask]
                safe = np.where(val_codes >= 0, val_codes, 0)
                tr_vals[val_mask] = np.where(val_codes >= 0, smoothed[safe], global_mean)

            # full-train encoding for valid/test
            sum_y = np.zeros(n_cats, dtype=np.float32)
            cnt = np.zeros(n_cats, dtype=np.float32)
            np.add.at(sum_y, tr_codes[tr_codes >= 0], y_bin[tr_codes >= 0])
            np.add.at(cnt, tr_codes[tr_codes >= 0], 1.0)
            smoothed_full = (sum_y + TE_SMOOTH * global_mean) / (cnt + TE_SMOOTH)
            safe_va = np.where(va_codes >= 0, va_codes, 0)
            va_vals = np.where(va_codes >= 0, smoothed_full[safe_va], global_mean).astype(np.float32)
            safe_te = np.where(te_codes >= 0, te_codes, 0)
            te_vals = np.where(te_codes >= 0, smoothed_full[safe_te], global_mean).astype(np.float32)

            name = f"TE_{c}_{cls_name}"
            X_train[name] = tr_vals
            X_valid[name] = va_vals
            X_test_fold[name] = te_vals
            added.append(name)
    return X_train, X_valid, X_test_fold, added


def te_sources_needed_for_top_features(top_features, available_te_cols):
    needed = []
    top_set = set(top_features)
    for c in available_te_cols:
        if any(f"TE_{c}_{cls}" in top_set for cls in CLASSES):
            needed.append(c)
    return needed
