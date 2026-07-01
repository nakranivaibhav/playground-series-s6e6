"""Unit tests for clean.py — evidence that each transform is correct & leak-safe."""
import numpy as np
import pandas as pd
import pytest

from clean import (
    add_color_features,
    add_extended_colors,
    add_redshift_features,
    add_qso_colorbox,
    add_galactic_coords,
    add_positional_features,
    cast_categoricals,
    feature_columns,
    CAT_COLS,
    ID_COL,
    TARGET_COL,
)


def _toy(n=50, with_target=True, seed=0):
    rng = np.random.default_rng(seed)
    d = {
        "id": np.arange(n),
        "alpha": rng.uniform(0, 360, n),
        "delta": rng.uniform(-18, 80, n),
        "u": rng.uniform(14, 28, n),
        "g": rng.uniform(13, 27, n),
        "r": rng.uniform(12, 25, n),
        "i": rng.uniform(11, 28, n),
        "z": rng.uniform(11, 27, n),
        "redshift": rng.uniform(-0.01, 7, n),
        "spectral_type": rng.choice(["M", "A/F", "G/K", "O/B"], n),
        "galaxy_population": rng.choice(["Red_Sequence", "Blue_Cloud"], n),
    }
    if with_target:
        d["class"] = rng.choice(["GALAXY", "QSO", "STAR"], n)
    return pd.DataFrame(d)


# --- cast_categoricals ---

def test_cast_preserves_rows_and_values():
    df = _toy()
    out = cast_categoricals(df)
    assert len(out) == len(df)                          # row count preserved
    for c in CAT_COLS:
        assert str(out[c].dtype) == "category"
        # values themselves unchanged (only dtype)
        assert list(out[c].astype(str)) == list(df[c].astype(str))


def test_cast_no_new_nans():
    df = _toy()
    out = cast_categoricals(df)
    assert out.isna().sum().sum() == df.isna().sum().sum() == 0


def test_cast_train_test_same_columns():
    tr, te = _toy(with_target=True), _toy(with_target=False, seed=1)
    otr, ote = cast_categoricals(tr), cast_categoricals(te)
    # identical transform on shared columns: same dtypes out
    shared = [c for c in ote.columns if c in otr.columns]
    for c in shared:
        assert otr[c].dtype == ote[c].dtype


# --- add_color_features ---

def test_color_values_correct():
    df = _toy()
    out = add_color_features(df)
    assert np.allclose(out["u_g"], df["u"] - df["g"])
    assert np.allclose(out["g_r"], df["g"] - df["r"])
    assert np.allclose(out["r_i"], df["r"] - df["i"])
    assert np.allclose(out["i_z"], df["i"] - df["z"])
    assert np.allclose(out["u_z"], df["u"] - df["z"])


def test_color_preserves_rows_and_adds_five_cols():
    df = _toy()
    out = add_color_features(df)
    assert len(out) == len(df)
    assert set(out.columns) - set(df.columns) == {"u_g", "g_r", "r_i", "i_z", "u_z"}
    assert out.isna().sum().sum() == 0                  # no new NaNs


def test_color_is_stateless_fit_inside_fold_safe():
    """Row-wise transform must be identical whether applied to a slice alone or
    as part of the whole frame — proves it learns nothing from other rows."""
    df = _toy(n=100)
    whole = add_color_features(df)
    part_idx = df.index[:30]
    part = add_color_features(df.loc[part_idx])
    for c in ["u_g", "g_r", "r_i", "i_z", "u_z"]:
        assert np.allclose(whole.loc[part_idx, c].values, part[c].values)


def test_color_noop_without_mags():
    df = _toy()[["id", "redshift", "class"]]
    out = add_color_features(df)
    assert list(out.columns) == list(df.columns)        # unchanged copy


# --- new research-seeded feature helpers ---

def test_extended_colors_values_and_stateless():
    df = _toy(n=80)
    out = add_extended_colors(df)
    assert np.allclose(out["u_r"], df["u"] - df["r"])
    assert np.allclose(out["g_i"], df["g"] - df["i"])
    assert np.allclose(out["c_ug_gr"], (df["u"] - df["g"]) - (df["g"] - df["r"]))
    assert len(out) == len(df) and out.isna().sum().sum() == 0
    # stateless: a slice transforms identically alone vs in the whole frame
    part = add_extended_colors(df.iloc[:20])
    assert np.allclose(out["u_i"].values[:20], part["u_i"].values)


def test_redshift_features_flags_and_log():
    df = pd.DataFrame({"redshift": [-0.001, 0.0, 0.001, 0.5, 1.5, 7.0]})
    out = add_redshift_features(df)
    assert np.allclose(out["log1p_redshift"], np.log1p(np.clip(df["redshift"], 0, None)))
    assert list(out["is_star_z"]) == [1, 1, 1, 0, 0, 0]      # z<0.0025
    assert list(out["is_highz"]) == [0, 0, 0, 0, 1, 1]       # z>1.0
    assert out.isna().sum().sum() == 0


def test_qso_colorbox_logic():
    # u-g<0.6 AND g-r>0 -> qso_box; u-g<0.4 -> uv_excess
    df = pd.DataFrame({"u": [10.0, 10.0, 10.0], "g": [9.5, 9.2, 8.0], "r": [9.4, 9.3, 7.0]})
    out = add_qso_colorbox(df)
    ug = df["u"] - df["g"]
    gr = df["g"] - df["r"]
    assert list(out["qso_box"]) == [int((a < 0.6) and (b > 0)) for a, b in zip(ug, gr)]
    assert list(out["uv_excess"]) == [int(a < 0.4) for a in ug]


def test_galactic_coords_known_point_and_range():
    # The north galactic pole (RA=192.85948, Dec=27.12825) must map to b≈+90.
    out = add_galactic_coords(pd.DataFrame({"alpha": [192.85948], "delta": [27.12825]}))
    assert abs(out["gal_b"].iloc[0] - 90.0) < 1e-3
    # general sanity on a spread of points: b in [-90,90], l in [0,360)
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"alpha": rng.uniform(0, 360, 200), "delta": rng.uniform(-18, 80, 200)})
    g = add_galactic_coords(big)
    assert g["gal_b"].between(-90, 90).all()
    assert g["gal_l"].between(0, 360).all() and len(g) == len(big)
    # stateless: identical row-wise
    part = add_galactic_coords(big.iloc[:50])
    assert np.allclose(g["gal_b"].values[:50], part["gal_b"].values)


def test_new_helpers_noop_without_inputs():
    bare = pd.DataFrame({"id": [1, 2], "class": ["STAR", "QSO"]})
    for fn in (add_extended_colors, add_redshift_features, add_qso_colorbox,
               add_galactic_coords, add_positional_features):
        out = fn(bare)
        assert list(out.columns) == list(bare.columns)   # unchanged copy


# --- add_positional_features (leak-safe positional) ---

def test_positional_unit_circle_and_sphere():
    df = _toy(n=120)
    out = add_positional_features(df)
    # sin/cos of RA lie on the unit circle; cartesian on the unit sphere
    assert np.allclose(out["alpha_sin"] ** 2 + out["alpha_cos"] ** 2, 1.0)
    assert np.allclose(out["sx"] ** 2 + out["sy"] ** 2 + out["sz"] ** 2, 1.0)
    # sz is exactly sin(declination)
    assert np.allclose(out["sz"], np.sin(np.radians(df["delta"])))
    assert len(out) == len(df)


def test_positional_interactions_and_no_label_use():
    df = _toy(n=60)
    out = add_positional_features(df)
    assert np.allclose(out["delta_x_redshift"], df["delta"] * df["redshift"])
    assert np.allclose(out["delta_x_logz"], df["delta"] * np.log1p(np.clip(df["redshift"], 0, None)))
    # the transform must NOT depend on the target at all (leak-safety): permuting the
    # label leaves every engineered positional column byte-identical.
    shuffled = df.copy()
    shuffled["class"] = shuffled["class"].sample(frac=1, random_state=1).values
    out2 = add_positional_features(shuffled)
    for c in ["alpha_sin", "alpha_cos", "sx", "sy", "sz", "delta_x_redshift", "delta_x_logz"]:
        assert np.allclose(out[c].values, out2[c].values)
    assert list(out["sky_cell"].astype(str)) == list(out2["sky_cell"].astype(str))


def test_positional_sky_cell_is_category_and_stateless():
    df = _toy(n=100, seed=3)
    out = add_positional_features(df)
    assert str(out["sky_cell"].dtype) == "category"
    assert out.isna().sum().sum() == 0
    # row-wise stateless: a slice yields identical cell ids / coords as the whole frame
    part = add_positional_features(df.iloc[:25])
    assert list(out["sky_cell"].astype(str)[:25]) == list(part["sky_cell"].astype(str))
    assert np.allclose(out["sx"].values[:25], part["sx"].values)


def test_positional_ra_wrap_continuity():
    # RA 359.9° and 0.1° are ~neighbours on the sky; sin/cos must reflect that
    df = pd.DataFrame({"alpha": [359.9, 0.1], "delta": [10.0, 10.0]})
    out = add_positional_features(df)
    assert abs(out["alpha_cos"].iloc[0] - out["alpha_cos"].iloc[1]) < 1e-3
    assert abs(out["alpha_sin"].iloc[0] - out["alpha_sin"].iloc[1]) < 1e-2


# --- feature_columns ---

def test_feature_columns_excludes_id_and_target():
    df = add_color_features(cast_categoricals(_toy()))
    feats = feature_columns(df)
    assert ID_COL not in feats and TARGET_COL not in feats
    assert "redshift" in feats and "u_g" in feats
