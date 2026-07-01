"""Decisive cheap A/B for the CdeOtte lever: do ORIGINAL-SDSS17 PRIOR features
(P(class|color-bin-key) computed on the original data, mapped onto train/test) +
curvature/slope FE lift a single LightGBM under our honest frozen folds?

This is the drift-ROBUST way to use external data (not row-concat, which we proved
hurts): orig provides relative class structure per color-bin, transferring through
the distribution shift. orig priors are computed on ALL of orig (orig labels are
disjoint from our train/val, so no fold leakage). Curvature/slope FE is stateless.

Arms:
 A  our 8 shared raw features (baseline, matches sdss17_ab_probe arm A 0.9644)
 B  + curvature/slope/mag-stat FE (stateless, richer FE)
 C  + original-prior features (P(class|key) from orig, several bin keys)
 D  B + C  (the full CdeOtte-style base)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parent
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}; NC = 3
RAW = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]

train = pd.read_csv(COMP / "data/train.csv")
ext = pd.read_csv(COMP / "data/sdss17/star_classification.csv"); ext.columns = [c.strip() for c in ext.columns]
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train); y = train["class"].map(L2I).to_numpy()
fval = [np.asarray(f["val_idx"]) for f in folds]

# clean orig (drop -9999 placeholders), keep shared cols + label
ext = ext[(ext["u"] > -1000) & (ext["g"] > -1000) & (ext["z"] > -1000)]
ext = ext[ext["class"].isin(LAB)].copy()
yo = ext["class"].map(L2I).to_numpy()
prior = np.bincount(yo, minlength=NC) / len(yo)

def color_fe(df):
    out = pd.DataFrame(index=df.index)
    for c in RAW: out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    band = out[BANDS].to_numpy("float32")
    out["mag_mean"] = band.mean(1); out["mag_std"] = band.std(1)
    out["mag_min"] = band.min(1); out["mag_max"] = band.max(1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["mag_argmin"] = band.argmin(1).astype("float32"); out["mag_argmax"] = band.argmax(1).astype("float32")
    out["blue_slope"] = (out["g"] - out["u"]) + (out["r"] - out["g"])
    out["red_slope"] = (out["i"] - out["r"]) + (out["z"] - out["i"])
    out["curv_ugr"] = out["u"] - 2 * out["g"] + out["r"]
    out["curv_gri"] = out["g"] - 2 * out["r"] + out["i"]
    out["curv_riz"] = out["r"] - 2 * out["i"] + out["z"]
    out["log1p_rs"] = np.log1p(np.clip(out["redshift"], 0, None))
    for a, b in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i")]:
        out[f"{a}_{b}"] = out[a] - out[b]
    return out

# bin keys for original-prior features (floor/quantile bins of colors + redshift)
def keys_of(df):
    g_r = (df["g"] - df["r"]); u_r = (df["u"] - df["r"]); rs = df["redshift"]
    K = pd.DataFrame(index=df.index)
    K["rs_tenth"] = np.floor(rs * 10).astype("int32").clip(-20, 200)
    K["gr_half"] = np.floor(g_r * 2).astype("int32").clip(-40, 40)
    K["ur_half"] = np.floor(u_r * 2).astype("int32").clip(-40, 40)
    K["gr_x_rs"] = K["gr_half"].astype(str) + "_" + K["rs_tenth"].astype(str)
    K["ur_x_rs"] = K["ur_half"].astype(str) + "_" + K["rs_tenth"].astype(str)
    return K

SMOOTH = 16.0
def orig_priors(tr_fe, va_fe, ext_df):
    """class-rate per key computed on ORIG only; mapped to train/val. drift-robust, no fold leak."""
    ek = keys_of(ext_df)
    feats_tr, feats_va = {}, {}
    for col in ["rs_tenth", "gr_half", "ur_half", "gr_x_rs", "ur_x_rs"]:
        df = pd.DataFrame({"key": ek[col].values, "y": yo})
        cnt = df.groupby("key").size()
        for ci, cn in enumerate(LAB):
            hit = (df["y"] == ci).astype("float32")
            rate = df.assign(h=hit).groupby("key")["h"].mean()
            sm = (rate * cnt + SMOOTH * prior[ci]) / (cnt + SMOOTH)
            feats_tr[f"op_{col}_{cn}"] = tr_fe["__"+col].map(sm).fillna(prior[ci]).astype("float32").values
            feats_va[f"op_{col}_{cn}"] = va_fe["__"+col].map(sm).fillna(prior[ci]).astype("float32").values
    return pd.DataFrame(feats_tr, index=tr_fe.index), pd.DataFrame(feats_va, index=va_fe.index)

FE = color_fe(train)
KE = keys_of(train)
for c in KE.columns: FE["__"+c] = KE[c].values  # carry keys alongside for mapping

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))
def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=127,
                          class_weight="balanced", n_jobs=-1, verbose=-1)

RAWFE = [c for c in FE.columns if not c.startswith("__")]

def run(arm):
    pf = []
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        Xtr, Xva = FE.iloc[tr][RAWFE].copy(), FE.iloc[vi][RAWFE].copy()
        if arm in ("C", "D"):
            ptr, pva = orig_priors(FE.iloc[tr], FE.iloc[vi], ext)
            Xtr = pd.concat([Xtr.reset_index(drop=True), ptr.reset_index(drop=True)], axis=1)
            Xva = pd.concat([Xva.reset_index(drop=True), pva.reset_index(drop=True)], axis=1)
        if arm == "A":  # raw only: drop the FE-added cols
            keep = RAW
            Xtr, Xva = FE.iloc[tr][keep], FE.iloc[vi][keep]
        if arm == "C":  # raw + priors only (no curvature FE)
            ptr, pva = orig_priors(FE.iloc[tr], FE.iloc[vi], ext)
            Xtr = pd.concat([FE.iloc[tr][RAW].reset_index(drop=True), ptr.reset_index(drop=True)], axis=1)
            Xva = pd.concat([FE.iloc[vi][RAW].reset_index(drop=True), pva.reset_index(drop=True)], axis=1)
        m = lgbm(); m.fit(Xtr, y[tr])
        pf.append(balacc(y[vi], m.predict(Xva)))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf)))

print(f"{'arm':40s} {'cv':>11s} {'sem':>9s} {'Δ vs A':>10s}", flush=True)
base = None
for arm, name in [("A","A  raw 8 feats"),("B","B  + curvature/slope FE"),
                  ("C","C  + original-prior feats only"),("D","D  curvature FE + original priors")]:
    cv, sem = run(arm)
    if base is None: base = cv
    print(f"{name:40s} {cv:11.6f} {sem:9.6f} {cv-base:+10.6f}", flush=True)
print("\nour included LightGBM base (rich FE, node_0030) = 0.96695; if D approaches/beats that, "
      "the original-prior lever is real → port into a full base + re-stack.", flush=True)
