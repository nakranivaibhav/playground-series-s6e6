"""Cheap single-base A/B: does adding cleaned SDSS17 rows to the TRAIN side of each
frozen fold lift a LightGBM's honest OOF balanced accuracy? Val folds stay PURE
playground rows (ext only augments train) so CV stays honest and comparable to our
node CVs. If this lifts even one base, the external-data lever is worth a full draft;
if it washes/hurts, abort it (drift AUC 0.909 said it would).

Three arms:
  A  train-only (baseline, our 8 shared features)
  B  train + cleaned ext (drop -9999 placeholder rows), all ext rows weight 1
  C  train + cleaned ext, ext rows DOWNWEIGHTED by adversarial propensity (domain
     adapt: rows that look more like real playground data count more)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parent
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}; NC = 3
FEATS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]  # shared with ext

train = pd.read_csv(COMP / "data/train.csv")
ext = pd.read_csv(COMP / "data/sdss17/star_classification.csv")
ext.columns = [c.strip() for c in ext.columns]
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train); y = train["class"].map(L2I).to_numpy()
Xtr = train[FEATS].apply(pd.to_numeric, errors="coerce").to_numpy()
fval = [np.asarray(f["val_idx"]) for f in folds]

# clean ext: drop the -9999 placeholder rows in u/g/z, align label
ext_clean = ext[(ext["u"] > -1000) & (ext["g"] > -1000) & (ext["z"] > -1000)].copy()
ext_clean = ext_clean[ext_clean["class"].isin(LAB)]
Xe = ext_clean[FEATS].apply(pd.to_numeric, errors="coerce").to_numpy()
ye = ext_clean["class"].map(L2I).to_numpy()
print(f"ext rows: {len(ext):,} → cleaned {len(Xe):,}")
print(f"ext class balance (cleaned): {pd.Series(ye).value_counts(normalize=True).round(3).to_dict()}")

# adversarial propensity for arm C: P(row looks like playground train) for each ext row
adv = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, n_jobs=-1, verbose=-1)
Xadv = np.vstack([Xtr[:80000], Xe[:80000]])
yadv = np.r_[np.ones(min(80000, n)), np.zeros(min(80000, len(Xe)))]  # 1 = playground
adv.fit(Xadv, yadv)
ext_w = adv.predict_proba(Xe)[:, 1]            # high = looks like playground
ext_w = ext_w / ext_w.mean()                   # normalize to mean 1
print(f"ext adversarial weight: mean={ext_w.mean():.3f} p10={np.percentile(ext_w,10):.3f} p90={np.percentile(ext_w,90):.3f}")

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))

def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=127,
                          class_weight="balanced", n_jobs=-1, verbose=-1)

def run(use_ext, weighted):
    pf = []
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        if use_ext:
            X = np.vstack([Xtr[tr], Xe]); yy = np.r_[y[tr], ye]
            w = np.r_[np.ones(len(tr)), ext_w] if weighted else None
        else:
            X, yy, w = Xtr[tr], y[tr], None
        m = lgbm(); m.fit(X, yy, sample_weight=w)
        pf.append(balacc(y[vi], m.predict(Xtr[vi])))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf)))

print(f"\n{'arm':40s} {'cv':>11s} {'sem':>9s} {'Δ vs A':>10s}", flush=True)
a_cv, a_sem = run(False, False); base = a_cv
print(f"{'A  train-only (8 shared feats)':40s} {a_cv:11.6f} {a_sem:9.6f} {0.0:+10.6f}", flush=True)
b_cv, b_sem = run(True, False)
print(f"{'B  + cleaned ext (weight 1)':40s} {b_cv:11.6f} {b_sem:9.6f} {b_cv-base:+10.6f}", flush=True)
c_cv, c_sem = run(True, True)
print(f"{'C  + ext, adversarial-downweighted':40s} {c_cv:11.6f} {c_sem:9.6f} {c_cv-base:+10.6f}", flush=True)
print("\nlift only counts if Δ > ~2·sem; otherwise the external-data lever is dead.", flush=True)
