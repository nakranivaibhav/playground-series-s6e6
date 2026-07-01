"""Cheap drift check: is the real SDSS17 dataset close enough to the playground
train that concatenating it as extra rows could help? If badly shifted, abort.

Three reads:
 1. column overlap (can we even align features + the class label?)
 2. per-feature distribution distance (KS statistic) on shared numeric columns
 3. adversarial AUC: a GBDT told to tell train-origin from SDSS17-origin. AUC~0.5
    = indistinguishable (great); AUC~1.0 = trivially separable (drift, risky).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

COMP = Path(__file__).resolve().parent
train = pd.read_csv(COMP / "data/train.csv")
ext = pd.read_csv(COMP / "data/sdss17/star_classification.csv")

print("=== train cols ===", list(train.columns))
print("=== sdss17 cols ===", list(ext.columns))
print(f"train n={len(train):,}  ext n={len(ext):,}")

# normalize names (sdss17 uses upper/lower variants); find shared numeric features
ext.columns = [c.strip() for c in ext.columns]
tcols = set(train.columns); ecols = set(ext.columns)
shared = sorted((tcols & ecols) - {"class", "id"})
print("=== shared columns (besides class/id) ===", shared)

# class label distribution
print("\n=== class balance ===")
print("train:", train["class"].value_counts(normalize=True).round(4).to_dict())
ccol = "class" if "class" in ext.columns else next((c for c in ext.columns if ext[c].dtype == object), None)
if ccol:
    print(f"ext[{ccol}]:", ext[ccol].value_counts(normalize=True).round(4).to_dict())

if not shared:
    print("\nNO shared numeric features — cannot align. ABORT external-data lever.")
    raise SystemExit

# KS per shared feature
print("\n=== per-feature KS (0=identical, 1=disjoint) ===")
for c in shared:
    a = pd.to_numeric(train[c], errors="coerce").dropna()
    b = pd.to_numeric(ext[c], errors="coerce").dropna()
    if len(a) and len(b):
        ks = ks_2samp(a.sample(min(len(a), 50000), random_state=0),
                      b.sample(min(len(b), 50000), random_state=0)).statistic
        print(f"  {c:14s} KS={ks:.3f}  train[{a.mean():.3f}±{a.std():.3f}] ext[{b.mean():.3f}±{b.std():.3f}]")

# adversarial AUC on shared features
Xa = train[shared].apply(pd.to_numeric, errors="coerce")
Xb = ext[shared].apply(pd.to_numeric, errors="coerce")
nb = min(len(Xa), len(Xb), 80000)
Xa = Xa.sample(nb, random_state=0); Xb = Xb.sample(nb, random_state=0)
X = pd.concat([Xa, Xb], ignore_index=True).fillna(-999)
yo = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, n_jobs=-1, verbose=-1)
p = cross_val_predict(clf, X, yo, cv=3, method="predict_proba")[:, 1]
auc = roc_auc_score(yo, p)
print(f"\n=== adversarial AUC (train-origin vs sdss17-origin) = {auc:.4f} ===")
print("  ~0.5 indistinguishable (safe to concat) | >0.8 strong drift (risky) | "
      ">0.95 trivially separable (abort/needs domain-adapt)")
