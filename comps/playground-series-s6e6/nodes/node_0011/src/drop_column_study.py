"""Drop-column feature-importance study on the single best GPU model (XGBoost-full, node_0011).

Drop-column (leave-one-out) importance:
  baseline_cv = 5-fold OOF balanced accuracy with all 28 features
  for each feature f:  cv_without_f = 5-fold OOF balanced accuracy with f removed
  importance(f) = baseline_cv - cv_without_f
A large positive value ⇒ f genuinely contributes; ≈0 ⇒ redundant given the rest;
NEGATIVE (CV improves when f is dropped) ⇒ f is noise/harmful → a prune candidate.

Uses node_0011's exact XGBoost config on GPU for speed. Identical 5-fold protocol for every
run, so any systematic optimism from early-stopping-on-val cancels in the deltas. Also records
XGBoost's native gain importance for comparison. Outputs JSON + CSV + a PNG plot.

NOTE on correlated features: drop-column UNDERSTATES importance of redundant features (drop u_g
and u_r covers for it). Read ≈0 as "redundant", and trust NEGATIVE values as the real prune signal.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NODE_SRC = Path(__file__).resolve().parent
COMP_DIR = NODE_SRC.parents[2]
_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
for p in (str(_r), str(COMP_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from clean import (cast_categoricals, add_color_features, add_extended_colors,
                   add_redshift_features, add_qso_colorbox, add_galactic_coords, feature_columns)

TARGET = "class"
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}


def engineer(df):
    for fn in (cast_categoricals, add_color_features, add_extended_colors,
               add_redshift_features, add_qso_colorbox, add_galactic_coords):
        df = fn(df)
    return df


def make_model():
    return XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=800, learning_rate=0.06,
        max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", enable_categorical=True, random_state=42,
        eval_metric="mlogloss", early_stopping_rounds=50, verbosity=0)


print("Loading + engineering …")
train = engineer(pd.read_csv(COMP_DIR / "data/train.csv"))
folds = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
ALL = feature_columns(train)
y = train[TARGET].map(L2I).astype(np.int32)
n = len(train)
fold_val = [np.asarray(f["val_idx"]) for f in folds]
print(f"  {len(ALL)} features, {n} rows")


def oof_balacc(cols):
    """5-fold OOF balanced accuracy using only `cols`."""
    Xc = train[cols]
    oof = np.empty(n, dtype=int)
    for val_idx in fold_val:
        tr_idx = np.setdiff1d(np.arange(n), val_idx)
        m = make_model()
        sw = compute_sample_weight("balanced", y.iloc[tr_idx].values)
        m.fit(Xc.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sw,
              eval_set=[(Xc.iloc[val_idx], y.iloc[val_idx])], verbose=False)
        oof[val_idx] = np.argmax(m.predict_proba(Xc.iloc[val_idx]), axis=1)
    return balanced_accuracy_score(y.values, oof)


print("Baseline (all features) …")
baseline = oof_balacc(ALL)
print(f"  baseline_cv = {baseline:.6f}")

# native gain importance from a single full-train fit
gm = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=800, learning_rate=0.06,
                   max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                   device="cuda", enable_categorical=True, random_state=42, verbosity=0)
gm.fit(train[ALL], y, sample_weight=compute_sample_weight("balanced", y.values))
gain_raw = gm.get_booster().get_score(importance_type="gain")   # keys may be feature names
gain = {f: float(gain_raw.get(f, 0.0)) for f in ALL}

print("Drop-column loop …")
rows = []
for j, f in enumerate(ALL):
    cv_wo = oof_balacc([c for c in ALL if c != f])
    imp = baseline - cv_wo
    rows.append({"feature": f, "cv_without": cv_wo, "drop_importance": imp, "gain": gain[f]})
    print(f"  [{j+1:2d}/{len(ALL)}] drop {f:<16} cv={cv_wo:.6f}  importance={imp:+.6f}")

df = pd.DataFrame(rows).sort_values("drop_importance", ascending=False).reset_index(drop=True)
df.to_csv(COMP_DIR / "feature_importance.csv", index=False)
(COMP_DIR / "feature_importance.json").write_text(json.dumps(
    {"model": "XGBoost-full (node_0011 cfg, GPU)", "metric": "Balanced Accuracy",
     "baseline_cv": baseline, "method": "drop-column 5-fold OOF",
     "features": df.to_dict("records")}, indent=2))

# ---- plot: drop-column (sorted) + native gain, all features ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
d = df.sort_values("drop_importance")
colors = ["#c0392b" if v < 0 else "#27ae60" for v in d["drop_importance"]]
ax1.barh(d["feature"], d["drop_importance"] * 1000, color=colors)
ax1.axvline(0, color="#333", lw=0.8)
ax1.set_xlabel("drop-column importance ×1000  (baseline_cv − cv_without)\nred = CV improves when dropped (prune candidate)")
ax1.set_title(f"Drop-column importance — XGBoost-full (5-fold OOF)\nbaseline balanced acc = {baseline:.6f}", fontsize=11)
ax1.grid(axis="x", alpha=0.3)

g = df.sort_values("gain")
ax2.barh(g["feature"], g["gain"], color="#2980b9")
ax2.set_xlabel("XGBoost native gain importance")
ax2.set_title("Native gain importance (full-train model)", fontsize=11)
ax2.grid(axis="x", alpha=0.3)
fig.suptitle("Feature importance — Predicting Stellar Class (28 engineered features)", fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(COMP_DIR / "feature_importance.png", dpi=130)
print(f"\nSaved: feature_importance.png / .csv / .json")
print("\n=== TOP positive (most helpful) ===")
print(df.head(8)[["feature", "drop_importance"]].to_string(index=False))
print("=== NEGATIVE (prune candidates) ===")
neg = df[df.drop_importance < 0]
print((neg[["feature", "drop_importance"]].to_string(index=False)) if len(neg) else "  none — every feature helps or is neutral")
print("Done.")
