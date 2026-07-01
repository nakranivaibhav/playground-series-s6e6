"""Regenerate node_0004's (XGBoost) test probabilities — all-train fit only.

Replicates node_0004's pipeline (same features, enable_categorical, balanced
sample weights, same params) and fits once on ALL train, saving test_probs.npy
in LABEL_ORDER [GALAXY,QSO,STAR] (XGB predict_proba cols are 0,1,2 = that order).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

NODE_SRC = Path(__file__).resolve().parent
COMP_DIR = NODE_SRC.parent.parent.parent
sys.path.insert(0, str(COMP_DIR / "src"))
from clean import cast_categoricals, add_color_features, feature_columns  # noqa

LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}
tr = add_color_features(cast_categoricals(pd.read_csv(COMP_DIR / "data/train.csv")))
te = add_color_features(cast_categoricals(pd.read_csv(COMP_DIR / "data/test.csv")))
fc = feature_columns(tr)
y_enc = tr["class"].map(L2I).astype(int)
m = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=800,
                  learning_rate=0.06, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", enable_categorical=True, n_jobs=-1,
                  random_state=42, verbosity=0)
m.fit(tr[fc], y_enc, sample_weight=compute_sample_weight("balanced", y_enc.values))
proba = m.predict_proba(te[fc])          # columns 0,1,2 == LABEL_ORDER
np.save(NODE_SRC.parent / "test_probs.npy", proba.astype(np.float64))
print("node_0004 test_probs.npy saved", proba.shape)
