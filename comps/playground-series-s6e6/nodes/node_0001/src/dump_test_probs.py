"""Regenerate node_0001's test probabilities (all-train fit only, no CV).

node_0001 computed but discarded test probabilities; the blend needs them. This
replicates node_0001's exact pipeline (same features, same LGBM params) and fits
once on ALL train, saving test_probs.npy in LABEL_ORDER [GALAXY,QSO,STAR].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

NODE_SRC = Path(__file__).resolve().parent
COMP_DIR = NODE_SRC.parent.parent.parent
sys.path.insert(0, str(COMP_DIR / "src"))
from clean import cast_categoricals, add_color_features, feature_columns  # noqa

LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}
tr = add_color_features(cast_categoricals(pd.read_csv(COMP_DIR / "data/train.csv")))
te = add_color_features(cast_categoricals(pd.read_csv(COMP_DIR / "data/test.csv")))
fc = feature_columns(tr)
m = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500,
                   learning_rate=0.05, num_leaves=63, n_jobs=-1,
                   class_weight="balanced", random_state=42, verbosity=-1)
m.fit(tr[fc], tr["class"])
proba = m.predict_proba(te[fc])
co = list(m.classes_)
out = np.zeros((len(te), 3))
for lbl in LABEL_ORDER:
    out[:, L2I[lbl]] = proba[:, co.index(lbl)]
np.save(NODE_SRC.parent / "test_probs.npy", out)
print("node_0001 test_probs.npy saved", out.shape)
