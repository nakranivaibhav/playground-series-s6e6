"""Generate missing test_probs.npy for node_0003 (CatBoost).

node_0003's solution.py did not save test_probs.npy at build time.
This script retrains the same final model (same config, full train) and saves it.
Run once; node_0020/solution.py calls this if the file is absent.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE3_DIR = COMP / "nodes" / "node_0003"
# Add comp src for clean.py helpers
COMP_SRC = COMP / "src"
if str(COMP_SRC) not in sys.path:
    sys.path.insert(0, str(COMP_SRC))

from clean import cast_categoricals, add_color_features, feature_columns  # noqa

CAT_COLS = ["spectral_type", "galaxy_population"]


def main():
    out = NODE3_DIR / "test_probs.npy"
    if out.exists():
        print(f"test_probs.npy already exists for node_0003, skipping.")
        return

    print("Generating node_0003 test_probs.npy (full train refit) ...")
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")

    train = cast_categoricals(train)
    train = add_color_features(train)
    test  = cast_categoricals(test)
    test  = add_color_features(test)

    feat_cols = feature_columns(train)
    cat_feature_indices = [i for i, c in enumerate(feat_cols) if c in CAT_COLS]

    X = train[feat_cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str)
    y = train["class"].copy()

    X_test = test[feat_cols].copy()
    for c in CAT_COLS:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype(str)

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=6,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )
    full_pool = Pool(X, label=y, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, cat_features=cat_feature_indices)
    model.fit(full_pool)

    test_proba = model.predict_proba(test_pool)
    np.save(out, test_proba)
    print(f"  saved test_probs.npy {test_proba.shape} -> {out}")


if __name__ == "__main__":
    main()
