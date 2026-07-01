"""
node_0044 — xgb-v5 zoo port (canary base) [early-stop on balanced-error]
Port of cdeotte's xgb-v5-for-s6e6.py (cudf/cuml) to CPU pandas + GPU XGBoost.
Uses OUR frozen folds.json (5-fold StratifiedKFold).
All encoders (TE, frequency) fit INSIDE each train fold. Original-prior
features computed on sdss17 data only.

CHANGE vs original: early-stopping now uses a custom balanced-error eval metric
(1 - balanced_accuracy, minimized) instead of mlogloss, so XGBoost picks the
number of trees that maximises OUR competition metric.
"""
import gc
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# allow importing fs_zoo from this directory
sys.path.insert(0, str(Path(__file__).parent))
from fs_zoo import (
    CLASSES, CLASS_TO_INT, INT_TO_CLASS, TOP_FEATURES, BANDS,
    cat_key, build_base_matrix, add_fold_safe_te,
    te_sources_needed_for_top_features, select_te_cols,
)

import xgboost as xgb

SEED = 42
N_SPLITS = 5
TARGET = 'class'
ID_COL = 'id'
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 1.0
TE_MAX_CARDINALITY = 5000

BASE = Path(__file__).parent.parent  # nodes/node_0044/
COMP = BASE.parent.parent            # comps/playground-series-s6e6/
DATA = COMP / 'data'
OUT = BASE


def balanced_error_eval(y_true, y_score, sample_weight=None):
    """Custom XGBoost sklearn-API eval metric: balanced classification error = 1 - balanced_accuracy.
    XGBoost sklearn wrapper calls feval(y_true, y_score, sample_weight=...).
    y_score shape: (N, num_class) predicted probabilities.
    Returns a scalar; XGBoost minimises (smaller is better for early stopping).
    balanced_accuracy_score already handles class-level averaging so sample_weight is ignored here.
    """
    preds = np.argmax(y_score, axis=1)
    score = balanced_accuracy_score(y_true.astype(np.int32), preds)
    return 1.0 - score


def make_xgb_params(seed):
    return dict(
        objective='multi:softprob',
        num_class=len(CLASSES),
        tree_method='hist',
        device='cuda',
        learning_rate=0.012,
        n_estimators=7000,
        early_stopping_rounds=180,
        max_depth=0,
        max_leaves=72,
        grow_policy='lossguide',
        max_bin=960,
        min_child_weight=10,
        gamma=0.2,
        reg_alpha=0.30,
        reg_lambda=4.0,
        subsample=0.82,
        colsample_bytree=0.74,
        colsample_bylevel=0.86,
        random_state=seed,
        n_jobs=4,
        eval_metric=balanced_error_eval,
        verbosity=0,
    )


def class_weights(y_arr):
    counts = np.bincount(y_arr.astype(np.int32), minlength=len(CLASSES)).astype(np.float32)
    w_per_class = len(y_arr) / (len(CLASSES) * np.maximum(counts, 1.0))
    weights = w_per_class[y_arr.astype(np.int32)]
    if CLASS_WEIGHT_POWER != 1.0:
        weights = np.power(weights, CLASS_WEIGHT_POWER)
    return weights.astype(np.float32)


def sorted_factorize(train_s, valid_s, test_s):
    from fs_zoo import sorted_factorize_three
    return sorted_factorize_three(train_s, valid_s, test_s)


def encode_model_categories(X_train, X_valid, X_test_fold, model_cat_cols):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    for c in model_cat_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize(X_train[c], X_valid[c], X_test_fold[c])
        X_train[c] = tr_codes
        X_valid[c] = va_codes
        X_test_fold[c] = te_codes
    return X_train, X_valid, X_test_fold


def main():
    print('Loading data...')
    train = pd.read_csv(DATA / 'train.csv')
    test = pd.read_csv(DATA / 'test.csv')
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    orig_raw = pd.read_csv(DATA / 'sdss17' / 'star_classification.csv')

    # Clean original: drop -9999 placeholder rows
    for c in ['u', 'g', 'z']:
        if c in orig_raw.columns:
            orig_raw = orig_raw[orig_raw[c] != -9999.0]
    orig_raw = orig_raw.reset_index(drop=True)

    # Build spectral_type / galaxy_population for orig if missing
    from fs_zoo import spectral_type_from_gr, galaxy_population_from_ur
    if 'spectral_type' not in orig_raw.columns:
        orig_raw['spectral_type'] = spectral_type_from_gr(orig_raw['r'] - orig_raw['g'])
    if 'galaxy_population' not in orig_raw.columns:
        orig_raw['galaxy_population'] = galaxy_population_from_ur(orig_raw['u'] - orig_raw['r'])
    orig_raw['spectral_type'] = orig_raw['spectral_type'].astype(str).fillna('__NA__')
    orig_raw['galaxy_population'] = orig_raw['galaxy_population'].astype(str).fillna('__NA__')

    # Prepare numeric columns
    for c in ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']:
        orig_raw[c] = pd.to_numeric(orig_raw[c], errors='coerce').astype('float32')
        train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
        test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
    train['spectral_type'] = train['spectral_type'].astype(str).fillna('__NA__')
    train['galaxy_population'] = train['galaxy_population'].astype(str).fillna('__NA__')
    test['spectral_type'] = test['spectral_type'].astype(str).fillna('__NA__')
    test['galaxy_population'] = test['galaxy_population'].astype(str).fillna('__NA__')

    keep_orig = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift',
                 'spectral_type', 'galaxy_population', TARGET]
    orig_df = orig_raw[[c for c in keep_orig if c in orig_raw.columns]].copy()
    orig_y_arr = orig_df[TARGET].map(CLASS_TO_INT).values.astype(np.int8)
    orig_df = orig_df.drop(columns=[TARGET])

    y = train[TARGET].map(CLASS_TO_INT).values.astype(np.int8)
    test_ids = test[ID_COL].values

    train_base = train.drop(columns=[TARGET, ID_COL], errors='ignore')
    test_base = test.drop(columns=[ID_COL], errors='ignore')

    print(f'train={train_base.shape}, test={test_base.shape}, orig={orig_df.shape}')

    # Load frozen folds
    with open(COMP / 'folds.json') as f:
        folds_data = json.load(f)
    folds = folds_data['folds']  # list of {fold, val_idx}

    print('Building base feature matrix (stateless, runs once)...')
    X, X_test, cat_cols = build_base_matrix(train_base, test_base, orig_df, orig_y_arr)
    print(f'X={X.shape}, X_test={X_test.shape}, cat_cols={len(cat_cols)}')

    # Determine TE columns needed
    available_te_cols = select_te_cols(X, cat_cols, max_card=TE_MAX_CARDINALITY)
    TE_COLS = te_sources_needed_for_top_features(TOP_FEATURES, available_te_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in set(TOP_FEATURES)]
    print(f'TE sources: {len(TE_COLS)}, model cat cols: {len(MODEL_CAT_COLS)}')

    # OOF arrays
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_pred_sum = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)

    fold_scores = []

    for fold_info in folds:
        fold = fold_info['fold']
        va_idx = np.array(fold_info['val_idx'], dtype=np.int64)
        all_idx = np.arange(len(X), dtype=np.int64)
        tr_mask = np.ones(len(X), dtype=bool)
        tr_mask[va_idx] = False
        tr_idx = all_idx[tr_mask]

        print(f'\n===== Fold {fold}/{N_SPLITS} | train={len(tr_idx)} val={len(va_idx)} =====')
        fold_seed = SEED + fold * 100

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_va = y[va_idx]
        X_te = X_test.copy()

        # In-fold TE (fit ONLY on train fold)
        X_tr, X_va, X_te, added_te = add_fold_safe_te(X_tr, y_tr, X_va, X_te, TE_COLS)
        print(f'TE features added: {len(added_te)}')

        # Encode raw cat cols
        X_tr, X_va, X_te = encode_model_categories(X_tr, X_va, X_te, MODEL_CAT_COLS)

        # Select TOP_FEATURES
        missing = [c for c in TOP_FEATURES if c not in X_tr.columns]
        if missing:
            print(f'  Missing {len(missing)} top features: {missing[:5]}')
        features = [c for c in TOP_FEATURES if c in X_tr.columns]
        print(f'  Using {len(features)} features')

        X_tr_np = X_tr[features].astype(np.float32).values
        X_va_np = X_va[features].astype(np.float32).values
        X_te_np = X_te[features].astype(np.float32).values

        sw = class_weights(y_tr) if USE_CLASS_WEIGHTS else None
        vw = class_weights(y_va) if USE_CLASS_WEIGHTS else None

        model = xgb.XGBClassifier(**make_xgb_params(fold_seed))
        model.fit(
            X_tr_np, y_tr,
            sample_weight=sw,
            eval_set=[(X_va_np, y_va)],
            sample_weight_eval_set=[vw] if vw is not None else None,
            verbose=250,
        )

        va_probs = model.predict_proba(X_va_np).astype(np.float32)
        te_probs = model.predict_proba(X_te_np).astype(np.float32)
        oof[va_idx] = va_probs
        test_pred_sum += te_probs / N_SPLITS

        fold_score = balanced_accuracy_score(y_va, np.argmax(va_probs, axis=1))
        best_iter = getattr(model, 'best_iteration', None)
        print(f'fold {fold} balanced_accuracy={fold_score:.8f} | best_iteration={best_iter}')
        fold_scores.append(float(fold_score))

        del model, X_tr, X_va, X_te, X_tr_np, X_va_np, X_te_np
        gc.collect()

    # CV summary
    cv_mean = float(np.mean(fold_scores))
    cv_sem = float(np.std(fold_scores, ddof=1) / np.sqrt(N_SPLITS))
    oof_cv = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    for i, s in enumerate(fold_scores, 1):
        print(f'fold={i} score={s:.8f}')
    print(f'mean_fold_cv={cv_mean:.8f} sem={cv_sem:.8f}')
    print(f'cv={oof_cv:.8f}')

    # Save outputs
    np.save(OUT / 'oof.npy', oof)
    np.save(OUT / 'test_probs.npy', test_pred_sum)

    pred_labels = [INT_TO_CLASS[i] for i in np.argmax(test_pred_sum, axis=1)]
    sub = sample.copy()
    sub[TARGET] = pred_labels
    sub.to_csv(OUT / 'submission.csv', index=False)

    # features.txt
    features_final = [c for c in TOP_FEATURES if c in X.columns or c.startswith('TE_')]
    with open(OUT / 'features.txt', 'w') as f:
        f.write('\n'.join(features_final))

    print(f'Saved oof.npy {oof.shape}, test_probs.npy {test_pred_sum.shape}, submission.csv {sub.shape}')
    print('DONE')


if __name__ == '__main__':
    main()
