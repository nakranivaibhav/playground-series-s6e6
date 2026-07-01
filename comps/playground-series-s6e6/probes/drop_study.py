"""probe: drop-model study on the champion (node_0091) FULL pool.

GOAL (human-directed): rank every base in the champion's stacking pool by its
CONTRIBUTION, so we know which models pull the most weight → the "top models"
to bag (train more seed/bootstrap variants of) and add back to the pool.

Method:
  - Rebuild the EXACT node_0091 FULL pool (bank-17 + FT-T + 36 TIGHT + 9 WEAK
    in-house bases), loader copied VERBATIM from nodes/node_0099/src/solution.py.
  - Each base = a contiguous 3-col block of clipped log-probs.
  - Baseline: fit the champion meta (balanced multinomial LogReg, C=0.003)
    5-fold OOF over the full pool → cv_full (must ≈ 0.970355).
  - Leave-One-Base-Out: for each base, drop its 3 cols, refit 5-fold OOF at
    C=0.003 → cv_without. delta = cv_full - cv_without  (POSITIVE = base helps;
    dropping it hurts CV). Rank by delta desc → importance map.
  - 5-fold (not 1-fold): deltas are ~1e-4 scale, single-fold noise ~5e-4 would
    bury them. The 5-fold OOF mean is the only way to get signal.
  - Also report |coef| ranking from the single full-pool fit (free cross-check;
    biased by correlation but cheap — agreement with LOO = confident top base).

NO submission, NO retraining of bases — pure diagnostic on saved OOF.
Outputs: probes/drop_study_ranking.csv + a printed table.
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
OUT_CSV = COMP / "probes/drop_study_ranking.csv"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
NC = 3
C_FIXED = 0.003  # champion's winning C

TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]


# ---- helpers (verbatim from node_0099) ------------------------------------
def logp(a):
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a):
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s

def score_fn(y_true, y_pred):
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def rd(path, nr):
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c):
        return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c):
        return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3:
        return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float)
    return v.reshape(nr, 3)

def load_ext_csv(path, nr):
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


def loo_one(i, OOF_full, y, fval, n_bases, C):
    """Drop base i (its 3 cols) and return (i, 5-fold OOF cv). Module-level so
    joblib auto-memmaps the big OOF_full arg (shared read-only across workers)."""
    keep = np.r_[0:3 * i, 3 * i + 3:3 * n_bases]
    cv_i, _ = arm_cv(OOF_full[:, keep], y, fval, C)
    return i, cv_i


def arm_cv(OOF, y, fval, C):
    """5-fold OOF mean BA for the champion meta at fixed C. Returns (cv, per_fold)."""
    n = len(y)
    oof_pred = np.zeros((n, NC))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        m = LogisticRegression(class_weight="balanced", C=C, max_iter=2000,
                               n_jobs=1, random_state=42, solver="lbfgs",
                               multi_class="multinomial")
        m.fit(OOF[tr], y[tr])
        oof_pred[vi] = m.predict_proba(OOF[vi])
    pf = [score_fn(y[vi], oof_pred[vi].argmax(1)) for vi in fval]
    return float(np.mean(pf)), pf


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    n = len(train)
    y = train["class"].map(L2I).to_numpy()
    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    fval = [np.asarray(f["val_idx"]) for f in folds_data]
    print(f"n_train={n}  n_folds={len(fval)}", flush=True)
    assert n == 577347, f"unexpected n_train={n}"

    # ---- load bank-17 (verbatim manifest) ---------------------------------
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"
    MANIFEST = {
        'xgb-0':     K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",
        'xgb-1':     K/"xgb-v1-for-s6e6/oof_preds.npy",
        'realmlp-0': B/"oof_preds_realmlp0_v12.csv",
        'realmlp-1': K/"realmlp-v1-for-s6e6/oof_preds.npy",
        'tabm-0':    B/"oof_preds_tabm0_v2.csv",
        'cat-0':     K/"cat-v0-for-s6e6/catboost_oof_predictions.csv",
        'realmlp-2': B/"oof_preds_realmlp2_v10.csv",
        'tabicl-2':  K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",
        'lgbm-3':    K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",
        'logreg-1':  K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",
        'nn-1':      K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",
        'xgb-3':     K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy",
        'xgb-5':     K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",
        'realmlp-5': K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",
        'nn-2':      K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",
        'cat-3':     K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",
        'lgbm-5':    B/"oof_preds_lgbm5_v1.csv",
        'xgb-6':     B/"oof_final_xgb6_v1.csv",
        'tabm-1':    B/"oof_final_tabm1_v1.csv",
    }
    blocks = []  # list of (name, kind, solo_ba, oof_logp[n,3])
    for name, op in MANIFEST.items():
        try:
            o = norm(rd(op, n))
            assert o.shape == (n, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            if 0.90 < ba < 0.972:
                blocks.append((name, "bank", ba, logp(o)))
        except Exception as e:
            print(f"  bank {name} FAIL {str(e)[:50]}", flush=True)
    print(f"loaded {len(blocks)} bank models", flush=True)

    # ---- FT-T -------------------------------------------------------------
    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    ft = norm(load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n))
    assert ft.shape == (n, 3)
    blocks.append(("ft_transformer", "bank", score_fn(y, ft.argmax(1)), logp(ft)))

    # ---- in-house TIGHT then WEAK (verbatim order) ------------------------
    for kind, ids in [("inhouse_tight", TIGHT_IDS), ("inhouse_weak", WEAK_EXTRA_IDS)]:
        for nid in ids:
            nm = f"node_{nid:04d}"
            try:
                o_raw = np.load(COMP / "nodes" / nm / "oof.npy").astype(float)
                assert o_raw.shape == (n, 3) and not np.isnan(o_raw).any()
                o = norm(o_raw)
                solo = score_fn(y, o.argmax(1))
                if solo < 0.5:
                    print(f"  {nm} SKIP (col-order bug, solo={solo:.4f})", flush=True)
                    continue
                blocks.append((nm, kind, solo, logp(o)))
            except Exception as e:
                print(f"  {nm} FAIL {str(e)[:50]}", flush=True)

    names = [b[0] for b in blocks]
    kinds = [b[1] for b in blocks]
    solos = [b[2] for b in blocks]
    n_bases = len(blocks)
    OOF_full = np.concatenate([b[3] for b in blocks], axis=1)
    print(f"\nFULL pool: {n_bases} bases × 3 = {OOF_full.shape[1]} cols, matrix {OOF_full.shape}", flush=True)
    print(f"  bank+ft: {kinds.count('bank')}  tight: {kinds.count('inhouse_tight')}  weak: {kinds.count('inhouse_weak')}", flush=True)

    # ---- baseline: full pool @ C=0.003 ------------------------------------
    print("\n=== BASELINE: full pool @ C=0.003 (5-fold OOF) ===", flush=True)
    t0 = time.time()
    cv_full, pf_full = arm_cv(OOF_full, y, fval, C_FIXED)
    per_fit = (time.time() - t0) / len(fval)
    print(f"cv_full={cv_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)
    print(f"  champion ref = 0.970355  delta={abs(cv_full-0.970355):.6f}  (fixed-C may differ slightly from nested)", flush=True)
    assert abs(cv_full - 0.970355) < 0.0004, f"baseline off ({cv_full:.6f}) — pool ingest wrong, STOP"
    print(f"  per-fit≈{per_fit:.1f}s → est LOO time ≈ {per_fit*5*n_bases/60:.0f} min", flush=True)

    # ---- |coef| cross-check (single full fit on all data) -----------------
    print("\n=== |coef| cross-check (full fit, all train) ===", flush=True)
    m_all = LogisticRegression(class_weight="balanced", C=C_FIXED, max_iter=2000,
                               n_jobs=1, random_state=42, solver="lbfgs",
                               multi_class="multinomial")
    m_all.fit(OOF_full, y)
    # coef_ is (NC, n_cols); per base = sum |coef| over its 3 cols and 3 classes
    abscoef = np.abs(m_all.coef_)  # (3, n_cols)
    coef_by_base = np.array([abscoef[:, 3*i:3*i+3].sum() for i in range(n_bases)])

    # ---- LOO over every base (parallel across bases) ----------------------
    print(f"\n=== LEAVE-ONE-BASE-OUT (5-fold @ C=0.003, {n_bases} bases, n_jobs=6) ===", flush=True)
    res = Parallel(n_jobs=6, verbose=10, backend="loky")(
        delayed(loo_one)(i, OOF_full, y, fval, n_bases, C_FIXED) for i in range(n_bases)
    )
    deltas = np.zeros(n_bases)
    cv_without = np.zeros(n_bases)
    for i, cv_i in sorted(res):
        cv_without[i] = cv_i
        deltas[i] = cv_full - cv_i  # positive => base helps
        print(f"  drop {names[i]:16s} cv_without={cv_i:.6f}  delta={deltas[i]:+.6f}", flush=True)

    # ---- rank + save ------------------------------------------------------
    order = np.argsort(-deltas)
    rows = []
    for rank, i in enumerate(order, 1):
        rows.append(dict(rank=rank, base=names[i], kind=kinds[i], solo_ba=round(solos[i], 6),
                         cv_without=round(cv_without[i], 6), delta=round(deltas[i], 6),
                         abscoef=round(float(coef_by_base[i]), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    sem_full = float(np.std(pf_full, ddof=1) / np.sqrt(len(pf_full)))
    print("\n" + "=" * 78, flush=True)
    print(f"cv_full={cv_full:.6f}  sem={sem_full:.6f}  (delta>0 = base helps; |delta|<sem ⇒ within noise)", flush=True)
    print(f"{'rank':>4s} {'base':16s} {'kind':14s} {'solo':>8s} {'delta':>9s} {'abscoef':>8s}", flush=True)
    for r in rows:
        flag = "  *signif*" if abs(r["delta"]) > sem_full else ""
        print(f"{r['rank']:>4d} {r['base']:16s} {r['kind']:14s} {r['solo_ba']:8.5f} {r['delta']:+9.6f} {r['abscoef']:8.3f}{flag}", flush=True)
    print(f"\nsaved → {OUT_CSV}", flush=True)
    print(f"cv_full={cv_full:.6f}", flush=True)


if __name__ == "__main__":
    main()
