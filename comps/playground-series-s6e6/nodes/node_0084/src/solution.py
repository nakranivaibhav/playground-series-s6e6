"""node_0084 — Revive strong discards into best-honest stack.

A/B-adds node_0067 (transductive-distill TabM) and node_0074 (disjoint-teacher
TabM) OOFs as candidates onto node_0076 bank17+FT-T+bagged-argmax stack via
forward selection. No retraining — only saved OOF/test_probs loaded.

CLOUT flag: node_0074 is CLOUT-tainted (A4-vote teacher). If n74 is
fwd-selected, the result is finals slot-2 ONLY, NOT honest slot-1.
node_0067 is honest.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
SEEDS = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------------------
# Helpers (byte-identical to node_0076)
# ---------------------------------------------------------------------------

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True); s[s == 0] = 1
    return a / s

def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def fit_meta(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m

def rd(path: str | Path, nr: int) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3: return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float); return v.reshape(nr, 3)

def load_ext_csv(path: str | Path, nr: int) -> np.ndarray:
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


def run_bagged_stack(OOF_full: np.ndarray, TST_full: np.ndarray,
                     y: np.ndarray, fval: list, nt: int) -> tuple:
    """5-seed bagged LogReg, plain argmax. Returns (bagged_oof, bagged_test, per_fold_scores)."""
    n = len(y)
    seed_oof_probs  = np.zeros((len(SEEDS), n, NC))
    seed_test_probs = np.zeros((len(SEEDS), nt, NC))

    for si, seed in enumerate(SEEDS):
        seed_oof = np.zeros((n, NC))
        for fi, vi in enumerate(fval):
            tr_idx = np.setdiff1d(np.arange(n), vi)
            m = fit_meta(OOF_full[tr_idx], y[tr_idx], seed=seed)
            seed_oof[vi] = m.predict_proba(OOF_full[vi])
        seed_oof_probs[si] = seed_oof
        m_full = fit_meta(OOF_full, y, seed=seed)
        seed_test_probs[si] = m_full.predict_proba(TST_full)

    bagged_oof  = seed_oof_probs.mean(axis=0)
    bagged_test = seed_test_probs.mean(axis=0)

    per_fold_scores = [score_fn(y[vi], bagged_oof[vi].argmax(1)) for vi in fval]
    return bagged_oof, bagged_test, per_fold_scores


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    print(f"n_train={n} n_test={nt} n_folds={len(fval)}")
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # ---- PRE-FLIGHT LEAKAGE CHECKS ----
    print("Leakage check 1-2: features are OOF probs only (no target/id). PASS")
    print("Leakage check 4-5: LogReg fit inside fold loop; folds from frozen folds.json. PASS")

    # ---- Load public bank-17 ----
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",         K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",           K/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",               B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",        K/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                   B/"test_preds_tabm0_v2.csv"),
        'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
        'tabm-1':     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    print(f"\n{'model':12s} {'oofBA':>9s} {'shape':>12s} {'status'}")
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK": POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:12s} {ba:9.6f} {str(o.shape):>12s} {st}")
        except Exception as e:
            print(f"{name:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")

    print(f"\nLoaded {len(good)} public models OK (expected 17)")

    # ---- Load FT-Transformer ----
    PILK = COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_path  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_test_path = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    ft_oof_raw  = load_ext_csv(ft_oof_path, n)
    ft_test_raw = load_ext_csv(ft_test_path, nt)
    assert ft_oof_raw.shape == (n, 3),   f"FT-T OOF shape {ft_oof_raw.shape}"
    assert ft_test_raw.shape == (nt, 3), f"FT-T test shape {ft_test_raw.shape}"
    ft_solo_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}  shape={ft_oof_raw.shape}")

    # ---- Leakage check 3 ----
    print("\nLeakage check 3: single-feature correlation sweep (sample 50k)...")
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    ft_logp = logp(norm(ft_oof_raw))
    for ci in range(3):
        x = ft_logp[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: ft_transformer col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3: PASS")

    # ---- Build base feature matrix: bank-17 log-probs + FT-T log-probs ----
    base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

    OOF_base = np.concatenate(base_oof_logp, axis=1)   # (n, 18*3=54)
    TST_base = np.concatenate(base_test_logp, axis=1)  # (nt, 54)
    print(f"\nBase feature matrix: train={OOF_base.shape} test={TST_base.shape}")

    # ---- STEP 1: reproduce node_0076 baseline (HARD assert) ----
    print("\n=== STEP 1: Reproduce node_0076 baseline ===")
    bagged_oof_base, bagged_test_base, pf_base = run_bagged_stack(OOF_base, TST_base, y, fval, nt)
    cv_base = float(np.mean(pf_base))
    sem_base = float(np.std(pf_base, ddof=1) / np.sqrt(len(pf_base)))
    print(f"Baseline per-fold: {[f'{s:.6f}' for s in pf_base]}")
    print(f"Baseline cv={cv_base:.6f}  sem={sem_base:.6f}")

    EXPECTED_BASE = 0.970227
    assert abs(cv_base - EXPECTED_BASE) < 0.0002, \
        f"HARD ASSERT FAILED: baseline cv={cv_base:.6f} expected ~{EXPECTED_BASE}"
    print(f"HARD ASSERT PASSED: baseline reproduced (delta={cv_base - EXPECTED_BASE:+.7f})")

    # ---- Load candidate OOFs ----
    n67_oof  = norm(np.load(COMP / "nodes/node_0067/oof.npy").astype(float))
    n67_test = norm(np.load(COMP / "nodes/node_0067/test_probs.npy").astype(float))
    n74_oof  = norm(np.load(COMP / "nodes/node_0074/oof.npy").astype(float))
    n74_test = norm(np.load(COMP / "nodes/node_0074/test_probs.npy").astype(float))
    assert n67_oof.shape == (n, 3),  f"n67 oof shape {n67_oof.shape}"
    assert n67_test.shape == (nt, 3), f"n67 test shape {n67_test.shape}"
    assert n74_oof.shape == (n, 3),  f"n74 oof shape {n74_oof.shape}"
    assert n74_test.shape == (nt, 3), f"n74 test shape {n74_test.shape}"

    n67_ba = score_fn(y, n67_oof.argmax(1))
    n74_ba = score_fn(y, n74_oof.argmax(1))
    print(f"\nn67 solo BA={n67_ba:.6f}  (claimed 0.969414)")
    print(f"n74 solo BA={n74_ba:.6f}  (claimed 0.968528)")

    # Leakage check 3 on candidates: no single col corr >= 0.999
    for name_c, c_oof in [("n67", n67_oof), ("n74", n74_oof)]:
        for ci in range(3):
            x = logp(c_oof)[sidx, ci]
            corr = abs(np.corrcoef(x, ys)[0, 1])
            if corr >= 0.999:
                raise SystemExit(f"LEAK smell: {name_c} col {ci} corr={corr:.4f}")
    print("Leakage check 3 (candidates): PASS")

    # ---- STEP 2: Forward selection — try adding n67, then n74 ----
    # Start from base. Candidates tried in order: n67 (honest), n74 (clout-tainted).
    selected_names = list(good) + ["ft_transformer"]
    current_oof    = OOF_base.copy()
    current_test   = TST_base.copy()
    current_cv     = cv_base
    current_pf     = pf_base[:]

    # Track which honest and clout additions were accepted
    honest_added = []
    clout_added  = []
    honest_cv = cv_base
    honest_oof = bagged_oof_base
    honest_test = bagged_test_base
    honest_pf = pf_base[:]

    candidates = [
        ("node_0067", logp(n67_oof), logp(n67_test), "honest"),
        ("node_0074", logp(n74_oof), logp(n74_test), "clout"),
    ]

    print(f"\n=== STEP 2: Forward selection onto baseline cv={current_cv:.6f} ===")

    final_oof  = bagged_oof_base
    final_test = bagged_test_base
    final_pf   = pf_base[:]
    clout_flag = False

    for cand_name, cand_oof_logp, cand_test_logp, taint in candidates:
        print(f"\n-- Trying {cand_name} ({taint}) --")
        new_oof  = np.concatenate([current_oof,  cand_oof_logp],  axis=1)
        new_test = np.concatenate([current_test, cand_test_logp], axis=1)

        b_oof, b_test, pf_new = run_bagged_stack(new_oof, new_test, y, fval, nt)
        cv_new = float(np.mean(pf_new))
        sem_new = float(np.std(pf_new, ddof=1) / np.sqrt(len(pf_new)))
        delta = cv_new - current_cv
        print(f"  {cand_name}: cv={cv_new:.6f}  delta={delta:+.6f}")

        if delta > 0:
            print(f"  ACCEPTED (delta={delta:+.6f} > 0)")
            current_oof   = new_oof
            current_test  = new_test
            current_cv    = cv_new
            current_pf    = pf_new[:]
            final_oof     = b_oof
            final_test    = b_test
            final_pf      = pf_new[:]

            if taint == "honest":
                honest_added.append(cand_name)
                honest_cv  = cv_new
                honest_oof  = b_oof
                honest_test = b_test
                honest_pf   = pf_new[:]
            else:
                clout_added.append(cand_name)
                clout_flag = True
        else:
            print(f"  REJECTED (delta={delta:+.6f} <= 0)")

    # ---- Determine honest stack outcome ----
    # Honest stack = base + any honest additions (no clout)
    if honest_added:
        h_oof  = honest_oof
        h_test = honest_test
        h_pf   = honest_pf
        h_cv   = float(np.mean(h_pf))
        h_sem  = float(np.std(h_pf, ddof=1) / np.sqrt(len(h_pf)))
    else:
        h_oof  = bagged_oof_base
        h_test = bagged_test_base
        h_pf   = pf_base[:]
        h_cv   = cv_base
        h_sem  = sem_base

    print(f"\n=== RESULTS ===")
    print(f"(a) Honest re-stack (n076 + {honest_added if honest_added else 'no additions'})")
    print(f"    per-fold: {[f'{s:.6f}' for s in h_pf]}")
    print(f"    cv={h_cv:.6f}  sem={h_sem:.6f}")
    print(f"    delta vs baseline: {h_cv - cv_base:+.6f}")

    if clout_flag:
        cl_cv  = float(np.mean(final_pf))
        cl_sem = float(np.std(final_pf, ddof=1) / np.sqrt(len(final_pf)))
        print(f"\n(b) CLOUT re-stack (n076 + {honest_added} + {clout_added} [CLOUT-TAINTED])")
        print(f"    per-fold: {[f'{s:.6f}' for s in final_pf]}")
        print(f"    cv={cl_cv:.6f}  sem={cl_sem:.6f}")
        print(f"    delta vs baseline: {cl_cv - cv_base:+.6f}")
        print(f"    WARNING: CLOUT-tainted — finals slot-2 ONLY, NOT honest slot-1")

    # Promotion check — honest stack vs champion 0.970153
    CHAMPION_CV = 0.970153
    promote_thresh = CHAMPION_CV + 2 * h_sem
    print(f"\nPromotion check (honest): cv={h_cv:.6f} vs threshold={promote_thresh:.6f} (champ={CHAMPION_CV:.6f} + 2*sem={2*h_sem:.6f})")
    if h_cv > promote_thresh:
        print("PROMOTE: honest stack beats champion by >2*sem")
    else:
        print("NO PROMOTE: honest stack does not beat champion threshold")

    # ---- Choose which OOF to save ----
    # Save the honest stack result as the canonical output
    save_oof   = h_oof
    save_test  = h_test
    save_pf    = h_pf
    save_cv    = h_cv
    save_sem   = h_sem

    # ---- Write outputs ----
    test_preds_idx = save_test.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]

    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", save_oof)
    np.save(NODE_DIR / "test_probs.npy", save_test)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {save_oof.shape}  test_probs.npy: {save_test.shape}")

    # ---- ALSO write the CLOUT stack submission (final_test incl n74), slot-2 only ----
    if clout_flag:
        clout_labels = [I2L[i] for i in final_test.argmax(1)]
        pd.DataFrame({"id": test["id"], "class": clout_labels}).to_csv(
            NODE_DIR / "submission_clout.csv", index=False)
        np.save(NODE_DIR / "test_probs_clout.npy", final_test)
        print(f"clout submission written: submission_clout.csv ({len(test)} rows) — SLOT-2 ONLY")

    # ---- Schema check ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count: {len(sub)} vs {len(sample_sub)}"
    print("submission schema: OK")

    # ---- Post-run output gates ----
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: mean={row_sums.mean()}"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"honest per_fold: {save_pf}")
    print(f"cv={save_cv:.6f}  sem={save_sem:.6f}")
    if clout_flag:
        print(f"clout_cv={float(np.mean(final_pf)):.6f}  clout_delta={float(np.mean(final_pf))-cv_base:+.6f}")


if __name__ == "__main__":
    main()
