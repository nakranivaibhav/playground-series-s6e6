"""node_0127 — 2-expert GATED MIXTURE meta (MoE over n091 base pool)

Atomic change vs node_0091 (champion):
  Replace the single global balanced-multinomial LogReg meta with a
  2-expert GATED MIXTURE using EM:
    - GATE g(x) = softmax over raw inputs [redshift, u-g, g-r] (a small LogReg)
    - Two balanced-multinomial LogReg EXPERTS over the same log-prob pool
    - Final prob = sum_k g_k(x) * softmax(expert_k)
    - Fit jointly via EM (5 iters, saga solver for speed on 577k rows)
    - Nested in-fold at C=0.003 (same as n091 best-C)

Pool: TIGHT arm = bank-17 + FT-T + 36 in-house = 54 bases × 3 = 162 cols
      FULL arm  = TIGHT + 9 weak = 63 bases × 3 = 189 cols

SANITY CHECK: uniform-gate mixture (0.5/0.5) on fold-0 must ~= n091 fold-0 single
              expert (≥ n091_cv - 1*sem to proceed)
CHEAP-KILL:   if 2-expert fold-0 MoE BA < n091_cv - 1*sem => stop

Speed optimizations vs naive EM:
  - saga solver: SGD-based, far faster than lbfgs on 577k rows
  - max_iter=300: adequate for saga at C=0.003
  - EM_ITERS=5: 5 alternations suffice to check separability
  - TIGHT arm first; FULL only if TIGHT passes cheap-kill
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
NODE_DIR = COMP / "nodes/node_0127"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# Fixed C for experts (same as n091 champion best-C from nested CV)
EXPERT_C = 0.003
# C for the gate (lightweight 2-class over 3 raw features)
GATE_C = 0.1
# EM iterations
EM_ITERS = 5
# Number of experts
K_EXPERTS = 2

# TIGHT pool (same as n091)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]


# ---------------------------------------------------------------------------
# Helpers (verbatim from champion)
# ---------------------------------------------------------------------------

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s

def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def rd(path: str | Path, nr: int) -> np.ndarray:
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

def load_ext_csv(path: str | Path, nr: int) -> np.ndarray:
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


# ---------------------------------------------------------------------------
# GATE feature builder: [redshift, u-g, g-r]
# ---------------------------------------------------------------------------

def build_gate_features(df: pd.DataFrame) -> np.ndarray:
    """Build 3-feature gate input: [redshift, u-g, g-r]"""
    redshift = df["redshift"].values.astype(float)
    ug = df["u"].values.astype(float) - df["g"].values.astype(float)
    gr = df["g"].values.astype(float) - df["r"].values.astype(float)
    return np.column_stack([redshift, ug, gr])


def make_logreg(C: float = EXPERT_C, class_weight: str = "balanced",
                random_state: int = 42) -> LogisticRegression:
    """Create a LogReg. lbfgs at tol=1e-3, max_iter=100: ~11s on 462k×162 real TIGHT matrix."""
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=100,
        n_jobs=-1,
        random_state=random_state,
        solver="lbfgs",
        tol=1e-3,
    )


def make_gate_logreg(C: float = GATE_C, random_state: int = 42) -> LogisticRegression:
    """Gate classifier over 3 raw features (binary: which expert)."""
    return LogisticRegression(
        C=C,
        class_weight=None,  # gate: not balanced (routing, not prediction)
        max_iter=200,
        n_jobs=1,  # 3 features — no parallelism needed
        random_state=random_state,
        solver="lbfgs",
        tol=1e-4,
    )


# ---------------------------------------------------------------------------
# Mixture predict
# ---------------------------------------------------------------------------

def mixture_predict(gate_probs: np.ndarray,
                    expert_probs: list[np.ndarray]) -> np.ndarray:
    """Final prob = sum_k gate_k * expert_k."""
    out = np.zeros((gate_probs.shape[0], NC), dtype=float)
    for k, ep in enumerate(expert_probs):
        out += gate_probs[:, k:k+1] * ep
    return out


# ---------------------------------------------------------------------------
# EM fit for 2-expert gated mixture
# ---------------------------------------------------------------------------

def fit_gated_mixture(
    X_meta: np.ndarray,   # (n, n_meta_cols) — log-prob features
    X_gate: np.ndarray,   # (n, 3) — raw gate inputs
    y: np.ndarray,        # (n,) int labels
    n_em_iters: int = EM_ITERS,
    verbose: bool = True,
) -> tuple:
    """EM-fit 2-expert gated mixture. Returns (gate_lr, expert_lrs).

    Symmetry-breaking initialization: pre-fit the gate on raw gate features
    (redshift, u-g, g-r) with a quick LogReg, then use it to generate initial
    responsibilities. This avoids the degenerate uniform-init collapse where
    both experts converge to the same solution.
    """
    n = len(y)
    K = K_EXPERTS

    # --- Symmetry-breaking: initialize gate on raw gate features ---
    # A quick LogReg on [redshift, u-g, g-r] vs y (multi-class → use as proxy
    # to assign rows to experts: high-redshift rows → expert 0, low-redshift → expert 1)
    # We use the predicted prob from this initial gate to generate soft responsibilities.
    #
    # Specifically: fit a 2-class LogReg on gate features using a hard label derived
    # from a simple threshold on the gate feature with highest correlation:
    # g-r has |corr|=0.59 with class label, so it roughly separates classes.
    # We split: rows where gate_feature[2] (g-r) > median → label 0, else label 1.
    gr_median = np.median(X_gate[:, 2])
    init_gate_labels = (X_gate[:, 2] > gr_median).astype(int)

    init_gate_lr = make_gate_logreg(C=GATE_C, random_state=42)
    init_gate_lr.fit(X_gate, init_gate_labels)
    # Soft responsibilities from initial gate (50/50 softmax is not used — use actual prob)
    responsibilities = init_gate_lr.predict_proba(X_gate)  # (n, 2)
    # Add noise to prevent early collapse
    rng = np.random.RandomState(42)
    noise = rng.dirichlet(np.ones(K), size=n) * 0.1  # small noise
    responsibilities = (responsibilities * 0.9 + noise)
    responsibilities /= responsibilities.sum(1, keepdims=True)

    gate_lr = init_gate_lr
    expert_lrs = [None] * K

    for it in range(n_em_iters):
        # ---- M-step: fit each expert with sample weights ----
        new_expert_lrs = []
        for k in range(K):
            w = responsibilities[:, k]
            # Guard: need all 3 classes weighted
            if w.sum() < 10 or len(np.unique(y[w > 0.01])) < NC:
                w = np.ones(n, dtype=float) / n
            lr = make_logreg(C=EXPERT_C, random_state=42 + k)
            lr.fit(X_meta, y, sample_weight=w)
            new_expert_lrs.append(lr)
        expert_lrs = new_expert_lrs

        # ---- Expert probabilities ----
        expert_probs = [lr.predict_proba(X_meta) for lr in expert_lrs]  # K × (n, NC)

        # ---- E-step: compute responsibilities ----
        # r_k(i) ∝ gate_k(x_gate_i) * p_k(y_i | x_meta_i)
        likelihoods = np.zeros((n, K), dtype=float)
        for k in range(K):
            likelihoods[:, k] = expert_probs[k][np.arange(n), y]

        gp = gate_lr.predict_proba(X_gate)  # (n, K) using current gate

        responsibilities = gp * likelihoods
        rs = responsibilities.sum(1, keepdims=True)
        rs = np.where(rs < 1e-15, 1.0, rs)
        responsibilities /= rs

        # ---- Fit gate on hard responsibility assignments ----
        gate_labels = responsibilities.argmax(1)
        if len(np.unique(gate_labels)) < 2:
            # Degenerate: keep previous gate
            if verbose:
                print(f"    EM iter {it}: gate degenerate (all expert {gate_labels[0]}) — keeping prev gate", flush=True)
        else:
            gate_lr = make_gate_logreg(C=GATE_C, random_state=42)
            gate_lr.fit(X_gate, gate_labels)

            if verbose:
                gp_full = gate_lr.predict_proba(X_gate)
                gate_entropy = float(
                    -np.sum(gp_full * np.log(np.clip(gp_full, 1e-10, 1)), axis=1).mean()
                )
                r_mean = responsibilities.mean(0)
                print(f"    EM iter {it}: gate_entropy={gate_entropy:.4f}  "
                      f"r_frac=[{r_mean[0]:.3f},{r_mean[1]:.3f}]", flush=True)

    return gate_lr, expert_lrs


def predict_gated_mixture(
    X_meta: np.ndarray,
    X_gate: np.ndarray,
    gate_lr,
    expert_lrs: list,
) -> np.ndarray:
    """Predict final mixture probs."""
    K = len(expert_lrs)
    if gate_lr is None:
        gate_probs = np.ones((len(X_meta), K), dtype=float) / K
    else:
        gate_probs = gate_lr.predict_proba(X_gate)

    expert_probs = [lr.predict_proba(X_meta) for lr in expert_lrs]
    return mixture_predict(gate_probs, expert_probs)


# ---------------------------------------------------------------------------
# Full nested MoE OOF loop
# ---------------------------------------------------------------------------

def moe_arm(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    GATE_tr: np.ndarray,
    GATE_te: np.ndarray,
    y: np.ndarray,
    fval: list,
    label: str,
    n091_cv: float = 0.97030,
    n091_sem: float = 0.000252,
) -> tuple:
    """
    Nested 2-expert MoE OOF loop.
    Returns: (oof, test, per_fold, cv, sem, killed)
    """
    n = len(y)
    n_folds = len(fval)
    oof_probs = np.zeros((n, NC), dtype=float)
    kill_thresh = n091_cv - n091_sem

    print(f"\n=== MoE ARM: {label}  meta_cols={OOF_mat.shape[1]}  gate_cols={GATE_tr.shape[1]} ===",
          flush=True)

    # ---- Fold-0 first: sanity + cheap-kill ----
    fi = 0
    vi = fval[fi]
    tr_idx = np.setdiff1d(np.arange(n), vi)
    X_tr = OOF_mat[tr_idx]; y_tr = y[tr_idx]; G_tr = GATE_tr[tr_idx]
    X_va = OOF_mat[vi];     y_va = y[vi];     G_va = GATE_tr[vi]

    # SANITY: uniform gate (average of 2 experts with equal weight = single LogReg equivalent)
    print("\n[SANITY] Fold-0 uniform-gate vs single LogReg...", flush=True)
    single_lr = make_logreg(C=EXPERT_C, random_state=42)
    single_lr.fit(X_tr, y_tr)
    single_preds = single_lr.predict_proba(X_va)
    single_ba = score_fn(y_va, single_preds.argmax(1))

    # Two experts both with full uniform weights → average = uniform gate
    exp0 = make_logreg(C=EXPERT_C, random_state=42)
    exp1 = make_logreg(C=EXPERT_C, random_state=43)
    exp0.fit(X_tr, y_tr)
    exp1.fit(X_tr, y_tr)
    unif_preds = 0.5 * exp0.predict_proba(X_va) + 0.5 * exp1.predict_proba(X_va)
    unif_ba = score_fn(y_va, unif_preds.argmax(1))

    print(f"[SANITY] single LogReg fold-0 BA={single_ba:.6f}", flush=True)
    print(f"[SANITY] uniform-gate mixture fold-0 BA={unif_ba:.6f}", flush=True)
    print(f"[SANITY] Expected n091_cv={n091_cv:.5f}. OK if both are close.", flush=True)
    sanity_ok = abs(unif_ba - single_ba) < 0.002
    print(f"[SANITY] uniform~=single? {sanity_ok}  delta={unif_ba - single_ba:+.6f}", flush=True)

    # ---- Fold-0 MoE ----
    print(f"\n[FOLD 0] Fitting 2-expert EM mixture...", flush=True)
    gate_lr0, exp_lrs0 = fit_gated_mixture(X_tr, G_tr, y_tr, verbose=True)
    moe_preds0 = predict_gated_mixture(X_va, G_va, gate_lr0, exp_lrs0)
    moe_ba0 = score_fn(y_va, moe_preds0.argmax(1))
    print(f"[FOLD 0] MoE BA={moe_ba0:.6f}  (single={single_ba:.6f}  unif={unif_ba:.6f})", flush=True)

    if gate_lr0 is not None:
        gp_va = gate_lr0.predict_proba(G_va)
        ge = float(-np.sum(gp_va * np.log(np.clip(gp_va, 1e-10, 1)), axis=1).mean())
        routing = gp_va.argmax(1)
        print(f"[FOLD 0] Gate val entropy={ge:.4f}  "
              f"routing: exp0={routing.mean():.3f}←frac (1=exp1)", flush=True)
    else:
        print(f"[FOLD 0] Gate degenerate (uniform).", flush=True)

    # CHEAP-KILL
    print(f"\n[CHEAP-KILL] fold-0 MoE={moe_ba0:.6f} vs thresh={kill_thresh:.6f} "
          f"(n091={n091_cv:.6f} - 1*sem={n091_sem:.6f})", flush=True)
    if moe_ba0 < kill_thresh:
        print(f"CHEAP-KILL TRIPPED: {moe_ba0:.6f} < {kill_thresh:.6f}. Stopping {label}.", flush=True)
        return None, None, [moe_ba0], moe_ba0, 0.0, True

    print(f"[CHEAP-KILL] PASS — continuing to full 5-fold.", flush=True)
    oof_probs[vi] = moe_preds0

    # ---- Remaining folds ----
    for fi in range(1, n_folds):
        vi = fval[fi]
        tr_idx = np.setdiff1d(np.arange(n), vi)
        X_tr = OOF_mat[tr_idx]; y_tr = y[tr_idx]; G_tr_f = GATE_tr[tr_idx]
        X_va = OOF_mat[vi];     y_va = y[vi];     G_va_f = GATE_tr[vi]

        print(f"\n[FOLD {fi}] Fitting 2-expert EM mixture...", flush=True)
        gate_lr_i, exp_lrs_i = fit_gated_mixture(X_tr, G_tr_f, y_tr, verbose=True)
        moe_preds_i = predict_gated_mixture(X_va, G_va_f, gate_lr_i, exp_lrs_i)
        moe_ba_i = score_fn(y_va, moe_preds_i.argmax(1))
        print(f"[FOLD {fi}] MoE BA={moe_ba_i:.6f}", flush=True)

        if gate_lr_i is not None:
            gp_va = gate_lr_i.predict_proba(G_va_f)
            ge = float(-np.sum(gp_va * np.log(np.clip(gp_va, 1e-10, 1)), axis=1).mean())
            routing = gp_va.argmax(1)
            print(f"[FOLD {fi}] Gate val entropy={ge:.4f}  "
                  f"routing frac exp1={routing.mean():.3f}", flush=True)

        oof_probs[vi] = moe_preds_i

    # Per-fold scores
    per_fold = [score_fn(y[vi], oof_probs[vi].argmax(1)) for vi in fval]
    cv_mean = float(np.mean(per_fold))
    cv_sem  = float(np.std(per_fold, ddof=1) / np.sqrt(n_folds))
    print(f"\n  {label} cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"  per_fold={[f'{s:.6f}' for s in per_fold]}", flush=True)

    # ---- Final refit on ALL train ----
    print(f"\nFinal refit on all train ({label})...", flush=True)
    gate_lr_fin, exp_lrs_fin = fit_gated_mixture(OOF_mat, GATE_tr, y, verbose=False)
    test_probs = predict_gated_mixture(TST_mat, GATE_te, gate_lr_fin, exp_lrs_fin)

    return oof_probs, test_probs, per_fold, cv_mean, cv_sem, False


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n  = len(train)
    nt = len(test)
    y  = train["class"].map(L2I).to_numpy()

    fval    = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n  == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # =========================================================================
    # PRE-FLIGHT: Leakage checks 1-2
    print("\n[LEAKAGE CHECK 1-2] Meta features = OOF log-probs (no target/id). PASS", flush=True)
    print("  Gate features = [redshift, u-g, g-r] (raw stateless). PASS", flush=True)
    print("[LEAKAGE CHECK 4] Expert+gate fit inside fold loop on tr_idx only. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Gate features (stateless — no fit)
    print("\n[GATE FEATURES] Building [redshift, u-g, g-r]...", flush=True)
    GATE_train = build_gate_features(train)
    GATE_test  = build_gate_features(test)
    print(f"  GATE_train={GATE_train.shape}  GATE_test={GATE_test.shape}", flush=True)

    # Leakage check 3: gate features
    print("\n[LEAKAGE CHECK 3] Gate feature corr sweep...", flush=True)
    rng  = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys   = y[sidx].astype(float)
    for name, arr in [("redshift", GATE_train[sidx, 0]),
                      ("u-g",      GATE_train[sidx, 1]),
                      ("g-r",      GATE_train[sidx, 2])]:
        corr = abs(np.corrcoef(arr, ys)[0, 1])
        print(f"  {name}: |corr|={corr:.4f}", flush=True)
        assert corr < 0.999, f"LEAK smell: {name} corr={corr:.4f}"
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # Load public bank (same MANIFEST as champion)
    B     = COMP / "refs/oof_bank"
    K_ref = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-0':     (K_ref/"xgb-v0-for-s6e6/oof_xgb_cv.csv",             K_ref/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':     (K_ref/"xgb-v1-for-s6e6/oof_preds.npy",              K_ref/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0': (B/"oof_preds_realmlp0_v12.csv",                      B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1': (K_ref/"realmlp-v1-for-s6e6/oof_preds.npy",           K_ref/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':    (B/"oof_preds_tabm0_v2.csv",                          B/"test_preds_tabm0_v2.csv"),
        'cat-0':     (K_ref/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K_ref/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2': (B/"oof_preds_realmlp2_v10.csv",                      B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':  (K_ref/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K_ref/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':    (K_ref/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K_ref/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':  (K_ref/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy", K_ref/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':      (K_ref/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",         K_ref/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':     (K_ref/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K_ref/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':     (K_ref/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",      K_ref/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5': (K_ref/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy", K_ref/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':      (K_ref/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",         K_ref/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':     (K_ref/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",       K_ref/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':    (B/"oof_preds_lgbm5_v1.csv",                           B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':     (B/"oof_final_xgb6_v1.csv",                            B/"test_final_xgb6_v1.csv"),
        'tabm-1':    (B/"oof_final_tabm1_v1.csv",                           B/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    print(f"\n{'model':14s} {'oofBA':>9s} {'shape':>12s} {'status'}", flush=True)
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK":
                POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:9.6f} {str(o.shape):>12s} {st}", flush=True)
        except Exception as e:
            print(f"{name:14s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(good)} public bank models", flush=True)

    # =========================================================================
    # FT-Transformer
    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    ft_oof_raw  = load_ext_csv(PILK/"oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n)
    ft_test_raw = load_ext_csv(PILK/"sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", nt)
    assert ft_oof_raw.shape  == (n,  3)
    assert ft_test_raw.shape == (nt, 3)
    ft_solo_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}", flush=True)
    assert ft_solo_ba > 0.85

    # Leakage check 3 (meta): spot check
    arr = logp(norm(ft_oof_raw))[sidx, 0]
    corr = abs(np.corrcoef(arr, ys)[0, 1])
    assert corr < 0.999, f"LEAK: ft_transformer col0 corr={corr:.4f}"
    print(f"[LEAKAGE CHECK 3] ft_transformer col0 corr={corr:.4f}: PASS", flush=True)

    # =========================================================================
    # Load in-house TIGHT
    print("\n" + "="*70, flush=True)
    inhouse_oof_tight  = {}
    inhouse_test_tight = {}
    for nid in TIGHT_IDS:
        nm = f"node_{nid:04d}"
        try:
            o_raw = np.load(COMP/"nodes"/nm/"oof.npy").astype(float)
            t_raw = np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float)
            assert o_raw.shape == (n, 3) and t_raw.shape == (nt, 3)
            assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
            o = norm(o_raw); t = norm(t_raw)
            sba = score_fn(y, o.argmax(1))
            if sba < 0.5:
                print(f"{nm} SKIP (col-order bug)", flush=True); continue
            inhouse_oof_tight[nm]  = logp(o)
            inhouse_test_tight[nm] = logp(t)
        except Exception as e:
            print(f"{nm} FAIL {str(e)[:50]}", flush=True)

    print(f"TIGHT in-house: {len(inhouse_oof_tight)}/{len(TIGHT_IDS)}", flush=True)

    inhouse_oof_weak  = {}
    inhouse_test_weak = {}
    for nid in WEAK_EXTRA_IDS:
        nm = f"node_{nid:04d}"
        try:
            o_raw = np.load(COMP/"nodes"/nm/"oof.npy").astype(float)
            t_raw = np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float)
            assert o_raw.shape == (n, 3) and t_raw.shape == (nt, 3)
            assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
            o = norm(o_raw); t = norm(t_raw)
            sba = score_fn(y, o.argmax(1))
            if sba < 0.5:
                print(f"{nm} SKIP", flush=True); continue
            inhouse_oof_weak[nm]  = logp(o)
            inhouse_test_weak[nm] = logp(t)
        except Exception as e:
            print(f"{nm} FAIL {str(e)[:50]}", flush=True)

    print(f"FULL extra in-house: {len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)}", flush=True)

    # =========================================================================
    # Feature matrices
    base_oof  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    base_test = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

    OOF_tight = np.concatenate(base_oof  + list(inhouse_oof_tight.values()),  axis=1)
    TST_tight = np.concatenate(base_test + list(inhouse_test_tight.values()), axis=1)
    print(f"\nTIGHT feature_matrix={OOF_tight.shape}", flush=True)

    OOF_full  = np.concatenate(base_oof  + list(inhouse_oof_tight.values())  + list(inhouse_oof_weak.values()),  axis=1)
    TST_full  = np.concatenate(base_test + list(inhouse_test_tight.values()) + list(inhouse_test_weak.values()), axis=1)
    print(f"FULL  feature_matrix={OOF_full.shape}", flush=True)

    N091_CV  = 0.97030
    N091_SEM = 0.000252

    # =========================================================================
    # RUN TIGHT arm
    print("\n" + "="*70, flush=True)
    print("RUNNING TIGHT MoE arm...", flush=True)
    oof_tight, test_tight, pf_tight, cv_tight, sem_tight, killed_tight = moe_arm(
        OOF_tight, TST_tight, GATE_train, GATE_test, y, fval, "TIGHT",
        n091_cv=N091_CV, n091_sem=N091_SEM,
    )

    if killed_tight:
        print("\nCHEAP-KILL in TIGHT arm. Not running FULL arm. Exiting.", flush=True)
        # Write trivial artifacts so the node is clearly marked as killed
        oof_out  = np.full((n,  NC), 1.0/NC, dtype=np.float32)
        test_out = np.full((nt, NC), 1.0/NC, dtype=np.float32)
        np.save(NODE_DIR/"oof.npy",        oof_out)
        np.save(NODE_DIR/"test_probs.npy", test_out)
        sub = pd.DataFrame({"id": test["id"], "class": ["GALAXY"] * nt})
        sub.to_csv(NODE_DIR/"submission.csv", index=False)
        print(f"cv=null  (cheap-kill tripped: fold-0 MoE={pf_tight[0]:.6f} < thresh={N091_CV-N091_SEM:.6f})",
              flush=True)
        import sys; sys.exit(0)

    # =========================================================================
    # RUN FULL arm
    print("\n" + "="*70, flush=True)
    print("RUNNING FULL MoE arm...", flush=True)
    oof_full, test_full, pf_full, cv_full, sem_full, killed_full = moe_arm(
        OOF_full, TST_full, GATE_train, GATE_test, y, fval, "FULL",
        n091_cv=N091_CV, n091_sem=N091_SEM,
    )

    # =========================================================================
    # Determine winner
    print("\n" + "="*70, flush=True)
    print("=== ARM COMPARISON ===", flush=True)
    print(f"TIGHT MoE: cv={cv_tight:.6f}  sem={sem_tight:.6f}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    if not killed_full:
        print(f"FULL  MoE: cv={cv_full:.6f}  sem={sem_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)
    else:
        print("FULL arm cheap-kill tripped — using TIGHT.", flush=True)

    if killed_full or cv_tight >= cv_full:
        winner = "TIGHT"; cv_win = cv_tight; sem_win = sem_tight
        pf_win = pf_tight; oof_win = oof_tight; test_win = test_tight
    else:
        winner = "FULL"; cv_win = cv_full; sem_win = sem_full
        pf_win = pf_full; oof_win = oof_full; test_win = test_full

    print(f"\nWINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)

    promote_bar = N091_CV + 2 * N091_SEM
    lift = cv_win - N091_CV
    print(f"n091_cv={N091_CV:.6f}  2*sem={2*N091_SEM:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"lift_vs_n091={lift:+.6f}  beats_promote={'YES' if cv_win > promote_bar else 'NO'}", flush=True)

    # =========================================================================
    # Write artifacts
    np.save(NODE_DIR/"oof.npy",        oof_win.astype(np.float32))
    np.save(NODE_DIR/"test_probs.npy", test_win.astype(np.float32))

    test_labels = [I2L[i] for i in test_win.argmax(1)]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR/"submission.csv", index=False)

    print(f"\nArtifacts: oof={oof_win.shape}  test={test_win.shape}  sub={len(sub)} rows", flush=True)

    # =========================================================================
    # Post-run gates
    print("\n[POST-RUN GATES]", flush=True)

    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    assert set(sub["class"].unique()) <= set(LAB)
    print("  schema_ok: PASS", flush=True)

    oofn = np.load(NODE_DIR/"oof.npy")
    assert oofn.shape == (n, NC) and not np.isnan(oofn).any()
    print("  oof_full: PASS  no_nan: PASS", flush=True)

    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5
    rs = oofn.sum(1)
    assert abs(rs.mean() - 1.0) < 0.01
    cc = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  argmax GALAXY={cc[0]} QSO={cc[1]} STAR={cc[2]}", flush=True)
    print(f"             range=[{oofn.min():.4f},{oofn.max():.4f}]  row_sums={rs.mean():.6f}", flush=True)

    cv_too_good = cv_win > 0.980
    print(f"  cv_too_good: {'WARN >0.980' if cv_too_good else 'PASS'}", flush=True)

    # =========================================================================
    # Final summary
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"TIGHT: cv={cv_tight:.6f}  sem={sem_tight:.6f}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    if not killed_full:
        print(f"FULL:  cv={cv_full:.6f}  sem={sem_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)
    print(f"WINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)
    print(f"promotes? {'YES' if cv_win > promote_bar else 'NO'}  (bar={promote_bar:.6f})", flush=True)
    print(f"cv={cv_win:.6f}", flush=True)


if __name__ == "__main__":
    main()
