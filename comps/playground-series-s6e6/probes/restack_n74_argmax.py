"""Fast argmax restack: does clout n74 help the bank-17 stack? Relative deltas only."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bac
C = Path(__file__).resolve().parent.parent
L = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
tr = pd.read_csv(C/'data/train.csv'); y = tr['class'].map(L).to_numpy(); n = len(y)
fv = [np.asarray(f['val_idx']) for f in json.loads((C/'folds.json').read_text())['folds']]
def norm(a): a = np.clip(a, 0, None); s = a.sum(1, keepdims=True); s[s == 0] = 1; return a/s
def lp(a): return np.log(np.clip(norm(a), 1e-6, 1))
def rd(p, nr):
    p = str(p); a = np.load(p) if p.endswith('.npy') else pd.read_csv(p).iloc[:, -3:].to_numpy()
    return (a.mean(0) if a.ndim == 3 else a)[:nr]
B = C/'refs/oof_bank'; K = C/'refs/kernel_out'
M = {'xgb-1': K/'xgb-v1-for-s6e6/oof_preds.npy', 'realmlp-0': B/'oof_preds_realmlp0_v12.csv', 'realmlp-1': K/'realmlp-v1-for-s6e6/oof_preds.npy', 'tabm-0': B/'oof_preds_tabm0_v2.csv', 'cat-0': K/'cat-v0-for-s6e6/catboost_oof_predictions.csv', 'realmlp-2': B/'oof_preds_realmlp2_v10.csv', 'tabicl-2': K/'tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy', 'lgbm-3': K/'lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy', 'logreg-1': K/'logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy', 'nn-1': K/'nn-v1-for-s6e6/train_oof/nn-1_oof.npy', 'xgb-5': K/'xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy', 'realmlp-5': K/'realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy', 'nn-2': K/'nn-v2-for-s6e6/train_oof/nn-2_oof.npy', 'cat-3': K/'cat-v3-for-s6e6/train_oof/cat-3_oof.npy', 'lgbm-5': B/'oof_preds_lgbm5_v1.csv', 'xgb-6': B/'oof_final_xgb6_v1.csv', 'tabm-1': B/'oof_final_tabm1_v1.csv'}
bank = []
for nm, op in M.items():
    o = norm(rd(op, n)); ba = bac(y, o.argmax(1))
    if 0.90 < ba < 0.972: bank.append(lp(o))
def cv(cols):
    X = np.concatenate(cols, 1); s = []
    for vi in fv:
        oth = np.setdiff1d(np.arange(n), vi)
        m = LogisticRegression(max_iter=400, C=1.0, class_weight='balanced').fit(X[oth], y[oth])
        s.append(bac(y[vi], m.predict_proba(X[vi]).argmax(1)))
    return float(np.mean(s)), float(np.std(s)/np.sqrt(len(s)))
n74 = lp(np.load(C/'nodes/node_0074/oof.npy')[:n]); n67 = lp(np.load(C/'nodes/node_0067/oof.npy')[:n])
b, bs = cv(bank); print(f'bank{len(bank)}_argmax cv={b:.6f} sem={bs:.6f}', flush=True)
for tag, cols in [('bank+n74', bank+[n74]), ('bank+n67+n74', bank+[n67, n74])]:
    c, _ = cv(cols); print(f'{tag} cv={c:.6f} delta={c-b:+.6f} beats2sem={c-b > 2*bs}', flush=True)
print('PROBE_DONE', flush=True)
