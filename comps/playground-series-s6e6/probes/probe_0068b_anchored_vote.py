"""
probe_0068b — bank17-anchored vote: flip only when >= k externals agree AGAINST bank17.
NOT a node. Journal line only.

Sweep k=2..8. For each k, report flip-count vs bank17 and vs node_0068.
No submission produced (budget guard).
"""

import os
import re
import numpy as np
import pandas as pd

COMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VOTE_BANK = os.path.join(COMP_DIR, "refs", "vote_bank")
BANK17_PATH = os.path.join(COMP_DIR, "champion", "submission.csv")
N68_PATH = os.path.join(COMP_DIR, "nodes", "node_0068", "submission.csv")

TOP_N = 10
LB_THRESHOLD = 0.970

# Load top-N (same as node_0068)
entries = []
for fname in os.listdir(VOTE_BANK):
    if not fname.endswith(".csv"):
        continue
    if fname.startswith("test_preds") or fname.startswith("ns__") or fname.startswith("cat-"):
        continue
    m = re.match(r"^([\d.]+)(\.[a-z]+)?\.csv$", fname)
    if not m:
        continue
    lb = float(m.group(1))
    suffix = m.group(2) or ""
    entries.append((lb, fname, os.path.join(VOTE_BANK, fname), suffix))

family_map = {}
for lb, fname, path, suffix in sorted(entries, reverse=True):
    base_key = str(lb)
    if base_key not in family_map:
        family_map[base_key] = (lb, path)

family_list = sorted(family_map.items(), key=lambda x: x[1][0], reverse=True)
top_n_entries = [(k, v) for k, v in family_list if v[0] >= LB_THRESHOLD][:TOP_N]

subs = []
for k, (lb, path) in top_n_entries:
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    subs.append(df["class"].values)

bank17 = pd.read_csv(BANK17_PATH).sort_values("id").reset_index(drop=True)
bank17_pred = bank17["class"].values
n68 = pd.read_csv(N68_PATH).sort_values("id").reset_index(drop=True)
n68_pred = n68["class"].values

votes = np.array(subs)  # (n_subs, n)
n = len(bank17_pred)
n_subs = len(subs)

# High-confidence rows: all externals agree with bank17
agree_all = np.all(votes == bank17_pred[np.newaxis, :], axis=0)
n_hc = agree_all.sum()

print(f"n_subs = {n_subs}, n = {n}")
print(f"Bank17 high-confidence rows (all {n_subs} agree): {n_hc}/{n} ({n_hc/n*100:.1f}%)")
print()

print(f"{'k':>4}  {'flips_vs_b17':>14}  {'flips_vs_n68':>14}  {'hc_flips':>10}")
print("-" * 50)

for k in range(2, n_subs + 1):
    final_pred = bank17_pred.copy()
    # For each row, count how many externals disagree with bank17 and agree on same alt class
    for i in range(n):
        b17_cls = bank17_pred[i]
        # Count agreements per non-bank17 class
        from collections import Counter
        ext_votes = votes[:, i]
        counter = Counter(ext_votes)
        # votes against bank17 = sum of non-bank17
        alt_votes = {cls: cnt for cls, cnt in counter.items() if cls != b17_cls}
        if alt_votes:
            best_alt_cls = max(alt_votes, key=alt_votes.get)
            if alt_votes[best_alt_cls] >= k:
                final_pred[i] = best_alt_cls

    flips_vs_b17 = np.sum(final_pred != bank17_pred)
    flips_vs_n68 = np.sum(final_pred != n68_pred)
    hc_flips = np.sum((final_pred != bank17_pred) & agree_all)
    print(f"{k:>4}  {flips_vs_b17:>14}  {flips_vs_n68:>14}  {hc_flips:>10}")

print()
print("Agreement with bank17 high-confidence rows (anchored vote never touches these).")
print("k=7 (majority of 10) is the A4 hard-vote equivalent threshold.")
