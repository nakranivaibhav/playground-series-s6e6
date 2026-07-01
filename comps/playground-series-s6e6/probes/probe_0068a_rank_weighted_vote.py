"""
probe_0068a — rank-weighted vote (weight subs by their public LB score).
NOT a node. Journal line only.

Outputs: flip-count vs bank17 and vs A4-vote (node_0068 submission.csv).
No submission produced (budget guard).
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

COMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VOTE_BANK = os.path.join(COMP_DIR, "refs", "vote_bank")
BANK17_PATH = os.path.join(COMP_DIR, "champion", "submission.csv")
A4_PATH = os.path.join(COMP_DIR, "refs", "a4_vote", "submission_vote.csv")
N68_PATH = os.path.join(COMP_DIR, "nodes", "node_0068", "submission.csv")

TOP_N = 10
LB_THRESHOLD = 0.970
CLASSES = ["GALAXY", "QSO", "STAR"]

# Load top-N vote_bank CSVs (same family-stripped set as node_0068)
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

# Strip families
family_map = {}
for lb, fname, path, suffix in sorted(entries, reverse=True):
    base_key = str(lb)
    if base_key not in family_map:
        family_map[base_key] = (lb, path)

family_list = sorted(family_map.items(), key=lambda x: x[1][0], reverse=True)
top_n_entries = [(k, v) for k, v in family_list if v[0] >= LB_THRESHOLD][:TOP_N]

print(f"Top-{TOP_N} subs selected:")
for k, (lb, path) in top_n_entries:
    print(f"  LB={lb:.5f}  {os.path.basename(path)}")

# Load
subs = []
weights = []
for k, (lb, path) in top_n_entries:
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    subs.append(df["class"].values)
    weights.append(lb)

bank17 = pd.read_csv(BANK17_PATH).sort_values("id").reset_index(drop=True)
bank17_ids = bank17["id"].values
bank17_pred = bank17["class"].values

n68 = pd.read_csv(N68_PATH).sort_values("id").reset_index(drop=True)
n68_pred = n68["class"].values

n = len(bank17_ids)
votes = np.array(subs)  # (n_subs, n_rows)
weights_arr = np.array(weights)

# Rank-weighted vote: sum weights per class, pick argmax; ties -> bank17
print("\nRunning rank-weighted vote...")
class_set = sorted(set(c for row in subs for c in row))
tie_count = 0
final_pred = np.empty(n, dtype=object)

for i in range(n):
    score = defaultdict(float)
    for j, cls in enumerate(votes[:, i]):
        score[cls] += weights_arr[j]
    max_score = max(score.values())
    winners = [cls for cls, s in score.items() if s == max_score]
    if len(winners) == 1:
        final_pred[i] = winners[0]
    else:
        final_pred[i] = bank17_pred[i]
        tie_count += 1

flips_vs_bank17 = np.sum(final_pred != bank17_pred)
flips_vs_n68 = np.sum(final_pred != n68_pred)

print(f"\nRank-weighted vote results:")
print(f"  ties resolved by bank17: {tie_count}")
print(f"  flips vs bank17:   {flips_vs_bank17} ({flips_vs_bank17/n*100:.3f}%)")
print(f"  flips vs node_0068 (plain hard-vote): {flips_vs_n68} ({flips_vs_n68/n*100:.3f}%)")

# Agreement with bank17 high-confidence rows
# "high confidence" = all top-10 externals agree with bank17
agree_all = np.all(votes == bank17_pred[np.newaxis, :], axis=0)
n_hc = agree_all.sum()
print(f"\nBank17 high-confidence rows (all top-10 agree with bank17): {n_hc}/{n} ({n_hc/n*100:.1f}%)")
weighted_flip_in_hc = np.sum((final_pred != bank17_pred) & agree_all)
print(f"  rank-weighted flips in high-conf rows: {weighted_flip_in_hc} (should be 0)")

uniq, cnts = np.unique(final_pred, return_counts=True)
print("\nClass distribution:")
for u, c in zip(uniq, cnts):
    print(f"  {u}: {c}")
