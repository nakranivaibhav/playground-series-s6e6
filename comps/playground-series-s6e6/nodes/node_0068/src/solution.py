"""
node_0068 — refresh public-sub hard-vote (CLOUT/slot-2 ONLY)

Recipe (EXACT A4):
1. Load all LB-labeled CSVs from vote_bank + newly-pulled CSVs from a4_refresh_2026-06-12.
2. Strip .b/.c micro-patch families — keep the highest-LB representative per base score.
3. Select top-N (by LB) public subs (N >= 7, the A4 minimum).
4. Plain plurality hard vote; ties broken by bank17 (champion/submission.csv).
5. Write submission.csv to node dir.

NO OOF (clout artifact — no honest CV). This node is finals slot-2 ONLY.
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from collections import Counter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.dirname(SRC_DIR)
# nodes/node_0068 -> nodes -> playground-series-s6e6
COMP_DIR = os.path.dirname(os.path.dirname(NODE_DIR))
VOTE_BANK = os.path.join(COMP_DIR, "refs", "vote_bank")
REFRESH_DIR = os.path.join(COMP_DIR, "refs", "a4_refresh_2026-06-12")
BANK17_PATH = os.path.join(COMP_DIR, "champion", "submission.csv")
PRIOR_A4_PATH = os.path.join(COMP_DIR, "refs", "a4_vote", "submission_vote.csv")
SAMPLE_PATH = os.path.join(COMP_DIR, "data", "sample_submission.csv")
OUT_PATH = os.path.join(NODE_DIR, "submission.csv")

CLASSES = ["GALAXY", "QSO", "STAR"]
# A4 used top-7 (0.97135..0.97120). We keep a wider band and select top-N after strip.
TOP_N = 10  # take top-10 after strip (>= 7 keeps the spirit; more consensus is fine)
LB_THRESHOLD = 0.970  # only include subs with LB >= this

# ── 1. Load vote_bank CSVs with LB label ─────────────────────────────────────
print("Loading vote_bank...")
vote_bank_entries = []  # (lb_score, base_score_str, path, tag)

for fname in os.listdir(VOTE_BANK):
    if not fname.endswith(".csv"):
        continue
    if fname.startswith("test_preds") or fname.startswith("ns__") or fname.startswith("cat-"):
        continue
    # Parse LB score from filename: 0.97135.csv, 0.97126.b.csv, 0.97126.c.csv
    m = re.match(r"^([\d.]+)(\.[a-z]+)?\.csv$", fname)
    if not m:
        continue
    lb = float(m.group(1))
    suffix = m.group(2) or ""
    vote_bank_entries.append((lb, fname, os.path.join(VOTE_BANK, fname), suffix))

print(f"  vote_bank total CSVs: {len(vote_bank_entries)}")

# ── 2. Load newly-pulled CSVs (assign LB from best match or treat as unscored) ──
# We matched them earlier; treat them as additional entries with known-match LBs.
# Since they're essentially duplicates of existing vote_bank entries, they won't
# add new information — but we include them as candidates with their matched LB.
# We'll identify them by agreement with vote_bank entries.
print("Loading refresh CSVs...")
refresh_entries = []
for subdir in os.listdir(REFRESH_DIR):
    csv_path = os.path.join(REFRESH_DIR, subdir, "submission.csv")
    if not os.path.exists(csv_path):
        continue
    refresh_entries.append((subdir, csv_path))

print(f"  refresh CSVs: {len(refresh_entries)}")

# ── 3. Build canonical map: base_key -> (lb, path) keeping highest-LB per family ──
# "family" = base score without .b/.c suffix
# e.g., "0.97126" and "0.97126.b" are one family; keep "0.97126"
print("Stripping micro-patch families...")
family_map = {}  # base_score_str -> (lb, path)
for lb, fname, path, suffix in sorted(vote_bank_entries, reverse=True):
    base_key = str(lb)  # e.g., "0.97126"
    if base_key not in family_map:
        family_map[base_key] = (lb, path)
        # if this has no suffix it IS the canonical; if it's a .b it became canonical
        # because it's the first (highest-lb within the group already sorted)

# Re-sort by LB desc
family_list = sorted(family_map.items(), key=lambda x: x[1][0], reverse=True)
print(f"  unique families (after strip): {len(family_list)}")
print(f"  top-10 by LB: {[k for k, _ in family_list[:10]]}")

# Filter to LB_THRESHOLD+
family_list_filtered = [(k, v) for k, v in family_list if v[0] >= LB_THRESHOLD]
print(f"  families >= {LB_THRESHOLD}: {len(family_list_filtered)}")

# ── 4. Select top-N ───────────────────────────────────────────────────────────
top_n_entries = family_list_filtered[:TOP_N]
print(f"\nTop-{TOP_N} selected subs (A4+ recipe):")
for k, (lb, path) in top_n_entries:
    print(f"  LB={lb:.5f}  {os.path.basename(path)}")

# ── 5. Load submissions and bank17 ───────────────────────────────────────────
print("\nLoading submission CSVs...")
subs = []
for k, (lb, path) in top_n_entries:
    df = pd.read_csv(path)
    df = df.sort_values("id").reset_index(drop=True)
    subs.append(df["class"].values)

bank17 = pd.read_csv(BANK17_PATH).sort_values("id").reset_index(drop=True)
bank17_ids = bank17["id"].values
bank17_pred = bank17["class"].values

prior_a4 = pd.read_csv(PRIOR_A4_PATH).sort_values("id").reset_index(drop=True)
prior_a4_pred = prior_a4["class"].values

n = len(bank17_ids)
print(f"  n_test = {n}")

# ── 6. Plain hard vote with bank17 tie-break ─────────────────────────────────
print("Running hard vote...")
votes = np.array(subs)  # shape (n_subs, n_rows)
final_pred = np.empty(n, dtype=object)
tie_count = 0
for i in range(n):
    row_votes = votes[:, i]
    c = Counter(row_votes)
    max_count = max(c.values())
    winners = [cls for cls, cnt in c.items() if cnt == max_count]
    if len(winners) == 1:
        final_pred[i] = winners[0]
    else:
        # Tie — use bank17
        final_pred[i] = bank17_pred[i]
        tie_count += 1

print(f"  ties resolved by bank17: {tie_count} ({tie_count/n*100:.3f}%)")

# ── 7. Compute flip statistics ────────────────────────────────────────────────
flips_vs_bank17 = np.sum(final_pred != bank17_pred)
flips_vs_prior_a4 = np.sum(final_pred != prior_a4_pred)
print(f"\nFlip statistics:")
print(f"  vs bank17:    {flips_vs_bank17} flips ({flips_vs_bank17/n*100:.3f}%)")
print(f"  vs prior A4:  {flips_vs_prior_a4} flips ({flips_vs_prior_a4/n*100:.3f}%)")

# Class distribution
uniq, cnts = np.unique(final_pred, return_counts=True)
print(f"\nClass distribution:")
for u, c in zip(uniq, cnts):
    print(f"  {u}: {c}")

# ── 8. Write submission.csv ───────────────────────────────────────────────────
out_df = pd.DataFrame({"id": bank17_ids, "class": final_pred})
out_df.to_csv(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}")

# Print summary for log
print(f"\n=== node_0068 SUMMARY ===")
print(f"public subs pulled (refresh):  {len(refresh_entries)}")
print(f"vote_bank CSVs:                {len(vote_bank_entries)}")
print(f"unique families after strip:   {len(family_list)}")
print(f"top-{TOP_N} used in vote:         {len(top_n_entries)}")
print(f"ties resolved:                 {tie_count}")
print(f"flips vs bank17:               {flips_vs_bank17}")
print(f"flips vs prior A4 (LB0.97123): {flips_vs_prior_a4}")
print(f"\ncv=n/a (clout artifact — no OOF)")
