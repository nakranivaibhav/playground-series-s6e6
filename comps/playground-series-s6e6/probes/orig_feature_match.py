"""probe: can we recover REAL labels by matching comp rows to the original
fedesoriano SDSS17 dataset on FEATURE VALUES (u,g,r,i,z,redshift)?

n083 already killed the COORDINATE match (alpha,delta vs the full 5.1M SDSS DR17
catalog → ~50% label agreement = generator assigns class independently of sky
position). This is a DIFFERENT test: match on the photometric feature values
themselves against the curated 100k original the comp was generated from.

Signature of a REAL identity match: at high precision (5-6 decimals) the comp
TRAIN rows that match an original row should agree with the original CLASS at
~100%. If matches are ~0, or agreement is ~base-rate (~65% GALAXY), the generator
produced fresh synthetic rows and the external-data avenue is closed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
ORIG = Path("/tmp/sdss17_orig/star_classification.csv")
COLS = ["u", "g", "r", "i", "z", "redshift"]

tr = pd.read_csv(COMP / "data/train.csv")
te = pd.read_csv(COMP / "data/test.csv")
orig = pd.read_csv(ORIG)
print(f"comp train={len(tr)}  test={len(te)}  orig={len(orig)}", flush=True)

# --- scale sanity: are comp features on the SAME system as the original? ------
print("\n=== feature ranges (comp train vs original) ===", flush=True)
for c in COLS:
    print(f"  {c:9s} comp[{tr[c].min():.4f},{tr[c].max():.4f}] mean={tr[c].mean():.4f}  "
          f"orig[{orig[c].min():.4f},{orig[c].max():.4f}] mean={orig[c].mean():.4f}", flush=True)

# original has some sentinel -9999 rows in u/g/z; drop them for matching
orig_clean = orig[(orig[COLS] > -100).all(axis=1)].copy()
print(f"\norig rows after dropping -9999 sentinels: {len(orig_clean)}", flush=True)

base_rate = tr["class"].value_counts(normalize=True)
print(f"comp train class base rate: {dict(base_rate.round(3))}", flush=True)

# --- match at several precisions on (u,g,r,i,z,redshift) ----------------------
print("\n=== FEATURE-MATCH (round to N decimals, key=u,g,r,i,z,redshift) ===", flush=True)
print(f"{'dec':>3s} {'train_match':>12s} {'train%':>7s} {'agree':>7s} {'test_match':>11s} {'test%':>7s}", flush=True)
for dec in [6, 5, 4, 3, 2]:
    o = orig_clean.copy()
    for c in COLS:
        o[c] = o[c].round(dec)
    # original key -> majority class
    lut = (o.groupby(COLS)["class"]
             .agg(lambda s: s.value_counts().index[0])
             .rename("orig_class"))

    def matched(df):
        d = df.copy()
        for c in COLS:
            d[c] = d[c].round(dec)
        return d.merge(lut, left_on=COLS, right_index=True, how="inner")

    m_tr = matched(tr)
    agree = (m_tr["class"] == m_tr["orig_class"]).mean() if len(m_tr) else float("nan")
    m_te = matched(te)
    print(f"{dec:>3d} {len(m_tr):>12d} {100*len(m_tr)/len(tr):>6.2f}% {agree:>7.3f} "
          f"{len(m_te):>11d} {100*len(m_te)/len(te):>6.2f}%", flush=True)

print("\nverdict: a REAL identity join needs high-precision (dec>=5) matches with "
      "agree~1.0 AND non-trivial test coverage. base-rate agreement (~0.65) = coincidental.", flush=True)
