#!/usr/bin/env python3
"""Validate a submission.csv against the competition's sample_submission.

The ONE submission gate, reused by the baseline and every experiment node — so a
node can never "score great" yet emit a malformed file that wastes a slot.
Checks: exact column header, row count, id set equality, and no NaN/inf.

Usage:
    uv run tools/validate_submission.py --submission sub.csv \
        --sample comps/<slug>/data/sample_submission.csv --id Id
    uv run tools/validate_submission.py --selftest

Exit 0 = valid; exit 1 = problems (printed one per line).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate(sub: pd.DataFrame, sample: pd.DataFrame, id_col: str) -> list[str]:
    problems: list[str] = []

    if list(sub.columns) != list(sample.columns):
        problems.append(
            f"columns {list(sub.columns)} != expected {list(sample.columns)}"
        )
        # column mismatch makes the rest unreliable
        return problems

    if len(sub) != len(sample):
        problems.append(f"row count {len(sub)} != expected {len(sample)}")

    if id_col in sub.columns and id_col in sample.columns:
        sub_ids, exp_ids = set(sub[id_col]), set(sample[id_col])
        if sub_ids != exp_ids:
            missing = len(exp_ids - sub_ids)
            extra = len(sub_ids - exp_ids)
            problems.append(f"id set mismatch: {missing} missing, {extra} unexpected")
        if sub[id_col].duplicated().any():
            problems.append(f"duplicate ids in '{id_col}'")

    value_cols = [c for c in sub.columns if c != id_col]
    block = sub[value_cols]
    if block.isna().any().any():
        problems.append("NaN present in prediction columns")
    num = block.select_dtypes(include=[np.number])
    if num.shape[1] and not np.isfinite(num.to_numpy()).all():
        problems.append("inf present in prediction columns")

    return problems


def _selftest() -> int:
    sample = pd.DataFrame({"Id": [1, 2, 3], "SalePrice": [0.0, 0.0, 0.0]})

    good = pd.DataFrame({"Id": [1, 2, 3], "SalePrice": [100.0, 200.0, 300.0]})
    assert validate(good, sample, "Id") == [], "good submission should pass"

    bad_cols = pd.DataFrame({"id": [1, 2, 3], "price": [1, 2, 3]})
    assert validate(bad_cols, sample, "Id"), "wrong columns should fail"

    bad_rows = pd.DataFrame({"Id": [1, 2], "SalePrice": [1.0, 2.0]})
    assert any("row count" in p for p in validate(bad_rows, sample, "Id"))

    bad_ids = pd.DataFrame({"Id": [1, 2, 9], "SalePrice": [1.0, 2.0, 3.0]})
    assert any("id set" in p for p in validate(bad_ids, sample, "Id"))

    nan_vals = pd.DataFrame({"Id": [1, 2, 3], "SalePrice": [1.0, np.nan, 3.0]})
    assert any("NaN" in p for p in validate(nan_vals, sample, "Id"))

    inf_vals = pd.DataFrame({"Id": [1, 2, 3], "SalePrice": [1.0, np.inf, 3.0]})
    assert any("inf" in p for p in validate(inf_vals, sample, "Id"))

    print("validate_submission selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--submission")
    p.add_argument("--sample")
    p.add_argument("--id", default="id")
    a = p.parse_args(argv)

    if a.selftest:
        return _selftest()
    if not a.submission or not a.sample:
        p.error("--submission and --sample are required (or use --selftest)")

    problems = validate(pd.read_csv(a.submission), pd.read_csv(a.sample), a.id)
    if problems:
        print("INVALID:")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print(f"OK: {Path(a.submission).name} matches {Path(a.sample).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
