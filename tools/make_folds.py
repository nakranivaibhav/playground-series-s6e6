#!/usr/bin/env python3
"""Freeze a leak-correct cross-validation split — once, then immutable.

Picks the scheme the spec implies (the split seed controls ONLY the split):
    --time-col set            -> TimeSeriesSplit   (expanding window, past->future)
    --group-key set           -> GroupKFold        (a group never straddles folds)
    --task-type classification-> StratifiedKFold
    otherwise                 -> KFold
Override with --scheme {kfold,stratified,group,timeseries}.

Writes folds.json = {scheme, n_splits, seed, n_rows, folds:[{fold, val_idx:[...]}]}
with positional integer indices into the (optionally time-sorted) train frame.

Usage:
    uv run tools/make_folds.py --train comps/<slug>/data/train.csv \
        --target SalePrice --task-type regression \
        --out comps/<slug>/folds.json --n-splits 5 --seed 42
    uv run tools/make_folds.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


def choose_scheme(task_type: str | None, group_key: str | None, time_col: str | None) -> str:
    if time_col:
        return "timeseries"
    if group_key:
        return "group"
    if task_type and task_type.startswith("classification"):
        return "stratified"
    return "kfold"


def make_folds(
    df: pd.DataFrame,
    *,
    scheme: str,
    target: str | None = None,
    group_key: str | None = None,
    time_col: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    work = df
    if scheme == "timeseries" and time_col:
        work = df.sort_values(time_col).reset_index(drop=True)
    n = len(work)
    folds: list[dict] = []

    if scheme == "kfold":
        sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = sp.split(np.arange(n))
    elif scheme == "stratified":
        if not target:
            raise ValueError("stratified scheme requires --target")
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = sp.split(np.arange(n), work[target])
    elif scheme == "group":
        if not group_key:
            raise ValueError("group scheme requires --group-key")
        sp = GroupKFold(n_splits=n_splits)  # deterministic; no seed needed
        it = sp.split(np.arange(n), groups=work[group_key])
    elif scheme == "timeseries":
        sp = TimeSeriesSplit(n_splits=n_splits)
        it = sp.split(np.arange(n))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    for i, (_, val_idx) in enumerate(it):
        folds.append({"fold": i, "val_idx": [int(x) for x in val_idx]})

    return {"scheme": scheme, "n_splits": n_splits, "seed": seed, "n_rows": n, "folds": folds}


def _selftest() -> int:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "y_reg": rng.normal(size=n),
            "y_cls": rng.integers(0, 3, size=n),
            "grp": rng.integers(0, 25, size=n),
            "t": np.arange(n),
        }
    )

    # kfold: disjoint val folds covering every row exactly once
    f = make_folds(df, scheme="kfold", n_splits=5, seed=42)
    cover = sorted(i for fold in f["folds"] for i in fold["val_idx"])
    assert cover == list(range(n)), "kfold must cover all rows once"

    # stratified: full coverage + each class present across val folds
    f = make_folds(df, scheme="stratified", target="y_cls", n_splits=5, seed=42)
    cover = sorted(i for fold in f["folds"] for i in fold["val_idx"])
    assert cover == list(range(n))

    # group: a group never appears in both train and val of a fold
    f = make_folds(df, scheme="group", group_key="grp", n_splits=5)
    allidx = set(range(n))
    for fold in f["folds"]:
        val = set(fold["val_idx"])
        train = allidx - val
        val_groups = set(df.loc[list(val), "grp"])
        train_groups = set(df.loc[list(train), "grp"])
        assert val_groups.isdisjoint(train_groups), "group leak across fold"

    # timeseries: every val index comes strictly after its train indices
    f = make_folds(df, scheme="timeseries", time_col="t", n_splits=5)
    for fold in f["folds"]:
        val = fold["val_idx"]
        assert min(val) > 0, "ts val should not include the very first rows"

    # scheme selection
    assert choose_scheme("regression", None, None) == "kfold"
    assert choose_scheme("classification_binary", None, None) == "stratified"
    assert choose_scheme("regression", "patient", None) == "group"
    assert choose_scheme("regression", "patient", "date") == "timeseries"
    print("make_folds selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--train")
    p.add_argument("--target")
    p.add_argument("--task-type")
    p.add_argument("--group-key")
    p.add_argument("--time-col")
    p.add_argument("--scheme", choices=["kfold", "stratified", "group", "timeseries"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    a = p.parse_args(argv)

    if a.selftest:
        return _selftest()
    if not a.train or not a.out:
        p.error("--train and --out are required (or use --selftest)")

    df = pd.read_csv(a.train)
    scheme = a.scheme or choose_scheme(a.task_type, a.group_key, a.time_col)
    result = make_folds(
        df,
        scheme=scheme,
        target=a.target,
        group_key=a.group_key,
        time_col=a.time_col,
        n_splits=a.n_splits,
        seed=a.seed,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out}: scheme={scheme} n_splits={a.n_splits} n_rows={result['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
