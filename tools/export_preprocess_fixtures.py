#!/usr/bin/env python3
"""Regenerate the golden fixtures of Rust preprocessing from CPython, SciPy, and scikit-learn.

- tests/fixtures/python_random.json: draws from CPython's ``random.Random``,
  for src/preprocess/python_random.rs to reproduce exactly, and TabICL's
  feature shuffles, for TabICL preprocessing to reproduce.
- tests/fixtures/yeo_johnson.json: SciPy's Yeo-Johnson log-likelihood and
  lambda, and scikit-learn's ``PowerTransformer``, on float64 data, for
  src/preprocess/yeo_johnson.rs.

Each fixture records the versions that made it.
"""

from __future__ import annotations

import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SEEDS = (0, 1, 42, -7, 2**31 - 1, 2**32 + 5, 2**64 - 1, 2**70 + 3)


def random_streams() -> list[dict]:
    """Each seed's draws through ``_randbelow``, ``choice``, ``shuffle``, and ``sample``.

    One generator serves every call of a case, so the cases also pin how many
    words each call consumes."""
    cases = []
    for seed in SEEDS:
        rng = random.Random(seed)
        below = [rng._randbelow(n) for n in (1, 2, 3, 7, 8, 1000, 2**31, 2**32 + 1, 2**40 + 3, 2**63)]
        choices = [rng.choice(range(n)) for n in (1, 2, 5, 100)]
        shuffled = []
        for n in (0, 1, 2, 10, 100):
            values = list(range(n))
            rng.shuffle(values)
            shuffled.append(values)
        # k = n is the pool branch, k < n with a large population the set branch.
        samples = [rng.sample(range(n), k) for n, k in ((1, 1), (6, 6), (50, 50), (1000, 3), (5000, 40))]
        cases.append({"seed": str(seed), "randbelow": below, "choice": choices, "shuffle": shuffled, "sample": samples})
    return cases


def floats(values) -> list[float]:
    return [float(value) for value in np.asarray(values).ravel()]


def columns() -> dict[str, np.ndarray]:
    """One-dimensional inputs for each branch of SciPy's Yeo-Johnson fit."""
    rng = np.random.default_rng(7)
    normal = rng.standard_normal(60)
    skewed = np.exp(rng.standard_normal(60))
    with_zeros = rng.standard_normal(60)
    with_zeros[::5] = 0
    outlier = rng.standard_normal(60)
    outlier[3] = 1000
    return {
        "mixed": (normal - normal.mean()) / normal.std(),
        "skewed": (skewed - skewed.mean()) / skewed.std(),
        "positive": skewed,
        "negative": -skewed * 3,
        "with_zeros": with_zeros,
        "outlier": outlier,
        "clipped": np.clip(rng.standard_normal(60) * 80, -100, 100),
        "two_values": np.array([-1.0, 2.0]),
    }


def yeo_johnson() -> list[dict]:
    cases = []
    for name, x in columns().items():
        cases.append({
            "name": name,
            "x": floats(x),
            "lambda": float(stats.yeojohnson_normmax(x)),
            "llf": [[at, float(stats.yeojohnson_llf(at, x))] for at in (-1.5, 0.0, 0.7, 1.0, 2.0, 3.0)],
        })
    return cases


def power_transformers() -> list[dict]:
    rng = np.random.default_rng(11)
    x = np.column_stack([rng.standard_normal(50), np.exp(rng.standard_normal(50)), np.full(50, 3.0), rng.uniform(-100, 100, 50)])
    q = np.column_stack([rng.standard_normal(9) * 3, np.exp(rng.standard_normal(9)), np.full(9, 4.0), rng.uniform(-100, 100, 9)])
    cases = []
    for name, (train, query) in {"four_columns": (x, q), "one_column": (x[:, 1:2], q[:, 1:2])}.items():
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        fitted = power.fit_transform(train)
        cases.append({
            "name": name,
            "columns": train.shape[1],
            "x": floats(train),
            "query": floats(query),
            "lambdas": floats(power.lambdas_),
            "mean": floats(power._scaler.mean_),
            "scale": floats(power._scaler.scale_),
            "fit_transform": floats(fitted),
            "transform": floats(power.transform(query)),
        })
    return cases


def reference_feature_shuffles(features: int, estimators: int, seed: int) -> list[list[int]]:
    """TabICL 2.1.1's ``Shuffler(method="latin")``, rewritten without its other methods."""
    indices = list(range(features))
    if estimators == 1:
        return [indices]
    rng = random.Random(seed)
    if features > 4000:
        return [rng.sample(indices, features) for _ in range(estimators)]
    removed = []
    while len(indices) > 1:
        symbol = rng.choice(indices)
        indices.remove(symbol)
        removed.append(symbol)
    square = [indices]
    for symbol in reversed(removed):
        square.append(square[0].copy())
        for index, row in enumerate(square):
            row.insert(index, symbol)
    rng.shuffle(square)
    transposed = list(zip(*square))
    rng.shuffle(transposed)
    return [list(row) for row in transposed]


def digest(values) -> str:
    """FNV-1a over the values, for orders too large to store."""
    state = 0xCBF29CE484222325
    for value in values:
        state = ((state ^ value) * 0x100000001B3) % (1 << 64)
    return f"{state:016x}"


def feature_shuffles() -> list[dict]:
    cases = []
    for features, estimators, seed in ((1, 8, 0), (2, 8, 0), (5, 8, 42), (17, 8, 42), (5, 1, 3), (300, 8, -7), (4000, 3, 1), (4001, 3, 0), (5000, 8, 2**70 + 3)):
        orders = reference_feature_shuffles(features, estimators, seed)
        case = {"features": features, "estimators": estimators, "seed": str(seed), "count": len(orders)}
        if features <= 17:
            case["orders"] = orders
        else:
            case["digest"] = digest(value for order in orders for value in order)
        cases.append(case)
    return cases


def write(name: str, content: dict) -> None:
    content = {"python": platform.python_version(), **content}
    path = FIXTURES / name
    path.write_text(json.dumps(content, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    write("python_random.json", {"streams": random_streams(), "feature_shuffles": feature_shuffles()})
    packages = {package: version(package) for package in ("numpy", "scipy", "scikit-learn")}
    write("yeo_johnson.json", {**packages, "normmax": yeo_johnson(), "power_transformer": power_transformers()})


if __name__ == "__main__":
    main()
