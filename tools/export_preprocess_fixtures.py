#!/usr/bin/env python3
"""Regenerate the golden fixtures of Rust preprocessing from CPython.

- tests/fixtures/python_random.json: draws from CPython's ``random.Random``,
  for src/preprocess/python_random.rs to reproduce exactly, and TabICL's
  feature shuffles, for TabICL preprocessing to reproduce.

Each fixture records the versions that made it.
"""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path

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


if __name__ == "__main__":
    main()
