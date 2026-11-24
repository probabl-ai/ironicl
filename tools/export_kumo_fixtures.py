#!/usr/bin/env python3
"""Regenerate the Kumo-Tabular parity fixtures from NVIDIA's code.

Needs NVIDIA's ``structured-data-models`` at commit
3fdf108a1f1b719b1b983373b71bbba79fcab7e6, with pandas, in its own environment:

    pip install pandas "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models@3fdf108a1f1b719b1b983373b71bbba79fcab7e6"
    python tools/export_kumo_fixtures.py tests/fixtures/kumo_tabular

Each case fits on the CPU in FP32 with a seeded ``torch.Generator`` and
records, per ensemble member, the random draws, the preprocessed rows the
model sees, and the model's outputs, then the final prediction. Rust feeds its
own pipeline the recorded draws, so everything after them can match exactly.
Non-finite input values are written as the strings "nan", "inf", and "-inf".
"""

from __future__ import annotations

import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sdm
import sdm.processing as sp
from sdm.models.base.callback import Callback
from sdm.models.base.ecoc import ECOC

DRAWS: dict[str, list] = {}


def record(cls, method: str, key: str, value) -> None:
    """Wrap ``cls.method`` to append ``value(self, result)`` to ``DRAWS[key]``."""
    original = getattr(cls, method)

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        DRAWS.setdefault(key, []).append(value(self, result))
        return result

    setattr(cls, method, wrapped)


record(sp.FlipSign, "_fit_ensemble", "flip_signs", lambda self, _: [s.flatten().tolist() for s in self._signs])
record(sp.ShuffleColumns, "_fit_ensemble", "column_orders", lambda self, _: [list(p) for p in self._host_permutations])
record(
    sp.ShuffleCategories,
    "_fit_ensemble",
    "class_orders",
    lambda self, _: [self._permutations[i][0].tolist() for i in self._permutation_ids],
)
record(ECOC, "_draw_codebook", "codebooks", lambda self, result: result.tolist())


def cell(value):
    if isinstance(value, str) or value is None:
        return value
    value = float(value)
    return value if math.isfinite(value) else str(value)


def table_rows(table) -> list:
    return [[cell(v) for v in row] for row in table.numerical.tolist()]


class Capture(Callback):
    def __init__(self) -> None:
        self.contexts, self.queries, self.outputs = [], [], []

    def on_context_preprocessing_end(self, model, x, y, x_context_only, related_tables):
        self.contexts.append(
            {
                "columns": list(x.numerical_columns),
                "x": table_rows(x),
                "y": y.categorical.code.flatten().tolist() if y.categorical.size(-1) else y.numerical.flatten().tolist(),
            }
        )
        return x, y, x_context_only, related_tables

    def on_query_preprocessing_end(self, model, x, related_tables):
        self.queries.append(table_rows(x))
        return x, related_tables

    def on_model_forward_end(self, model, out):
        self.outputs.append({"columns": list(out.numerical_columns), "values": out.numerical.tolist()})
        return out


def compact(value):
    """FP32 outputs with the 9 significant digits that round-trip them."""
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return float(f"{value:.9g}") if isinstance(value, float) and math.isfinite(value) else value


def frame(columns: dict) -> pd.DataFrame:
    return pd.DataFrame(columns)


def numeric(rng, rows: int) -> dict:
    values = {
        "normal": rng.standard_normal(rows),
        "skewed": np.exp(2 * rng.standard_normal(rows)),
        "missing": np.where(rng.random(rows) < 0.2, np.nan, rng.standard_normal(rows) * 5 + 3),
        "constant": np.full(rows, 7.0),
        "ties": rng.integers(0, 5, rows).astype(float),
        "outlier": rng.standard_normal(rows),
        "empty": np.full(rows, np.nan),
    }
    values["outlier"][3] = np.inf
    values["outlier"][5] = 1e6
    values["ties"][-1] = -np.inf
    return values


def categorical(rng, rows: int) -> dict:
    small = rng.choice(["red", "green", "blue", "amber"], rows).astype(object)
    small[rng.random(rows) < 0.15] = None
    wide = np.array([f"k{i:02d}" for i in rng.integers(0, 70, rows)], dtype=object)
    return {"color": small, "key": wide, "flag": rng.choice(["yes", "no"], rows).astype(object)}


def case(name, size, task, estimators, seed, x: pd.DataFrame, y, context_rows: int) -> dict:
    DRAWS.clear()
    stypes = sdm.infer_stypes(x)
    table = sdm.TableTensor.from_pandas(x, stypes=stypes)
    target = pd.DataFrame({"y": y})
    y_stype = {"y": "categorical" if task == "classification" else "numerical"}
    y_table = sdm.TableTensor.from_pandas(target, stypes=y_stype)
    model = sdm.models.KumoTabular(task=task, size=size, device="cpu")
    capture = Capture()
    generator = torch.Generator("cpu").manual_seed(seed)
    with torch.no_grad():
        model.fit(
            table[:context_rows],
            y_table[:context_rows],
            num_estimators=estimators,
            generator=generator,
            callbacks=[capture],
        )
        prediction = model.predict(table[context_rows:], callbacks=[capture])
    return {
        "name": name,
        "size": size,
        "task": task,
        "estimators": estimators,
        "seed": seed,
        "context_rows": context_rows,
        "columns": [
            {"name": column, "categorical": str(stypes[column]) != "numerical", "values": [cell(v) for v in x[column]]}
            for column in x.columns
        ],
        "y": [cell(v) for v in y],
        "draws": dict(DRAWS),
        "members": compact(
            [
                {**context, "query": query, "output": output}
                for context, query, output in zip(capture.contexts, capture.queries, capture.outputs, strict=True)
            ]
        ),
        "prediction": compact({"columns": list(prediction.numerical_columns), "values": prediction.numerical.tolist()}),
    }


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    rows = 64
    x = frame(numeric(rng, rows))
    labels = rng.integers(0, 3, rows)
    mixed_rng = np.random.default_rng(1)
    mixed_rows = 140
    mixed = frame({**{k: v for k, v in numeric(mixed_rng, mixed_rows).items() if k in ("normal", "missing")},
                   **categorical(mixed_rng, mixed_rows)})
    mixed_labels = mixed_rng.choice(["ant", "bee", "cat", "dog"], mixed_rows).astype(object)
    many_rng = np.random.default_rng(2)
    many_rows = 108
    many = frame({k: v for k, v in numeric(many_rng, many_rows).items() if k in ("normal", "skewed", "missing", "ties")})
    many_labels = many_rng.integers(0, 12, many_rows) * 10
    target = x["normal"].fillna(0).to_numpy() * 3 + np.nan_to_num(x["skewed"].to_numpy(), posinf=0) + 100
    cases = [
        case("numeric", "small", "classification", 4, 0, x, labels, 48),
        case("mixed", "small", "classification", 5, 1, mixed, mixed_labels, 120),
        case("many_classes", "small", "classification", 2, 2, many, many_labels, 96),
        case("mixed_medium", "medium", "classification", 2, 3, mixed, mixed_labels, 120),
        case("mixed_large", "large", "classification", 2, 4, mixed, mixed_labels, 120),
        # 999 quantiles per row: 8 query rows suffice.
        case("numeric_regression", "small", "regression", 5, 5, x[:56], target[:56], 48),
        case("mixed_regression_large", "large", "regression", 2, 6, mixed[:128], mixed_rng.standard_normal(mixed_rows)[:128], 120),
    ]
    meta = {"structured-data-models": version("structured-data-models"), "torch": torch.__version__}
    for fixture in cases:
        fixture["versions"] = meta
        (out / f"{fixture['name']}.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
        print(fixture["name"], [len(m["columns"]) for m in fixture["members"]])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
