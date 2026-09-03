#!/usr/bin/env python3
"""Export official TabICL-preprocessed real data as an IRINPUT bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


INPUT_MAGIC = b"IRINPUT\0"
VERSION = 1
INPUT_KIND = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("covertype", "miniboone", "fashion-mnist"), required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--test-rows", type=int, required=True)
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument("--features", type=int, help="Keep this many highest-variance features before TabICL preprocessing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--tabicl-root", type=Path, required=True)
    args = parser.parse_args()
    if args.train_rows < 2 or args.test_rows < 1 or args.estimators < 1:
        parser.error("row counts and estimator count must be positive")
    if args.features is not None and args.features < 1:
        parser.error("--features must be positive")
    return args


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_covtype, fetch_openml

    if name == "covertype":
        data = fetch_covtype()
        return np.asarray(data.data), np.asarray(data.target), "sklearn.fetch_covtype"
    data_id = 41150 if name == "miniboone" else 40996
    data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    return np.asarray(data.data), np.asarray(data.target), f"openml:{data_id}"


def stratified_rows(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: int,
    test_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if train_rows + test_rows > len(y):
        raise ValueError(f"requested {train_rows + test_rows} rows from a dataset with {len(y)} rows")
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_rows,
        test_size=test_rows,
        random_state=seed,
        stratify=y,
    )
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def select_features(x_train: np.ndarray, x_test: np.ndarray, count: int | None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if count is None or count >= x_train.shape[1]:
        return x_train, x_test, list(range(x_train.shape[1]))
    numeric = np.asarray(x_train, dtype=np.float64)
    variances = np.nanvar(numeric, axis=0)
    selected = np.argsort(variances, kind="stable")[-count:]
    selected.sort()
    return x_train[:, selected], x_test[:, selected], selected.tolist()


def encode_bundle(x: np.ndarray, y: np.ndarray) -> bytes:
    x = np.ascontiguousarray(x, dtype="<f4")
    y = np.ascontiguousarray(y, dtype="<f4")
    b, t, h = x.shape
    if y.ndim != 2 or y.shape[0] != b or y.shape[1] >= t:
        raise ValueError(f"invalid transformed shapes X={x.shape}, y={y.shape}")
    header = struct.pack(
        "<8sIIQQQQQQQQ",
        INPUT_MAGIC,
        VERSION,
        INPUT_KIND,
        b,
        t,
        h,
        y.shape[1],
        x.size,
        y.size,
        x.nbytes,
        y.nbytes,
    )
    payload = header + x.tobytes(order="C") + y.tobytes(order="C")
    return payload + hashlib.sha256(payload).digest()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("set PYTHONHASHSEED=0 before running to preserve TabICL ensemble ordering")
    sys.path.insert(0, str(args.tabicl_root.resolve()))
    from sklearn.preprocessing import LabelEncoder
    from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical

    x, labels, source = load_dataset(args.dataset)
    source_shape = [int(x.shape[0]), int(x.shape[1])]
    x_train, x_test, y_train_raw, y_test_raw = stratified_rows(
        x, labels, args.train_rows, args.test_rows, args.seed
    )
    x_train, x_test, selected_features = select_features(x_train, x_test, args.features)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    unknown_test = sorted(set(np.unique(y_test_raw)) - set(label_encoder.classes_))
    if unknown_test:
        raise ValueError(f"test split contains classes absent from training: {unknown_test}")

    numerical = TransformToNumerical(verbose=False)
    x_train_numeric = numerical.fit_transform(x_train)
    x_test_numeric = numerical.transform(x_test)
    generator = EnsembleGenerator(
        classification=True,
        n_estimators=args.estimators,
        norm_methods=["none", "power"],
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        random_state=args.seed,
    )
    generator.fit(x_train_numeric, y_train)
    groups = generator.transform(x_test_numeric, mode="both")
    transformed_x = np.concatenate([group[0] for group in groups.values()], axis=0).astype(np.float32)
    transformed_y = np.concatenate([group[1] for group in groups.values()], axis=0).astype(np.float32)
    if not np.isfinite(transformed_x).all() or not np.isfinite(transformed_y).all():
        raise ValueError("preprocessed tensors contain non-finite values")

    encoded = encode_bundle(transformed_x, transformed_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata = {
        "dataset": args.dataset,
        "source": source,
        "source_shape": source_shape,
        "selected_source_features": selected_features,
        "classes": [str(value) for value in label_encoder.classes_.tolist()],
        "seed": args.seed,
        "requested_estimators": args.estimators,
        "shape": [int(value) for value in transformed_x.shape],
        "training_rows": int(transformed_y.shape[1]),
        "test_rows": args.test_rows,
        "normalization_methods": list(groups.keys()),
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
