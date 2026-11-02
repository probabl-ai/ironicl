#!/usr/bin/env python3
"""Export Ironicl-preprocessed numeric data as an IRINPUT bundle.

Install the local Python package first. No TabICL checkout or PyTorch is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
    from ironicl.preprocessing import ClassifierPreprocessor

    x, labels, source = load_dataset(args.dataset)
    source_shape = [int(x.shape[0]), int(x.shape[1])]
    x_train, x_test, y_train_raw, y_test_raw = stratified_rows(
        x, labels, args.train_rows, args.test_rows, args.seed
    )
    x_train, x_test, selected_features = select_features(x_train, x_test, args.features)

    preprocessor = ClassifierPreprocessor(n_estimators=args.estimators, random_state=args.seed)
    train_groups = preprocessor.fit_transform(x_train, y_train_raw)
    test_groups = preprocessor.transform(x_test)
    y_train = preprocessor.encode_labels(y_train_raw)
    preprocessor.encode_labels(y_test_raw)  # Reject classes absent from training.

    # IRINPUT still requires expanded [B,T,H] data. Keep that legacy allocation
    # at the file-format boundary, not in the grouped Python preprocessing API.
    transformed_x = np.empty(
        (preprocessor.n_estimators_, len(x_train) + len(x_test), preprocessor.n_features_out_),
        dtype=np.float32,
    )
    transformed_y = np.empty((preprocessor.n_estimators_, len(y_train)), dtype=np.float32)
    for ensemble, group in enumerate(preprocessor.ensemble_groups_):
        features = preprocessor.feature_permutations_[ensemble]
        transformed_x[ensemble, :len(x_train)] = train_groups[group][:, features]
        transformed_x[ensemble, len(x_train):] = test_groups[group][:, features]
        transformed_y[ensemble] = preprocessor.class_permutations_[ensemble][y_train]
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
        "classes": [str(value) for value in preprocessor.classes_.tolist()],
        "seed": args.seed,
        "requested_estimators": args.estimators,
        "shape": [int(value) for value in transformed_x.shape],
        "training_rows": int(transformed_y.shape[1]),
        "test_rows": args.test_rows,
        "normalization_methods": list(preprocessor.group_names_),
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
