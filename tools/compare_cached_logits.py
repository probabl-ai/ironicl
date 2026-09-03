#!/usr/bin/env python3
"""Compare Ironicl cached logits or final probabilities with an oracle."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8sIIQQQQ")
LINE = re.compile(r"^logits\[(\d+),(\d+)\]:\s+(.+)$")
PROBABILITY_LINE = re.compile(r"^probabilities\[(\d+)\]:\s+(.+)$")
STATIC_HEADER = struct.Struct("<8sII32s32sQQQQ5I3IQQI")


def read_oracle(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < HEADER.size + 32 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("invalid IRLOGITS checksum")
    magic, version, kind, b, q, c, count = HEADER.unpack_from(data)
    if (magic, version, kind) != (b"IRLOGITS", 1, 1) or count != b * q * c:
        raise ValueError("invalid IRLOGITS metadata")
    if len(data) != HEADER.size + count * 4 + 32:
        raise ValueError("invalid IRLOGITS length")
    return np.frombuffer(data, dtype="<f4", offset=HEADER.size, count=count).reshape(b, q, c)


def read_actual(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    actual = np.full(shape, np.nan, dtype=np.float32)
    for line in path.read_text().splitlines():
        match = LINE.match(line)
        if match:
            ensemble, row = map(int, match.group(1, 2))
            values = np.fromstring(match.group(3), sep=" ", dtype=np.float32)
            if ensemble >= shape[0] or row >= shape[1] or len(values) != shape[2]:
                raise ValueError("printed logit shape does not match IRLOGITS")
            actual[ensemble, row] = values
    if not np.isfinite(actual).all():
        raise ValueError("printed logits are incomplete or non-finite")
    return actual


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def read_probabilities(path: Path, shape: tuple[int, int]) -> np.ndarray:
    actual = np.full(shape, np.nan, dtype=np.float32)
    for line in path.read_text().splitlines():
        match = PROBABILITY_LINE.match(line)
        if match:
            row = int(match.group(1))
            values = np.fromstring(match.group(2), sep=" ", dtype=np.float32)
            if row >= shape[0] or len(values) != shape[1]:
                raise ValueError("printed probability shape does not match IRLOGITS")
            actual[row] = values
    if not np.isfinite(actual).all():
        raise ValueError("printed probabilities are incomplete or non-finite")
    return actual


def read_class_permutations(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    with path.open("rb") as file:
        header = file.read(STATIC_HEADER.size)
        if len(header) != STATIC_HEADER.size:
            raise ValueError("static-context header is truncated")
        fields = STATIC_HEADER.unpack(header)
        b, _, _, c = fields[5:9]
        if fields[0] != b"IRONICL\0" or fields[2] != 1 or (b, c) != (shape[0], shape[2]):
            raise ValueError("static context does not match IRLOGITS")
        permutations = np.fromfile(file, dtype="<u8", count=b * c)
    if permutations.size != b * c:
        raise ValueError("static-context class permutations are truncated")
    return permutations.reshape(b, c)


def aggregate_probabilities(logits: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    corrected = np.zeros_like(logits[0])
    for estimator, permutation in enumerate(permutations):
        corrected += logits[estimator][:, permutation]
    return softmax(corrected / logits.shape[0] / np.float32(0.9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    actual = parser.add_mutually_exclusive_group(required=True)
    actual.add_argument("--actual", type=Path, help="Printed estimator logits.")
    actual.add_argument("--probabilities-actual", type=Path, help="Printed final probabilities.")
    parser.add_argument("--context", type=Path, help="Static context needed for final probability comparison.")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    oracle = read_oracle(args.oracle)
    if args.probabilities_actual:
        if not args.context:
            parser.error("--probabilities-actual requires --context")
        permutations = read_class_permutations(args.context, oracle.shape)
        expected = aggregate_probabilities(oracle, permutations)
        actual_probabilities = read_probabilities(args.probabilities_actual, expected.shape)
        maximum = float(np.max(np.abs(actual_probabilities - expected)))
        label_agreement = float(
            np.mean(np.argmax(actual_probabilities, axis=-1) == np.argmax(expected, axis=-1))
        )
        print(
            f"shape={list(expected.shape)} max_abs={maximum:.9e} "
            f"label_agreement={label_agreement:.9f}"
        )
        if maximum > args.atol:
            raise SystemExit(f"maximum absolute error {maximum:.9e} exceeds {args.atol:.9e}")
        return

    assert args.actual is not None
    actual = read_actual(args.actual, oracle.shape)
    maximum = float(np.max(np.abs(actual - oracle)))
    oracle_probabilities = softmax(oracle)
    actual_probabilities = softmax(actual)
    probability_maximum = float(
        np.max(np.abs(actual_probabilities - oracle_probabilities))
    )
    label_agreement = float(
        np.mean(
            np.argmax(actual_probabilities, axis=-1)
            == np.argmax(oracle_probabilities, axis=-1)
        )
    )
    print(
        f"shape={list(oracle.shape)} max_abs={maximum:.9e} "
        f"max_estimator_probability_abs={probability_maximum:.9e} "
        f"estimator_label_agreement={label_agreement:.9f}"
    )
    if maximum > args.atol:
        raise SystemExit(f"maximum absolute error {maximum:.9e} exceeds {args.atol:.9e}")


if __name__ == "__main__":
    main()
