#!/usr/bin/env python3
"""Export a deterministic TabICLv2 FP32 CPU oracle fixture for Ironicl."""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any


CLASSIFIER_CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
INPUT_MAGIC = b"IRINPUT\0"
LOGITS_MAGIC = b"IRLOGITS"
TRACE_MAGIC = b"IRTRACE\0"
VERSION = 1
INPUT_KIND = 1
LOGITS_KIND = 1
TRACE_DTYPE_F32 = b"f32"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted official TabICLv2 classifier checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=root / "fixtures" / "tabicl-oracle.irinput",
        help="Output IRINPUT v1 bundle path.",
    )
    parser.add_argument(
        "--logits-output",
        type=Path,
        default=root / "fixtures" / "tabicl-oracle.irlogits",
        help="Output checked logits fixture path.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Optional IRTRACE v1 path for coarse model-stage tensors.",
    )
    parser.add_argument(
        "--tabicl-root",
        type=Path,
        default=root.parent / "tabicl" / "src",
        help="Path to the sibling tabicl/src directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checked(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = data + hashlib.sha256(data).digest()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def encode_input(x_values: list[float], y_values: list[float], b: int, t: int, h: int, n: int) -> bytes:
    """Encode exactly the IRINPUT v1 layout in src/input_bundle.rs."""
    header = struct.pack(
        "<8sIIQQQQQQQQ",
        INPUT_MAGIC,
        VERSION,
        INPUT_KIND,
        b,
        t,
        h,
        n,
        len(x_values),
        len(y_values),
        len(x_values) * 4,
        len(y_values) * 4,
    )
    return header + struct.pack(f"<{len(x_values)}f", *x_values) + struct.pack(f"<{len(y_values)}f", *y_values)


def encode_logits(values: list[float], b: int, test_rows: int, classes: int) -> bytes:
    """Encode the documented IRLOGITS v1 layout, before its SHA-256 trailer."""
    header = struct.pack("<8sIIQQQQ", LOGITS_MAGIC, VERSION, LOGITS_KIND, b, test_rows, classes, len(values))
    return header + struct.pack(f"<{len(values)}f", *values)


def tensor_values(tensor: Any) -> list[float]:
    """Return a contiguous CPU FP32 tensor as row-major Python values."""
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1).tolist()


def encode_trace(tensors: list[tuple[str, Any]]) -> bytes:
    """Encode IRTRACE v1 named row-major FP32 tensors, before its SHA-256 trailer.

    Layout: magic, u32 version, u32 tensor count; then, per tensor, u16 name
    length, u16 dtype length, u32 rank, u64 element count, ASCII name and dtype,
    rank little-endian u64 dimensions, and little-endian FP32 data.
    """
    data = bytearray(struct.pack("<8sII", TRACE_MAGIC, VERSION, len(tensors)))
    for name, tensor in tensors:
        name_bytes = name.encode("ascii")
        shape = tuple(tensor.shape)
        values = tensor_values(tensor)
        data.extend(struct.pack("<HHIQ", len(name_bytes), len(TRACE_DTYPE_F32), len(shape), len(values)))
        data.extend(name_bytes)
        data.extend(TRACE_DTYPE_F32)
        data.extend(struct.pack(f"<{len(shape)}Q", *shape))
        data.extend(struct.pack(f"<{len(values)}f", *values))
    return bytes(data)


def print_tensor_stats(name: str, tensor: Any) -> None:
    values = tensor_values(tensor)
    print(
        f"{name} shape={list(tensor.shape)} min={min(values):.9g} "
        f"max={max(values):.9g} sum={sum(values):.9g} first8={values[:8]}"
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    tabicl_root = args.tabicl_root.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if not (tabicl_root / "tabicl" / "__init__.py").is_file():
        raise SystemExit(f"--tabicl-root must contain tabicl/__init__.py: {tabicl_root}")

    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CLASSIFIER_CHECKPOINT_SHA256:
        raise SystemExit(
            "checkpoint SHA-256 does not match the official TabICLv2 classifier: "
            f"{checkpoint_hash}"
        )

    sys.path.insert(0, str(tabicl_root))
    import torch
    from tabicl._model.inference_config import InferenceConfig
    from tabicl._model.tabicl import TabICL

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TabICL(**checkpoint_data["config"]).cpu().float()
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()

    # Direct neural input: [B=1, T=5, H=3], with four training rows then one test row.
    x_values = [
        0.25, -1.0, 2.0,
        1.5, 0.0, -0.5,
        -0.75, 2.0, 0.25,
        0.0, -0.25, 1.0,
        0.5, 1.25, -1.5,
    ]
    y_values = [0.0, 1.0, 0.0, 1.0]
    b, t, h, n = 1, 5, 3, 4
    x = torch.tensor(x_values, dtype=torch.float32, device="cpu").reshape(b, t, h)
    y_train = torch.tensor(y_values, dtype=torch.float32, device="cpu").reshape(b, n)
    cpu_config = {"device": "cpu", "use_amp": False, "use_fa3": False, "offload": False, "verbose": False}
    inference_config = InferenceConfig(
        COL_CONFIG=cpu_config,
        ROW_CONFIG=cpu_config,
        ICL_CONFIG=cpu_config,
    )
    with torch.no_grad():
        logits = model(
            x,
            y_train,
            embed_with_test=False,
            return_logits=True,
            inference_config=inference_config,
        )
    logits = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(logits.shape[:2]) != (b, t - n):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    classes = logits.shape[2]
    logits_values = logits.reshape(-1).tolist()

    if args.trace_output is not None:
        with torch.no_grad():
            column_output = model.col_embedder(
                x,
                y_train=y_train,
                embed_with_test=False,
                feature_shuffles=None,
                mgr_config=inference_config.COL_CONFIG,
            )
            row_output = model.row_interactor(column_output, mgr_config=inference_config.ROW_CONFIG)
            # _icl_predictions is the ICL predictor before test-row and class slicing.
            model.icl_predictor.inference_mgr.configure(**inference_config.ICL_CONFIG)
            raw_decoder_output = model.icl_predictor._icl_predictions(row_output.clone(), y_train)
        traces = [
            ("col_embedder", column_output),
            ("row_interactor", row_output),
            ("decoder_raw", raw_decoder_output),
            ("logits", logits),
        ]
        trace_hash = write_checked(args.trace_output, encode_trace(traces))
        print(f"trace={args.trace_output} sha256={trace_hash}")
        for name, tensor in traces:
            print_tensor_stats(name, tensor)

    bundle_hash = write_checked(args.bundle_output, encode_input(x_values, y_values, b, t, h, n))
    logits_hash = write_checked(args.logits_output, encode_logits(logits_values, b, t - n, classes))
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"bundle={args.bundle_output} sha256={bundle_hash}")
    print(f"logits={args.logits_output} sha256={logits_hash} shape={[b, t - n, classes]}")
    print(f"logits_values={logits_values}")


if __name__ == "__main__":
    main()
