"""Trace Ironicl's memory by stage: model load, fit, and predicts of 1, 64, and 1000 rows.

Prints each stage's current and lifetime-peak physical footprint, which on Apple Silicon
includes GPU buffers. Runs on macOS (the footprint comes from proc_pid_rusage). Set
IRONICL_NO_METAL=1 to trace the CPU path, and IRONICL_MEMORY_LIMIT (such as 8G) to plan for less memory.

    python tools/trace_memory.py --checkpoint PATH --rows 10000 --features 100 --optimize auto
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time

from sklearn.datasets import make_classification


def footprint() -> tuple[float, float]:
    """(current, lifetime peak) physical footprint in GiB, from rusage_info_v4."""
    info = (ctypes.c_uint64 * 64)()
    ctypes.CDLL("libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    # Skip the 16-byte UUID; fields 7 and 28 are ri_phys_footprint and
    # ri_lifetime_max_phys_footprint.
    return info[2 + 7] / 2**30, info[2 + 28] / 2**30


def stage(name: str, fn=lambda: None):
    start = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - start
    current, peak = footprint()
    print(f"  {name:26s} current {current:6.2f} GiB   peak {peak:6.2f} GiB   ({seconds:5.2f} s)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Ironicl's memory by stage.")
    parser.add_argument("--checkpoint", required=True, help="TabICLv2 classifier checkpoint")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--optimize", default="auto", choices=("safe", "auto", "fast"))
    args = parser.parse_args()

    n = args.rows
    x, y = make_classification(
        n_samples=n + 1000, n_features=args.features, n_informative=10, n_classes=3, random_state=0
    )
    metal = "off" if os.environ.get("IRONICL_NO_METAL") else "on"
    print(f"rows={n} features={args.features} optimize={args.optimize} metal={metal}")
    from ironicl import TabICLClassifier

    model = TabICLClassifier(model_path=args.checkpoint, optimize=args.optimize)
    stage("import + estimator")
    stage("fit", lambda: model.fit(x[:n], y[:n]))
    plan = model.memory_plan_
    print(f"  plan: {plan.builder} build, {plan.build_group} estimator(s) at a time, peak {plan.peak_bytes / 2**30:.2f} GiB")
    for rows in (1, 64, 1000):
        stage(f"predict {rows} rows", lambda: model.predict_proba(x[n : n + rows]))


if __name__ == "__main__":
    main()
