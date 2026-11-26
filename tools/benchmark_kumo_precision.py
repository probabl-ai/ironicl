"""Time warm Kumo fit and prediction. Compare Ironicl modes or time NVIDIA's eager FP16 model."""

import argparse
from contextlib import nullcontext
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np


def measure(fit, predict, one, batch, context=nullcontext):
    times = []
    for _ in range(6):  # Discard checkpoint loading, upload, and graph compilation.
        with context():
            t0 = perf_counter()
            fit()
            t1 = perf_counter()
            first = predict(one)
            t2 = perf_counter()
            rest = predict(batch)
            t3 = perf_counter()
        times.append((t1 - t0, t2 - t1, t3 - t2))
    outputs = np.vstack((first, rest))
    if not np.isfinite(outputs).all():
        raise ValueError("non-finite prediction")
    return "  ".join(f"{1000 * median(samples):.2f}" for samples in zip(*times[1:])), outputs


def nvidia_fp16(args, train, query, y):
    import pandas as pd
    import sdm
    import torch

    regression = args.task == "regressor"
    columns = [f"c{i}" for i in range(args.features)]

    def features(x):
        return sdm.TableTensor.from_pandas(pd.DataFrame(x, columns=columns), stypes=dict.fromkeys(columns, "numerical"))

    x, queries = features(train), features(query)
    targets = sdm.TableTensor.from_pandas(
        pd.DataFrame({"y": y.astype(np.float32) if regression else y.astype(str)}),
        stypes={"y": "numerical" if regression else "categorical"},
    )
    model = sdm.models.KumoTabular(task="regression" if regression else "classification", size=args.size, device="cuda")
    model.eval()

    def fit():
        model.fit(
            x.to("cuda"), targets.to("cuda"), num_estimators=args.members, estimator_batch_size=None,
            generator=torch.Generator("cuda").manual_seed(42),
        )
        torch.cuda.synchronize()

    def predict(rows):
        out = model.predict(rows.to("cuda"))
        values = out.numerical.float().cpu().numpy()
        return values if regression else values[:, np.argsort([int(c) for c in out.numerical_columns])]

    with torch.no_grad():
        timings, _ = measure(fit, predict, queries[:1], queries, partial(torch.autocast, "cuda", dtype=torch.float16))
    print("level  fit_ms  predict_1_ms  predict_batch_ms")
    print(f"fp16   {timings}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("classifier", "regressor"))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "metal"))
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--model-path", help="use this checkpoint instead of the default cache")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--nvidia-fp16", action="store_true", help="time NVIDIA's eager CUDA FP16 model instead")
    args = parser.parse_args()
    if args.nvidia_fp16 and (args.device != "cuda" or args.model_path is not None):
        parser.error("--nvidia-fp16 requires --device cuda and NVIDIA's released checkpoint")
    if min(args.rows, args.queries, args.members) < 1 or min(args.rows, args.features) < 2:
        parser.error("need at least two training rows and features, one query, and one member")

    rng = np.random.default_rng(971)
    x = rng.normal(size=(args.rows + args.queries, args.features)).astype(np.float32)
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.3, len(x))
    x.ravel()[::71] = np.nan
    train, query, y = x[:args.rows], x[args.rows:], y[:args.rows]
    regression = args.task == "regressor"
    if not regression:
        y = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))

    if args.nvidia_fp16:
        nvidia_fp16(args, train, query, y)
        return

    from ironicl import KumoTabularClassifier, KumoTabularRegressor

    def fit(estimator):
        estimator.fit(train, y)
        if estimator.fitted_.device != args.device or estimator.fitted_.cpu_members:
            raise RuntimeError(f"not all members ran on {args.device}")

    print("level  fit_ms  predict_1_ms  predict_batch_ms  max_error  label_flips")
    for level in ("safe", "auto", "fast"):
        estimator = KumoTabularRegressor if regression else KumoTabularClassifier
        model = estimator(
            size=args.size, n_estimators=args.members, optimize=level,
            model_path=args.model_path, allow_auto_download=False,
        )
        predict = partial(model.predict, output_type="raw_quantiles") if regression else model.predict_proba
        timings, outputs = measure(partial(fit, model), predict, query[:1], query)
        if level == "safe":
            reference = outputs
        error = np.max(np.abs(outputs - reference)) / (y.std() if regression else 1)
        flips = "—" if regression else np.count_nonzero(outputs[1:].argmax(axis=1) != reference[1:].argmax(axis=1))
        print(f"{level:4}  {timings}  {error:.3g}  {flips}", flush=True)
        del predict, model  # Don't keep the previous mode's fitted ensemble resident.


if __name__ == "__main__":
    main()
