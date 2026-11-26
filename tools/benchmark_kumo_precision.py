"""Compare warm Kumo precision levels with identical weights, data, and ensemble members."""

import argparse
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np

from ironicl import KumoTabularClassifier, KumoTabularRegressor


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
    args = parser.parse_args()
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

    print("level  fit_ms  predict_1_ms  predict_batch_ms  max_error  label_flips")
    for level in ("safe", "auto", "fast"):
        estimator = KumoTabularRegressor if regression else KumoTabularClassifier
        model = estimator(
            size=args.size, n_estimators=args.members, optimize=level,
            model_path=args.model_path, allow_auto_download=False,
        )
        predict = partial(model.predict, output_type="raw_quantiles") if regression else model.predict_proba
        times = []
        for _ in range(4):  # Discard checkpoint loading, upload, and graph compilation.
            t0 = perf_counter()
            model.fit(train, y)
            t1 = perf_counter()
            if model.fitted_.device != args.device or model.fitted_.cpu_members:
                raise RuntimeError(f"not all members ran on {args.device}")
            one = predict(query[:1])
            t2 = perf_counter()
            batch = predict(query)
            t3 = perf_counter()
            times.append((t1 - t0, t2 - t1, t3 - t2))
        outputs = np.vstack((one, batch))
        if not np.isfinite(outputs).all():
            raise ValueError(f"non-finite {level} output")
        if level == "safe":
            reference = outputs
        error = np.max(np.abs(outputs - reference)) / (y.std() if regression else 1)
        flips = "—" if regression else np.count_nonzero(batch.argmax(axis=1) != reference[1:].argmax(axis=1))
        timings = "  ".join(f"{1000 * median(samples):.2f}" for samples in zip(*times[1:]))
        print(f"{level:4}  {timings}  {error:.3g}  {flips}", flush=True)
        del predict, model  # Don't keep the previous mode's fitted ensemble resident.


if __name__ == "__main__":
    main()
