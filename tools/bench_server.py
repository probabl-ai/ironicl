#!/usr/bin/env python3
"""Measure the throughput and latency of `ironicl serve` with oha
(https://github.com/hatoo/oha).

For each optimize level, it rebuilds the exported model's KV cache with
`ironicl prepare` (overwriting MODEL/context.ircntxt), then for each
--max-batch-rows starts the server and runs oha for each request size at each
client count, or with --rates, at each request rate. Request rows are the
model's training rows, repeated as needed. Latencies are of successful
requests. It prints a Markdown table and appends each result to --output as a
JSON line.

    tools/bench_server.py MODEL --levels auto fast --batch-rows 256 1024 \\
        --request-rows 1 64 --clients 1 32 256 --duration 10s
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="A directory that export() wrote.")
    parser.add_argument("--ironicl", type=Path, default=Path("target/release/ironicl"))
    parser.add_argument("--levels", nargs="+", default=["safe"], choices=["safe", "auto", "fast"])
    parser.add_argument("--cpu", action="store_true", help="Serve on the CPU (IRONICL_NO_CUDA, IRONICL_NO_METAL).")
    parser.add_argument("--batch-rows", nargs="+", type=int, default=[1, 128], help="--max-batch-rows values.")
    parser.add_argument("--max-queue", type=int, help="The server's --max-queue, in ms.")
    parser.add_argument("--request-rows", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--rates", nargs="+", type=int, help="Requests per second, open loop, instead of --clients.")
    parser.add_argument("--rate-clients", type=int, default=256, help="Connections for --rates.")
    parser.add_argument("--duration", default="10s")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--output", type=Path, help="Append results here as JSON lines.")
    return parser.parse_args()


def wait_until_ready(url: str, server: subprocess.Popen) -> None:
    while server.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"the server exited with code {server.returncode}")


def oha(url: str, body: Path, duration: str, clients: int, rate: int | None) -> dict:
    command = ["oha", "--no-tui", "--output-format", "json", "-z", duration, "-w", "--stats-success-breakdown", "-c", str(clients)]
    if rate is not None:
        command += ["-q", str(rate), "--latency-correction"]
    command += ["-m", "POST", "-T", "application/json", "-D", str(body), url]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def main() -> None:
    args = parse_args()
    env = dict(os.environ, IRONICL_NO_CUDA="1", IRONICL_NO_METAL="1") if args.cpu else os.environ
    rows = np.load(args.model / "train_x.npy")
    bodies = Path(tempfile.mkdtemp())
    for count in args.request_rows:
        (bodies / f"{count}.json").write_text(json.dumps({"rows": np.resize(rows, (count, rows.shape[1])).tolist()}))
    largest = max(path.stat().st_size for path in bodies.iterdir())
    url = f"http://127.0.0.1:{args.port}"
    loads = [("rate", rate) for rate in args.rates] if args.rates else [("clients", c) for c in args.clients]
    print("| level | batch rows | request rows | load | req/s | rows/s | p50 ms | p90 ms | p99 ms | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for level in args.levels:
        prepare = [str(args.ironicl), "prepare", str(args.model), "--optimize", level]
        subprocess.run(prepare, check=True, env=env, capture_output=True)
        for batch_rows in args.batch_rows:
            command = [str(args.ironicl), "serve", str(args.model), "--bind", f"127.0.0.1:{args.port}"]
            command += ["--max-batch-rows", str(batch_rows)]
            command += ["--max-request-rows", str(max(args.request_rows)), "--max-body", str(largest)]
            if args.max_queue:
                command += ["--max-queue", str(args.max_queue)]
            server = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL)
            try:
                wait_until_ready(f"{url}/v1/health", server)
                for count in args.request_rows:
                    for kind, load in loads:
                        rate = load if kind == "rate" else None
                        clients = args.rate_clients if rate else load
                        result = oha(f"{url}/v1/predict", bodies / f"{count}.json", args.duration, clients, rate)
                        codes = result.get("statusCodeDistribution", {})
                        errors = sum(n for code, n in codes.items() if code != "200")
                        per_second = result["summary"]["requestsPerSec"] * codes.get("200", 0) / max(sum(codes.values()), 1)
                        percentiles = {key: (result["latencyPercentilesSuccessful"][key] or 0) * 1e3 for key in ("p50", "p90", "p99")}
                        record = {
                            "level": level, "cpu": args.cpu, "batch_rows": batch_rows,
                            "request_rows": count,
                            kind: load, "requests_per_second": per_second, "errors": errors, **percentiles,
                        }
                        load_text = f"{load}/s" if rate else f"{load} clients"
                        print(
                            f"| {level}{' cpu' if args.cpu else ''} | {batch_rows} | {count} | {load_text} "
                            f"| {per_second:,.0f} | {per_second * count:,.0f} | {percentiles['p50']:.2f} "
                            f"| {percentiles['p90']:.2f} | {percentiles['p99']:.2f} | {errors} |",
                            flush=True,
                        )
                        if args.output:
                            with open(args.output, "a") as file:
                                file.write(json.dumps(record) + "\n")
            finally:
                server.terminate()
                server.wait()


if __name__ == "__main__":
    main()
