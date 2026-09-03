#!/usr/bin/env bash
set -u -o pipefail

ROOT="${1:-/home/semyon/Projects/ironicl-bench}"
PYTHON="$ROOT/.venv/bin/python"
HARNESS="$ROOT/src/Ironicl/tools/benchmark_tabicl_sklearn_cache.py"
RESULTS="$ROOT/results/tabicl-large-cache"
ROWS="${IRONICL_BENCH_ROWS:-4096 16384 32768 50000 65536 100000}"
WARMUP="${IRONICL_BENCH_WARMUP:-1}"
ITERATIONS="${IRONICL_BENCH_ITERATIONS:-3}"

mkdir -p "$RESULTS"

run_point() {
  local rows="$1"
  local device="$2"
  local precision="$3"
  local output="$RESULTS/${device}-${precision}-kv-n${rows}-q128-b8.json"
  local failure="$output.failed.txt"
  local args=(
    "$HARNESS"
    --checkpoint "$ROOT/fixtures/tabicl-classifier-v2-20260212.ckpt"
    --tabicl-root "$ROOT/src/tabicl/src"
    --dataset covertype
    --data-home "$ROOT/data/sklearn"
    --context-rows "$rows"
    --query-rows 128
    --estimators 8
    --batch-size 1
    --threads 1
    --warmup "$WARMUP"
    --iterations "$ITERATIONS"
    --seed 42
    --device "$device"
    --offload-mode gpu
    --kv-cache kv
    --output "$output"
  )
  if [[ "$precision" == "amp" ]]; then
    args+=(--use-amp)
  fi
  rm -f "$failure"
  if ! "$PYTHON" "${args[@]}" 2>"$failure"; then
    printf 'failed: device=%s precision=%s rows=%s\n' "$device" "$precision" >>"$failure"
    return 0
  fi
  rm -f "$failure"
}

for rows in $ROWS; do
  run_point "$rows" cuda amp
  run_point "$rows" cuda fp32
done
