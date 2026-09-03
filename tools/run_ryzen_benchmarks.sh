#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/semyon/Projects/ironicl-bench}"
PYTHON="$ROOT/.venv/bin/python"
HARNESS="$ROOT/src/Ironicl/tools/benchmark_tabicl.py"
RESULTS="$ROOT/results"
COMMON=(
  "$HARNESS"
  --checkpoint "$ROOT/fixtures/tabicl-classifier-v2-20260212.ckpt"
  --model "$ROOT/fixtures/tabicl-classifier-v2.ironicl"
  --input "$ROOT/fixtures/input.irinput"
  --tabicl-root "$ROOT/src/tabicl/src"
  --ironicl "$ROOT/bin/ironicl-znver5"
  --iterations 100
)

mkdir -p "$RESULTS"

taskset -c 0 "$PYTHON" "${COMMON[@]}" \
  --device cpu --threads 1 --ironicl-threads 1 --warmup 20 \
  | tee "$RESULTS/cpu-1-thread.txt"

taskset -c 0-7 "$PYTHON" "${COMMON[@]}" \
  --device cpu --threads 8 --ironicl-threads 8 --warmup 20 \
  | tee "$RESULTS/cpu-8-threads.txt"

taskset -c 0-7 "$PYTHON" "${COMMON[@]}" \
  --device cuda --threads 1 --ironicl-threads 8 --warmup 50 \
  | tee "$RESULTS/cuda-fp32.txt"

taskset -c 0-7 "$PYTHON" "${COMMON[@]}" \
  --device cuda --amp --threads 1 --ironicl-threads 8 --warmup 50 \
  | tee "$RESULTS/cuda-amp.txt"
