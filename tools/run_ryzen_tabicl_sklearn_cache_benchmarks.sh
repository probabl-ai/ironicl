#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/semyon/Projects/ironicl-bench}"
PYTHON="$ROOT/.venv/bin/python"
HARNESS="$ROOT/src/Ironicl/tools/benchmark_tabicl_sklearn_cache.py"
RESULTS="$ROOT/results/tabicl-sklearn-cache"
COMMON=(
  "$HARNESS"
  --checkpoint "$ROOT/fixtures/tabicl-classifier-v2-20260212.ckpt"
  --tabicl-root "$ROOT/src/tabicl/src"
  --dataset covertype
  --data-home "$ROOT/data/sklearn"
  --context-rows 1024
  --query-rows 64
  --estimators 1
  --batch-size 1
  --threads 1
  --warmup 5
  --iterations 20
  --seed 42
)

mkdir -p "$RESULTS"

for cache_mode in kv repr; do
  taskset -c 0 "$PYTHON" "${COMMON[@]}" --device cpu --kv-cache "$cache_mode" \
    --output "$RESULTS/cpu-fp32-${cache_mode}.json"
  taskset -c 0 "$PYTHON" "${COMMON[@]}" --device cuda --kv-cache "$cache_mode" \
    --output "$RESULTS/cuda-fp32-${cache_mode}.json"
done
