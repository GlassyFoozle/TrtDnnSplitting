#!/bin/bash
set -euo pipefail

REMOTE="${1:-rubis@192.168.0.12}"
REMOTE_DIR="${2:-/home/rubis/workspace/TrtDnnSplitting}"

echo "[sync] timing/cache state -> ${REMOTE}:${REMOTE_DIR}"

ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

rsync -av --prune-empty-dirs \
  --include='/artifacts/' \
  --include='/artifacts/chunk_cache/' \
  --include='/artifacts/chunk_cache/*/' \
  --include='/artifacts/chunk_cache/*/int_*/' \
  --include='/artifacts/chunk_cache/*/int_*/timing.json' \
  --include='/artifacts/split_configs/' \
  --include='/artifacts/split_configs/*/' \
  --include='/artifacts/split_configs/*/dag_aligned_full.json' \
  --include='/results/' \
  --include='/results/evaluations/' \
  --include='/results/evaluations/*/' \
  --include='/results/evaluations/*/*_fp16.json' \
  --include='/results/evaluations/*/*_fp32.json' \
  --exclude='/results/evaluations/*/*_cpp_raw.json' \
  --include='/results/optimization/' \
  --include='/results/optimization/.profiling_cache.json' \
  --include='/results/optimization/measured_k_split_cache.json' \
  --include='/results/best_k_split_curves/' \
  --include='/results/best_k_split_curves/***' \
  --include='/fp16_endpoint_chunk_times.csv' \
  --exclude='*' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "[sync] done"
