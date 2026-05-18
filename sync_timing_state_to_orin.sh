#!/bin/bash
set -euo pipefail

REMOTE="${1:-rubis@192.168.0.11}"
REMOTE_DIR="${2:-/home/rubis/workspace/tensorrt/TrtDnnSplitting}"
MODE="${3:-all}"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
RSYNC_BIN="${RSYNC_BIN:-/usr/bin/rsync}"
SSH_CLEAN_ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD "${SSH_BIN}")

echo "[sync] timing/cache state -> ${REMOTE}:${REMOTE_DIR} (mode=${MODE})"

"${SSH_CLEAN_ENV[@]}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

if [[ "${MODE}" == "timing-json-only" ]]; then
  "${SSH_CLEAN_ENV[@]}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}/artifacts/chunk_cache'"
  "${RSYNC_BIN}" -av --prune-empty-dirs \
    -e "env -u LD_LIBRARY_PATH -u LD_PRELOAD ${SSH_BIN}" \
    --include='*/' \
    --include='timing.json' \
    --exclude='*' \
    artifacts/chunk_cache/ \
    "${REMOTE}:${REMOTE_DIR}/artifacts/chunk_cache/"
  "${SSH_CLEAN_ENV[@]}" "${REMOTE}" \
    "echo '[sync] remote hostname:' \$(hostname); \
     echo '[sync] remote ips:' \$(hostname -I 2>/dev/null || true); \
     echo '[sync] remote repo:' \$(readlink -f '${REMOTE_DIR}'); \
     cd '${REMOTE_DIR}' && \
     ls -ld artifacts artifacts/chunk_cache 2>/dev/null || true; \
     printf '[sync] remote timing.json count: ' && find artifacts/chunk_cache -name timing.json | wc -l; \
     find artifacts/chunk_cache -name timing.json | sed -n '1,3p'"
  echo "[sync] done"
  exit 0
fi

"${RSYNC_BIN}" -av --prune-empty-dirs \
  -e "env -u LD_LIBRARY_PATH -u LD_PRELOAD ${SSH_BIN}" \
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
