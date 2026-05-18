#!/usr/bin/env bash
set -euo pipefail

TABLE4_RUNNER="cpp_runtime/build_sync_wall/table4_runner"
if ! getcap "$TABLE4_RUNNER" 2>/dev/null | grep -q 'cap_sys_nice'; then
  echo "Granting cap_sys_nice to $TABLE4_RUNNER (needed for SCHED_FIFO wall-time profiling)..."
  sudo setcap cap_sys_nice+ep "$TABLE4_RUNNER"
fi

conda run --no-capture-output -n trt python -u scripts/16_profile_policy_chunk_table.py \
  --models alexnet resnet18 \
  --policy all \
  --precision fp32 \
  --builder-optimization-level 5 \
  --rebuild-engines \
  --warmup 20 --iters 500

conda run --no-capture-output -n trt python -u scripts/15_compare_k1_timing_semantics.py \
  --profile-k1 \
  --models alexnet resnet18 \
  --precision fp32 \
  --builder-optimization-level 5 \
  --rebuild-engine \
  --warmup 20 --iters 500
