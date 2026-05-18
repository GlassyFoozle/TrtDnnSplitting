#!/bin/bash
set -euo pipefail

TABLE4_RUNNER="cpp_runtime/build_sync_wall/table4_runner"
if ! getcap "$TABLE4_RUNNER" 2>/dev/null | grep -q 'cap_sys_nice'; then
  echo "Granting cap_sys_nice to $TABLE4_RUNNER (needed for SCHED_FIFO wall-time profiling)..."
  sudo setcap cap_sys_nice+ep "$TABLE4_RUNNER"
fi

MONOTONIC_REPAIR_MODE="${MONOTONIC_REPAIR_MODE:-inflate-ratio}"
case "$MONOTONIC_REPAIR_MODE" in
  reprofile)
    monotonic_args=(--reprofile-monotonic-violations --max-monotonic-reprofile-rounds 100)
    ;;
  inflate-ratio)
    monotonic_args=(--inflate-monotonic-children-by-parent-ratio)
    ;;
  none)
    monotonic_args=()
    ;;
  *)
    echo "Unknown MONOTONIC_REPAIR_MODE: $MONOTONIC_REPAIR_MODE" >&2
    echo "Expected one of: reprofile, inflate-ratio, none" >&2
    exit 2
    ;;
esac

conda run --no-capture-output -n trt python scripts/43_remeasure_all_caches_walltime.py \
  --precisions fp16 \
  --force-intervals \
  --force-eval-caches \
  --skip-profiling-db \
  "${monotonic_args[@]}" \
  "$@"
  

# 실패한것만 다시하기
# python3 scripts/43_remeasure_all_caches_walltime.py \
#   --precisions fp16 \
#   --reprofile-violation-list results/monotonic_interval_violations.json \
#   --reprofile-monotonic-violations \
#   --skip-profiling-db

# resnet18 interval cache만 다시 실측하고, inflate-ratio 보정까지 적용한 뒤,
# 그 보정된 interval 값 기준으로 resnet18 best-K cache만 다시 생성:

# MONOTONIC_REPAIR_MODE=inflate-ratio ./reprofile.sh \
#   --models resnet18 \
#   --skip-eval-caches

# conda run --no-capture-output -n trt python scripts/44_rebuild_best_k_cache.py \
#   --models resnet18 \
#   --precision fp16 \
#   --policy trt_fusion_safe
