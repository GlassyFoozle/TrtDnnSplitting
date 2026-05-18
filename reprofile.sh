#!/bin/bash
set -euo pipefail

TABLE4_RUNNER="cpp_runtime/build_sync_wall/table4_runner"
# Reprofile jobs are bulk cache maintenance, so keep them off the pinned/RT
# profiling path used by experiments.
export TRT_DISABLE_RT_PROFILE=1



  # alexnet
  # resnet18
  # vgg19
  # vit_b_16
  # inception_v3
  # mobilenet_v3_small

# Edit these when switching the default reprofile job.
MONOTONIC_REPAIR_MODE="inflate-ratio"
PRECISIONS="fp32"
MODELS=(
  vgg19
)
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

# conda run --no-capture-output -n trt python scripts/43_remeasure_all_caches_walltime.py \
#   --models "${MODELS[@]}" \
#   --precisions ${PRECISIONS} \
#   --force-intervals \
#   --force-eval-caches \
#   --skip-profiling-db \
#   "${monotonic_args[@]}" \
#   "$@"
  
# Update cache
for model in "${MODELS[@]}"; do
  conda run --no-capture-output -n trt python scripts/44_rebuild_best_k_cache.py \
    --models "$model" \
    --precision ${PRECISIONS} \
    --policy trt_fusion_safe
done
