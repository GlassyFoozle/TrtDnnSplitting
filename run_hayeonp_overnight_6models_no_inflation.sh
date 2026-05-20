#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CONFIG_DIR:-configs/yaml/gpu_util_configs}"
POSTFIX="${POSTFIX:-overnight_6models_no_inflation}"
CONFIGS_OVERRIDE="${CONFIGS_OVERRIDE:-}"
MONOTONIC_K_SPLIT_CACHE="${MONOTONIC_K_SPLIT_CACHE:-0}"
TRT_RAW_INTERVAL_TIMING="${TRT_RAW_INTERVAL_TIMING:-1}"
TRT_DISABLE_MONOTONIC_INFLATION="${TRT_DISABLE_MONOTONIC_INFLATION:-1}"
K_SPLIT_CACHE_PATH="${K_SPLIT_CACHE_PATH:-results/optimization/measured_k_split_cache_no_inflation.json}"
EVALUATION_CACHE_DIR="${EVALUATION_CACHE_DIR:-results/evaluations_no_inflation}"
PROFILING_CACHE_PATH="${PROFILING_CACHE_PATH:-results/optimization/.profiling_cache_no_inflation.json}"

TABLE4_RUNNER="cpp_runtime/build_sync_wall/table4_runner"
if ! getcap "$TABLE4_RUNNER" 2>/dev/null | grep -q 'cap_sys_nice'; then
  echo "Granting cap_sys_nice to $TABLE4_RUNNER (needed for SCHED_FIFO wall-time profiling)..."
  sudo setcap cap_sys_nice+ep "$TABLE4_RUNNER"
fi

CONFIGS=(
  "1_base.yaml"
  "2_C_ratio_00.yaml"
  "3_C_ratio_25.yaml"
  "4_C_ratio_50.yaml"
  "5_task1.yaml"
  "6_task3.yaml"
  "7_singleCPU_task4.yaml"
  "8_singleCPU_task8.yaml"
)
if [[ -n "${CONFIGS_OVERRIDE}" ]]; then
  read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE}"
fi

for config_name in "${CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${config_name}"
  run_suffix="${config_name%.yaml}"
  run_name="${run_suffix}_${POSTFIX}"

  echo "============================================================"
  echo "[run] ${config_path}"
  echo "[run] output: results/dnn_experiments/${run_name}"
  echo "[run] monotonic K-split cache: ${MONOTONIC_K_SPLIT_CACHE}"
  echo "[run] raw interval timing: ${TRT_RAW_INTERVAL_TIMING}"
  echo "[run] disable monotonic inflation: ${TRT_DISABLE_MONOTONIC_INFLATION}"
  echo "[run] K-split cache: ${K_SPLIT_CACHE_PATH}"
  echo "[run] evaluation cache: ${EVALUATION_CACHE_DIR}"
  echo "[run] profiling cache: ${PROFILING_CACHE_PATH}"
  echo "============================================================"

  monotonic_k_split_cache_args=()
  if [[ "${MONOTONIC_K_SPLIT_CACHE}" == "0" ]]; then
    monotonic_k_split_cache_args+=(--no-monotonic-k-split-cache)
  fi

  TRT_RAW_INTERVAL_TIMING="${TRT_RAW_INTERVAL_TIMING}" \
  TRT_DISABLE_MONOTONIC_INFLATION="${TRT_DISABLE_MONOTONIC_INFLATION}" \
  K_SPLIT_CACHE_PATH="${K_SPLIT_CACHE_PATH}" \
  EVALUATION_CACHE_DIR="${EVALUATION_CACHE_DIR}" \
  PROFILING_CACHE_PATH="${PROFILING_CACHE_PATH}" \
  conda run --no-capture-output -n trt python scripts/33_run_yaml_fig4_with_split_plots.py \
    --plot-algorithms SS-opt SS-heu SS-tol-fb UNI-opt UNI-heu UNI-tol-fb \
    --config "${config_path}" \
    --models alexnet resnet18 vit_b_16 vgg19 inception_v3 mobilenet_v3_small \
    --run-name "${run_name}" \
    --split-policy trt_fusion_safe \
    --utilizations 0.5 0.6 0.7 0.8 0.9 \
    --num-tasksets 50 \
    --precision fp32 \
    --algorithms \
      ss:opt:SS-opt \
      ss:heu:SS-heu \
      ss:tol-fb:SS-tol-fb \
      uni:opt:UNI-opt \
      uni:heu:UNI-heu \
      uni:tol-fb:UNI-tol-fb \
    --builder-optimization-level 1 \
    --live \
    --wcet-metric max \
    --max-candidates 1000000 \
    --max-profiles 1000000 \
    "${monotonic_k_split_cache_args[@]}"
done
