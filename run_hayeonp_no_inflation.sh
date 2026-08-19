#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-trt}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
export TRTEXEC
TABLE4_BUILD_DIR="cpp_runtime/build_sync_wall"
TABLE4_RUNNER="${TABLE4_BUILD_DIR}/table4_runner"
AUTO_BUILD_RUNNER="${AUTO_BUILD_RUNNER:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

CONFIG_DIR="${CONFIG_DIR:-configs/yaml/gpu_util_configs}"
RUN_LABEL="${RUN_LABEL:-fig4_6models_fp32_opt1_no_inflation}"
CONFIGS_OVERRIDE="${CONFIGS_OVERRIDE:-}"
MODELS_OVERRIDE="${MODELS_OVERRIDE:-}"
UTILIZATIONS_OVERRIDE="${UTILIZATIONS_OVERRIDE:-}"
NUM_TASKSETS="${NUM_TASKSETS:-50}"
MONOTONIC_K_SPLIT_CACHE="${MONOTONIC_K_SPLIT_CACHE:-0}"
TRT_RAW_INTERVAL_TIMING="${TRT_RAW_INTERVAL_TIMING:-1}"
TRT_DISABLE_MONOTONIC_INFLATION="${TRT_DISABLE_MONOTONIC_INFLATION:-1}"
K_SPLIT_CACHE_PATH="${K_SPLIT_CACHE_PATH:-results/optimization/measured_k_split_cache_no_inflation.json}"
EVALUATION_CACHE_DIR="${EVALUATION_CACHE_DIR:-results/evaluations_no_inflation}"
PROFILING_CACHE_PATH="${PROFILING_CACHE_PATH:-results/optimization/.profiling_cache_no_inflation.json}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda is not available in PATH." >&2
  exit 1
fi

for required_command in cmake getcap setcap; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "[error] Missing required command: ${required_command}" >&2
    echo "        Install build-essential, cmake, and libcap2-bin as described in INSTALL.md." >&2
    exit 1
  fi
done

if ! conda run -n "${CONDA_ENV}" python -c \
  'import matplotlib, numpy, onnx, torch, torchvision' >/dev/null 2>&1; then
  echo "[error] Conda environment '${CONDA_ENV}' is missing or incomplete." >&2
  echo "        Follow INSTALL.md, then retry." >&2
  exit 1
fi

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "[error] TensorRT trtexec was not found or is not executable: ${TRTEXEC}" >&2
  echo "        Set TRTEXEC=/path/to/trtexec when using a non-default installation." >&2
  exit 1
fi

if [[ ! -x "${TABLE4_RUNNER}" ]]; then
  if [[ "${AUTO_BUILD_RUNNER}" != "1" ]]; then
    echo "[error] Missing profiler: ${TABLE4_RUNNER}" >&2
    echo "        Build it as documented in INSTALL.md or set AUTO_BUILD_RUNNER=1." >&2
    exit 1
  fi
  echo "[setup] Building table4_runner in ${TABLE4_BUILD_DIR}..."
  cmake -S cpp_runtime -B "${TABLE4_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${TABLE4_BUILD_DIR}" --target table4_runner --parallel "$(nproc)"
fi

if ! getcap "$TABLE4_RUNNER" 2>/dev/null | grep -q 'cap_sys_nice'; then
  echo "[setup] Granting cap_sys_nice to ${TABLE4_RUNNER} for SCHED_FIFO profiling..."
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

MODELS=(alexnet resnet18 vit_b_16 vgg19 inception_v3 mobilenet_v3_small)
if [[ -n "${MODELS_OVERRIDE}" ]]; then
  read -r -a MODELS <<< "${MODELS_OVERRIDE}"
fi

UTILIZATIONS=(0.5 0.6 0.7 0.8 0.9)
if [[ -n "${UTILIZATIONS_OVERRIDE}" ]]; then
  read -r -a UTILIZATIONS <<< "${UTILIZATIONS_OVERRIDE}"
fi

mkdir -p \
  artifacts \
  "$(dirname -- "${K_SPLIT_CACHE_PATH}")" \
  "${EVALUATION_CACHE_DIR}" \
  "$(dirname -- "${PROFILING_CACHE_PATH}")"

for config_name in "${CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${config_name}"
  if [[ ! -f "${config_path}" ]]; then
    echo "[error] Missing experiment config: ${config_path}" >&2
    exit 1
  fi
  run_suffix="${config_name%.yaml}"
  run_name="${RUN_LABEL}_${run_suffix}"

  echo "============================================================"
  echo "[run] ${config_path}"
  echo "[run] output: results/dnn_experiments/${run_name}"
  echo "[run] monotonic K-split cache: ${MONOTONIC_K_SPLIT_CACHE}"
  echo "[run] raw interval timing: ${TRT_RAW_INTERVAL_TIMING}"
  echo "[run] disable monotonic inflation: ${TRT_DISABLE_MONOTONIC_INFLATION}"
  echo "[run] K-split cache: ${K_SPLIT_CACHE_PATH}"
  echo "[run] evaluation cache: ${EVALUATION_CACHE_DIR}"
  echo "[run] profiling cache: ${PROFILING_CACHE_PATH}"
  echo "[run] models: ${MODELS[*]}"
  echo "[run] utilizations: ${UTILIZATIONS[*]}"
  echo "[run] tasksets per utilization: ${NUM_TASKSETS}"
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
  conda run --no-capture-output -n "${CONDA_ENV}" python scripts/33_run_yaml_fig4_with_split_plots.py \
    --plot-algorithms SS-tol-fb \
    --config "${config_path}" \
    --models "${MODELS[@]}" \
    --run-name "${run_name}" \
    --split-policy trt_fusion_safe \
    --utilizations "${UTILIZATIONS[@]}" \
    --num-tasksets "${NUM_TASKSETS}" \
    --precision fp32 \
    --algorithms \
      ss:heu:SS-heu \
      ss:tol-fb:SS-tol-fb \
      uni:heu:UNI-heu \
      uni:tol-fb:UNI-tol-fb \
    --builder-optimization-level 1 \
    --live \
    --wcet-metric max \
    --max-candidates 1000000 \
    --max-profiles 1000000 \
    --min-free-gb "${MIN_FREE_GB}" \
    "${monotonic_k_split_cache_args[@]}"
done
