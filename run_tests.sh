#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs/yaml/260511_configs"
RESULT_LOG_DIR="results"
PLOT_DIR="results/plots"
MODELS="${MODELS:-alexnet resnet18 vgg19 vit_l_16}"
RUN_LABEL="${RUN_LABEL:-fig4_4models}"
NUM_TASKSETS_OVERRIDE="${NUM_TASKSETS_OVERRIDE:-50}"
UTILIZATIONS="${UTILIZATIONS:-}"
ALGORITHM_SET="${ALGORITHM_SET:-rtss}"
ALGORITHMS="${ALGORITHMS:-}"
CONFIGS_OVERRIDE="${CONFIGS_OVERRIDE:-}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
EXISTING_TASKSET_ROOT="${EXISTING_TASKSET_ROOT:-}"
VERBOSE_EVALUATOR="${VERBOSE_EVALUATOR:-0}"

mkdir -p "${RESULT_LOG_DIR}" "${PLOT_DIR}"

CONFIGS=(
  "1_base.yaml"
  "2_G_ratio_06.yaml"
  "3_G_ratio_08.yaml"
  "4_G_ratio_10.yaml"
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
  run_name="${RUN_LABEL}_${run_suffix}"
  log_path="${RESULT_LOG_DIR}/${run_name}.log"

  # echo "============================================================"
  # echo "[run] ${config_path}"
  # echo "[run] output: results/dnn_experiments/${run_name}"
  # echo "[run] log: ${log_path}"
  # echo "============================================================"

  # cmd=(conda run --no-capture-output -n trt python -u scripts/30_run_yaml_fig4_experiment.py \
  #   --config "${config_path}" \
  #   --models ${MODELS} \
  #   --split-policy major_blocks \
  #   --algorithm-set "${ALGORITHM_SET}" \
  #   --num-tasksets-override "${NUM_TASKSETS_OVERRIDE}" \
  #   --live \
  #   --max-candidates 1000000 \
  #   --max-profiles 1000000 \
  #   --min-free-gb 50 \
  #   --run-name "${run_name}")

  # if [[ -n "${UTILIZATIONS}" ]]; then
  #   cmd+=(--utilizations ${UTILIZATIONS})
  # fi
  # if [[ -n "${ALGORITHMS}" ]]; then
  #   cmd+=(--algorithms ${ALGORITHMS})
  # fi
  # if [[ -n "${EXISTING_TASKSET_ROOT}" ]]; then
  #   cmd+=(--existing-taskset-root "${EXISTING_TASKSET_ROOT}")
  # fi
  # if [[ "${VERBOSE_EVALUATOR}" == "1" ]]; then
  #   cmd+=(--verbose-evaluator)
  # fi

  # "${cmd[@]}" 2>&1 | tee "${log_path}"

  echo "============================================================"
  echo "[plot] ${run_name}"
  echo "============================================================"

  if [[ "${SKIP_PLOTS}" == "1" ]]; then
    echo "[plot] skipped (SKIP_PLOTS=1)"
  else
    conda run -n trt python scripts/31_plot_fig4.py \
      --run-dir "results/dnn_experiments/${run_name}" \
      --plot-mode all \
      --output "${run_name}" \
      --output-dir "${PLOT_DIR}"
  fi
done

echo "All experiments and plots completed."
