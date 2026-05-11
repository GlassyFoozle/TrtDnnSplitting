#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs/yaml/260511_configs"
RESULT_LOG_DIR="results"
PLOT_DIR="results/plots"

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
  # "9_G_ratio_09.yaml"
  # "10_base2.yaml"
)

for config_name in "${CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${config_name}"
  run_suffix="${config_name%.yaml}"
  run_name="fig4_3models_${run_suffix}"
  log_path="${RESULT_LOG_DIR}/${run_name}.log"

  echo "============================================================"
  echo "[run] ${config_path}"
  echo "[run] output: results/dnn_experiments/${run_name}"
  echo "[run] log: ${log_path}"
  echo "============================================================"

  conda run --no-capture-output -n trt python -u scripts/30_run_yaml_fig4_experiment.py \
    --config "${config_path}" \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 50 \
    --live \
    --max-candidates 1000000 \
    --max-profiles 1000000 \
    --min-free-gb 50 \
    --run-name "${run_name}" \
    2>&1 | tee "${log_path}"

  echo "============================================================"
  echo "[plot] ${run_name}"
  echo "============================================================"

  conda run -n trt python scripts/31_plot_fig4.py \
    --run-dir "results/dnn_experiments/${run_name}" \
    --plot-mode all \
    --output "${run_name}" \
    --output-dir "${PLOT_DIR}"
done

echo "All experiments and plots completed."
