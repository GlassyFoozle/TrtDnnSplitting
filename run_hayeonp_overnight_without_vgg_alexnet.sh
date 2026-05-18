#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CONFIG_DIR:-configs/yaml/gpu_util_configs}"
POSTFIX="${POSTFIX:-overnight_without_vgg_alexnet}"
CONFIGS_OVERRIDE="${CONFIGS_OVERRIDE:-}"
MONOTONIC_K_SPLIT_CACHE="${MONOTONIC_K_SPLIT_CACHE:-0}"

TABLE4_RUNNER="cpp_runtime/build_sync_wall/table4_runner"
if ! getcap "$TABLE4_RUNNER" 2>/dev/null | grep -q 'cap_sys_nice'; then
  echo "Granting cap_sys_nice to $TABLE4_RUNNER (needed for SCHED_FIFO wall-time profiling)..."
  sudo setcap cap_sys_nice+ep "$TABLE4_RUNNER"
fi

CONFIGS=("1_base.yaml")
if [[ -n "${CONFIGS_OVERRIDE}" ]]; then
  read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE}"
fi

suite_restart_count=0
while true; do
  suite_changed=0

  for config_name in "${CONFIGS[@]}"; do
    config_path="${CONFIG_DIR}/${config_name}"
    run_suffix="${config_name%.yaml}"
    run_name="${run_suffix}_${POSTFIX}"

    echo "============================================================"
    echo "[run] ${config_path}"
    echo "[run] output: results/dnn_experiments/${run_name}"
    echo "[run] monotonic K-split cache: ${MONOTONIC_K_SPLIT_CACHE}"
    echo "============================================================"

    monotonic_k_split_cache_args=()
    if [[ "${MONOTONIC_K_SPLIT_CACHE}" == "0" ]]; then
      monotonic_k_split_cache_args+=(--no-monotonic-k-split-cache)
    fi

    inflation_marker="results/dnn_experiments/${run_name}.inflate_ratio_changed.jsonl"
    rm -f "${inflation_marker}"
    MONOTONIC_INFLATION_MARKER="${inflation_marker}" \
    conda run --no-capture-output -n trt python scripts/33_run_yaml_fig4_with_split_plots.py \
      --plot-algorithms SS-heu SS-tol-fb UNI-heu UNI-tol-fb \
      --config "${config_path}" \
      --models resnet18 vit_b_16 inception_v3 mobilenet_v3_small \
      --run-name "${run_name}" \
      --split-policy trt_fusion_safe \
      --utilizations 0.5 0.6 0.7 0.8 0.9 \
      --num-tasksets 50 \
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
      "${monotonic_k_split_cache_args[@]}"

    if [[ ! -s "${inflation_marker}" ]]; then
      continue
    fi
    suite_changed=1
    suite_restart_count=$((suite_restart_count + 1))
    if (( suite_restart_count > 20 )); then
      echo "[error] inflate-ratio kept changing after ${suite_restart_count} suite attempts" >&2
      exit 1
    fi
    echo "[rerun] inflate-ratio changed interval timings during ${run_name}"
    changed_models="$(
      python3 - "${inflation_marker}" <<'PY'
import json, sys
models = sorted({
    json.loads(line)["model"]
    for line in open(sys.argv[1])
    if line.strip()
})
print(" ".join(models))
PY
    )"
    if [[ -n "${changed_models}" ]]; then
      echo "[rerun] rebuilding inflated best-K cache for: ${changed_models}"
      conda run --no-capture-output -n trt python scripts/44_rebuild_best_k_cache.py \
        --models ${changed_models} \
        --precision fp32 \
        --policy trt_fusion_safe
    fi
    echo "[rerun] invalidating all YAML results with postfix ${POSTFIX} and restarting the suite"
    for stale_config_name in "${CONFIGS[@]}"; do
      stale_suffix="${stale_config_name%.yaml}"
      rm -rf "results/dnn_experiments/${stale_suffix}_${POSTFIX}"
    done
    break
  done

  if (( suite_changed == 0 )); then
    break
  fi
done
