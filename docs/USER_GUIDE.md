# User Guide

The supported end-to-end entry point is
`./run_hayeonp_no_inflation.sh`. Run it from any directory; the script changes
to the repository root before resolving paths.

## Default run

```bash
./run_hayeonp_no_inflation.sh
```

The default run covers eight workload configurations, six models, five
utilization points, and 50 task sets per utilization. It uses FP32,
`trt_fusion_safe`, builder optimization level 1, and maximum measured wall-time
WCET.

## Useful overrides

All list-valued overrides are space-separated strings.

```bash
CONFIGS_OVERRIDE="1_base.yaml 5_task1.yaml" \
MODELS_OVERRIDE="alexnet resnet18" \
UTILIZATIONS_OVERRIDE="0.5 0.7" \
NUM_TASKSETS=5 \
RUN_LABEL=my_run \
MIN_FREE_GB=20 \
./run_hayeonp_no_inflation.sh
```

Additional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CONDA_ENV` | `trt` | Conda environment used for every Python process |
| `TRTEXEC` | `/usr/src/tensorrt/bin/trtexec` | TensorRT engine builder |
| `AUTO_BUILD_RUNNER` | `1` | Build `table4_runner` when missing |
| `K_SPLIT_CACHE_PATH` | `results/optimization/measured_k_split_cache_no_inflation.json` | measured best-K cache |
| `EVALUATION_CACHE_DIR` | `results/evaluations_no_inflation` | exact-mask evaluation cache |
| `PROFILING_CACHE_PATH` | `results/optimization/.profiling_cache_no_inflation.json` | base/profile metadata cache |

The no-inflation controls default to
`TRT_RAW_INTERVAL_TIMING=1`, `TRT_DISABLE_MONOTONIC_INFLATION=1`, and
`MONOTONIC_K_SPLIT_CACHE=0`. Do not override them for the canonical run.

## Generated data

```text
artifacts/chunk_cache/<model>/int_<start>_<end>/
  chunk.onnx
  chunk_fp32.engine
  timing.json

artifacts/split_configs/<model>/
  dag_aligned_full.json          versioned baseline metadata
  <generated-mask>.json          generated and ignored

results/evaluations_no_inflation/<model>/
  <mask>_fp32.json
  <mask>_fp32_cpp_raw.json

results/dnn_experiments/<run-name>/
  generated_tasksets/
  per_taskset_results.csv
  schedulability_ratio.csv
  split_activity.csv
  per_taskset_timing.csv
  timing_summary.csv
  all_results.json
  summary.md
  *.png / *.pdf
```

All generated paths are ignored by Git. Reusing the same cache paths avoids
rebuilding shared interval engines. Use a new `RUN_LABEL` when results from two
runs must coexist.

## Tests and CLI validation

```bash
conda run --no-capture-output -n trt python -m pytest -q

conda run --no-capture-output -n trt python \
  scripts/30_run_yaml_fig4_experiment.py --help
```

An analytical `--dry-run` still needs per-chunk timing metadata. A fresh clone
does not ship device measurements, so use the live smoke run to create them on
the target Orin. Approximate fallback values used during task generation are
not a substitute for mask-level TensorRT timing.
