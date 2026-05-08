# TrtDnnSplitting — User Guide

## Overview

TrtDnnSplitting implements four DNN splitting scheduling algorithms for Jetson Orin real-time
systems. It evaluates split-point strategies for DNN inference tasks under SS and UNI
real-time analysis, with dry-run (analytical) and live (TensorRT) modes.

## Prerequisites

See [INSTALL.md](../INSTALL.md) for full setup. Quick summary:

- Python 3.10+, PyTorch, ONNX, numpy, pandas, matplotlib
- TensorRT 8.6+ (live mode only) on Jetson AGX Orin
- `conda activate trt` before running scripts

## Dry-Run vs. Live Mode

| Mode | What it does | When to use |
|------|--------------|-------------|
| Dry-run (default) | Uses pre-profiled `dag_aligned_full.json` chunk times | Fresh clone, algorithm testing |
| Live | Builds TRT engines, profiles on real GPU | Actual Jetson Orin experiments |

All scripts default to dry-run. Pass `--live` (script 30) or no flag needed (scripts 40/41).

## Quick Start

```bash
# 1. Inspect a single taskset with all four algorithms
python scripts/10_inspect_single_taskset.py \
    --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json \
    --dry-run

# 2. Run Fig.4-style schedulability sweep (dry-run)
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --num-tasksets-override 20 \
    --dry-run

# 3. Plot Fig.4 results
python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/<run-name> \
    --output-dir results/plots

# 4. Run Fig.5 design-time experiment
python scripts/40_run_fig5_design_time.py \
    --models alexnet resnet18 vgg19 \
    --num-tasks 8 --num-tasksets 50

# 5. Plot Fig.5 results
python scripts/41_plot_fig5.py \
    --run results/dnn_experiments/<run-name> \
    --output-dir results/plots
```

## Algorithms

| Label | Model | Algorithm | Description |
|-------|-------|-----------|-------------|
| `SS_ours` | SS | `ss:tol-fb` | Tolerance-feedback splitting (ours) |
| `UNI_ours` | UNI | `uni:tol-fb` | UNI tolerance-feedback splitting (ours) |
| `SS_Buttazzo` | SS | `ss:opt` | Buttazzo OPT baseline |
| `UNI_Buttazzo` | UNI | `uni:opt` | UNI OPT baseline |

Pass `--algorithms SS_ours UNI_ours SS_Buttazzo UNI_Buttazzo` to select algorithms.

## Split Policies

Controlled via `--split-policy`. Available options:

| Policy | Description |
|--------|-------------|
| `all` | All N-1 boundaries enabled |
| `major_blocks` | Only boundaries at major DNN block transitions |
| `paper_like` | As used in the original paper |
| `stage` | Stage-level granularity |
| `five_points` | Exactly 5 evenly-spaced split points |
| `ten_points` | Exactly 10 evenly-spaced split points |

Policy definitions: `configs/split_point_policies.json`

## Taskset Configuration

### YAML-based (Fig.4)

YAML files in `configs/yaml/` control workload generation for Fig.4 sweeps:

```yaml
n_task_sets: 500
utilization_range: [0.7, 0.9]
utilization_step: 0.05
number_of_tasks_per_cpu_range: [1, 1]
period_range: [1000, 10000]
G_ratio_range: [0.6, 1.0]
```

Pass to `--config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml`.

### JSON-based (direct)

Static tasksets live in `configs/dnn_tasksets/`. Use `10_inspect_single_taskset.py`.

## Results Layout

```
results/
  dnn_experiments/<run-name>/
    schedulability_ratio.csv   ← per-algorithm schedulability by utilization
    all_results.json           ← full per-taskset result records
    per_taskset_results.csv    ← compact CSV version
    split_activity.csv         ← split trigger statistics
    summary.md                 ← human-readable run summary
  plots/
    <name>.png / <name>.pdf    ← produced by 31_plot_fig4.py / 41_plot_fig5.py
```

## Supported Models

| Model | Chunks (N) | Note |
|-------|-----------|------|
| alexnet | 22 | Shipped in `artifacts/split_configs/` |
| resnet18 | 14 | Shipped in `artifacts/split_configs/` |
| vgg19 | 46 | Shipped in `artifacts/split_configs/` |

Additional models require running `scripts/20_preflight_design.py` to generate split configs.

## Running Tests

```bash
conda run -n trt python -m pytest tests/ -q
```

All 19 tests cover: RTA import, taskset generation, split policy, cache validity.
