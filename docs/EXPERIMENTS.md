# TrtDnnSplitting — Experiments Guide

## Overview

Two main experiments correspond to the paper figures:

| Script | Figure | What it measures |
|--------|--------|-----------------|
| `30_run_yaml_fig4_experiment.py` | Fig. 4 | Schedulability ratio vs. GPU utilization |
| `40_run_fig5_design_time.py` | Fig. 5 | Design-time cost (masks evaluated, wall time) |

## Fig. 4 — Schedulability Ratio Sweep

### Config files

Five pre-defined YAML configs in `configs/yaml/`:

| Config | GPU util range | Tasks | CPU overlap |
|--------|---------------|-------|-------------|
| `1_GPU0.6-1.0_task8_ov5.yaml` | 0.6–1.0 | 8 | yes |
| `2_GPU0.6-1.0_task8_singleCPU_ov5.yaml` | 0.6–1.0 | 8 | single CPU |
| `3_GPU1.0_task8_ov5.yaml` | 1.0 fixed | 8 | yes |
| `4_GPU0.8_task8_ov5.yaml` | 0.8 fixed | 8 | yes |
| `5_GPU0.6_task8_ov5.yaml` | 0.6 fixed | 8 | yes |

### Running

```bash
# Dry-run (no TRT engines required)
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --num-tasksets-override 50 \
    --dry-run \
    --run-name fig4_dry_run

# Live (requires TRT engines from 20_preflight_design.py)
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --run-name fig4_live
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | alexnet resnet18 vgg19 | Models to include |
| `--split-policy` | all | Split point policy |
| `--num-tasksets-override` | from YAML | Override n_task_sets |
| `--algorithms` | all four | Algorithms to run |
| `--dry-run` | False | Use analytical timing |
| `--run-name` | auto | Output directory suffix |

### Output

```
results/dnn_experiments/<run-name>/
  schedulability_ratio.csv    ← x=utilization, y=sched_ratio per algorithm
  all_results.json            ← full result records
  per_taskset_results.csv
  split_activity.csv
  summary.md
  yaml_mapping_report.md      ← YAML-to-taskset mapping log
```

### Plotting

```bash
python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/<run-name> \
    --output fig4_result \
    --output-dir results/plots
```

Produces `results/plots/fig4_result.png` and `.pdf`.

---

## Fig. 5 — Design-Time Cost

Measures how many mask evaluations and wall-clock seconds each algorithm requires
across a batch of random tasksets at a fixed utilization.

### Running

```bash
# Dry-run (default)
python scripts/40_run_fig5_design_time.py \
    --models alexnet resnet18 vgg19 \
    --num-tasks 8 \
    --num-tasksets 100 \
    --utilization 0.8 \
    --split-policy major_blocks \
    --run-name fig5_dry

# Live mode
python scripts/40_run_fig5_design_time.py \
    --models alexnet resnet18 vgg19 \
    --num-tasks 8 \
    --num-tasksets 100 \
    --utilization 0.8 \
    --split-policy major_blocks \
    --live \
    --run-name fig5_live
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks` | 8 | Tasks per taskset |
| `--num-tasksets` | 100 | Number of random tasksets |
| `--utilization` | 0.8 | Total GPU utilization |
| `--algorithms` | UNI-opt UNI-heu SS-tol-fb | Algorithms to compare |
| `--live` | False | Enable live TRT profiling |

### Output

```
results/dnn_experiments/<run-name>/
  fig5_design_time_summary.csv   ← mean masks/time per algorithm
  per_taskset_algorithm_results.csv
  summary.md
```

### Plotting

```bash
python scripts/41_plot_fig5.py \
    --run results/dnn_experiments/<run-name> \
    --output fig5_result \
    --output-dir results/plots
```

Produces `fig5_result_cost.png`, `fig5_result_cost.pdf`, `fig5_result_runtime.png`, `fig5_result_runtime.pdf`.

---

## Single Taskset Inspection

```bash
python scripts/10_inspect_single_taskset.py \
    --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json \
    --algorithm SS_ours \
    --dry-run
```

Runs one taskset through one algorithm and prints per-task split decisions.

---

## Live Preflight (Jetson Only)

Before any live experiment, build and profile the base TRT engines:

```bash
# Build engines for all three models
python scripts/20_preflight_design.py \
    --models alexnet resnet18 vgg19 \
    --precision fp32

# Then run live Fig.4
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --run-name fig4_live
```

---

## Reproducing Paper Results

For paper-faithful results, use `configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml` with:
- `--num-tasksets-override 500` (full dataset, slow on live)
- `--split-policy major_blocks`
- Live mode on Jetson AGX Orin with `fp32` precision
