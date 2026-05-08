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
| `--algorithms` | uni:opt uni:heu ss:tol-fb | Algorithms (canonical or paper-label, e.g. `SS-tol-fb`) |
| `--algorithm-set` | — | Predefined set: `main4`, `full8`, `ss_only`, `uni_only` |
| `--live` | False | Enable live TRT profiling |

### Algorithm sets

| Set | Algorithms |
|-----|-----------|
| `main4` | SS-tol-fb, UNI-tol-fb, SS-opt, UNI-opt |
| `full8` | All 8 SS+UNI variants |
| `ss_only` | SS-opt, SS-heu, SS-tol, SS-tol-fb |
| `uni_only` | UNI-opt, UNI-heu, UNI-tol, UNI-tol-fb |

Both paper-style labels (`SS-tol-fb`) and canonical forms (`ss:tol-fb`) are accepted in
`--algorithms`. `--algorithm-set` overrides `--algorithms`.

### Progress output columns

```
SCHED wall=0.05s masks=8 real=0 cache=0 k1=8 skipped=0 split=False
```
- `masks`: total mask evaluations including K=1 baseline
- `real`: selected-mask TRT profiles (should be 0 in dry-run)
- `cache`: mask-level cache hits
- `k1`: K=1 no-split baseline evaluations (never triggers TRT)
- `skipped`: live evals suppressed by LiveProfileBudget

### Output

```
results/dnn_experiments/<run-name>/
  fig5_design_time_summary.csv   ← mean masks/time per algorithm
  per_taskset_algorithm_results.csv
  summary.md
```

New CSV columns vs. prior version: `baseline_k1_hits`, `interval_onnx_cache_hits/misses`,
`interval_engine_cache_hits/misses`, `interval_engine_build_wall_s`.

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

---

## Full 2×4 Algorithm Matrix (full8)

The complete comparison uses all eight SS + UNI algorithm variants.

### Available Algorithm Sets

| Set | Algorithms |
|-----|-----------|
| `main4` (default) | SS-tol-fb, UNI-tol-fb, SS-opt, UNI-opt |
| `full8` | All 8: SS-opt, SS-heu, SS-tol, SS-tol-fb, UNI-opt, UNI-heu, UNI-tol, UNI-tol-fb |
| `ss_only` | SS-opt, SS-heu, SS-tol, SS-tol-fb |
| `uni_only` | UNI-opt, UNI-heu, UNI-tol, UNI-tol-fb |

### Dry-run full8 sweep (recommended starting point)

```bash
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 50 \
    --dry-run \
    --run-name fig4_full8_dry
```

### Plotting full8 results

```bash
# All 8 algorithms in one plot
python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/fig4_full8_dry \
    --plot-mode all \
    --output fig4_full8 \
    --output-dir results/plots

# SS-only and UNI-only sub-plots
python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/fig4_full8_dry \
    --plot-mode ss_only \
    --output fig4_full8 \
    --output-dir results/plots

python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/fig4_full8_dry \
    --plot-mode uni_only \
    --output fig4_full8 \
    --output-dir results/plots
```

### Explicit algorithm override

```bash
# Run only SS-tol-fb and SS-opt
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --algorithms ss:tol-fb ss:opt \
    --dry-run \
    --run-name fig4_ss_subset
```

### Live full8 with profile cap (Jetson)

After running preflight (`scripts/20_preflight_design.py`):

```bash
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 100 \
    --global-max-real-profiles 200 \
    --run-name fig4_full8_live
```

---

## Interval Cache and Cold-Cache Design-Time Estimation

When running Fig.5 in live mode, the profiling pipeline uses a two-level cache:

1. **Mask-level cache** (`results/evaluations/`): If the exact same mask was already evaluated,
   the stored result is returned immediately.
2. **Interval-level cache** (`artifacts/chunk_cache/`): If a chunk with the same `source_chunk_ids`
   was already exported/built for any previous mask, the ONNX/engine is reused.

`fig5_design_time_summary.csv` includes:
- `total_interval_cache_hits` / `total_interval_cache_misses` — chunk-level reuse counts
- `mean_total_export_wall_s`, `mean_total_build_wall_s`, `mean_total_profile_wall_s` — phase wall times
- `mean_total_estimated_cold_s` — estimated design time if the interval cache had been empty

The cold-cache estimate is computed by summing per-interval `export_wall_s` and
`build_{precision}_wall_s` from `artifacts/chunk_cache/{model}/int_{start}_{end}/timing.json`
plus the actual profile time. This allows comparing algorithms by design-time cost even when
the benchmark benefits from caching across tasksets.

---

## Reproducing Paper Results

For paper-faithful results, use `configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml` with:
- `--num-tasksets-override 500` (full dataset, slow on live)
- `--split-policy major_blocks`
- Live mode on Jetson AGX Orin with `fp32` precision
