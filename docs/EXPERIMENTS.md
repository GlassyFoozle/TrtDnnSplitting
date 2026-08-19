# Experiment Semantics

## Canonical no-inflation experiment

`run_hayeonp_no_inflation.sh` invokes
`scripts/33_run_yaml_fig4_with_split_plots.py`, which runs the YAML experiment
driver and then generates schedulability and split-activity plots.

The experiment uses:

- configurations `1_base.yaml` through `8_singleCPU_task8.yaml` under
  `configs/yaml/gpu_util_configs/`;
- AlexNet, ResNet18, ViT-B/16, VGG19, InceptionV3, and
  MobileNetV3-Small;
- target GPU utilizations 0.5 through 0.9;
- 50 independently generated task sets per utilization;
- SS-heuristic, SS-tolerance-feedback, UNI-heuristic, and
  UNI-tolerance-feedback;
- TensorRT FP32 with builder optimization level 1;
- the `trt_fusion_safe` model-specific split policy;
- maximum synchronized CPU wall time as the WCET metric.

## Meaning of “no inflation”

Three controls isolate this experiment from the monotonic-repair path:

1. `TRT_RAW_INTERVAL_TIMING=1` selects raw measured interval timing.
2. `TRT_DISABLE_MONOTONIC_INFLATION=1` prevents interval timing inflation.
3. `--no-monotonic-k-split-cache` prevents legacy monotonic-envelope
   adjustment of measured best-K cache entries.

The runner also uses separate evaluation and profiling cache filenames with a
`no_inflation` suffix. The measured best-K cache remains enabled; it stores and
reuses measured selections but does not impose the disabled monotonic envelope.

## Cold-start behavior

On a checkout without generated artifacts:

1. Shipped `dag_aligned_full.json` files define legal base chunks and model
   graph metadata.
2. Task generation requests a measured K=1 value for each selected model.
3. Missing K=1 and candidate masks are converted to selected split configs.
4. Each new interval is exported to ONNX and built with `trtexec`.
5. `table4_runner` measures synchronized CPU wall time and GPU timing.
6. Interval, exact-mask, profiling, and measured best-K caches are updated.
7. Subsequent task sets and configurations reuse those artifacts.

TensorRT engines are deliberately not stored in Git. They must be rebuilt when
moving to a different device or incompatible TensorRT/CUDA stack.

## Workload matrix

| Config | CPU/task variation |
|---|---|
| `1_base.yaml` | 8 CPUs, 1–3 tasks/CPU, CPU ratio 0–0.5 |
| `2_C_ratio_00.yaml` | CPU ratio fixed at 0 |
| `3_C_ratio_25.yaml` | CPU ratio fixed at 0.25 |
| `4_C_ratio_50.yaml` | CPU ratio fixed at 0.5 |
| `5_task1.yaml` | one task per CPU |
| `6_task3.yaml` | three tasks per CPU |
| `7_singleCPU_task4.yaml` | one CPU, four tasks |
| `8_singleCPU_task8.yaml` | one CPU, eight tasks |

The command-line `--num-tasksets 50` overrides the YAML files' historical
`n_task_sets: 100` value.

## Interpretation and comparability

- The utilization column is target DNN/GPU utilization.
- K=1 is a measured all-zero-mask TensorRT configuration, not a sum of
  independently measured finest chunks.
- Candidate interval timings need not be monotonic as boundaries are added or
  removed; this experiment intentionally preserves the raw measurements.
- SS/UNI tolerance-feedback results include final RTA and tolerance checks.
- Different power modes, clock policies, thermal conditions, TensorRT tactics,
  or builder versions can change WCET and therefore schedulability.

Record the following with every reported run:

- Git commit and run label;
- Jetson Linux/JetPack, CUDA, TensorRT, PyTorch, and torchvision versions;
- device model, power mode, and clock policy;
- free disk space and whether caches were cold or warm;
- any non-default runner environment overrides.
