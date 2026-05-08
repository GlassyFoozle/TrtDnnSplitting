# TrtDnnSplitting

Standalone implementation of DNN-splitting schedulability analysis for NVIDIA Jetson Orin.
Profiles TensorRT split-compiled DNN inference segments, then applies SS and UNI
real-time scheduling analysis to find the minimum-splitting configuration that meets
task deadlines.

## Repository layout

```
TrtDnnSplitting/
├── configs/
│   ├── yaml/                     # Fig.4 experiment YAML configs (5 variants)
│   ├── dnn_tasksets/             # Hand-crafted taskset JSON files
│   └── split_point_policies.json # Per-model boundary policies (all/major_blocks/…)
├── cpp_runtime/                  # C++ profiler source (table4_runner)
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── scripts/
│   ├── 10_inspect_single_taskset.py   # Run one algorithm on one taskset JSON
│   ├── 20_preflight_design.py         # Live design preflight: export/build/profile base
│   ├── 30_run_yaml_fig4_experiment.py # Fig.4 schedulability sweep (YAML-driven)
│   ├── 31_plot_fig4.py                # Plot Fig.4 results
│   ├── 40_run_fig5_design_time.py     # Fig.5 design-time / profiling-cost sweep
│   ├── 41_plot_fig5.py                # Plot Fig.5 results
│   ├── internal_export_selected_split.py   # (called by compiler.py — not user-facing)
│   └── internal_build_selected_engines.sh  # (called by compiler.py — not user-facing)
├── src/
│   ├── rta/          # SS + UNI scheduling analysis (ported from DNNSplitting paper)
│   ├── integration/  # DNN task model, algorithm runner, mask applicator
│   ├── optimization/ # Mask evaluator, compiler, profiling DB
│   ├── splitting/    # DAG-aligned split generation
│   ├── export/       # ONNX exporter
│   ├── models/       # Model registry (AlexNet, ResNet18, VGG19)
│   └── utils/        # Path helpers
├── tests/            # Unit tests
├── artifacts/        # Generated: ONNX models, TRT engines, split configs
└── results/          # Generated: evaluation JSONs, plots
```

## Quick start

### 1. Build the C++ profiler

```bash
cd cpp_runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Dry-run (no GPU required)

Uses pre-profiled `dag_aligned_full` timings for schedulability analysis without
building any new engines:

```bash
python scripts/30_run_yaml_fig4_experiment.py \
    --yaml configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --policy major_blocks \
    --n-tasksets 2 \
    --dry-run
```

### 3. Live mode (Jetson Orin with TensorRT)

Remove `--dry-run` and add `--max-real-profiles N` to cap GPU profiling cost:

```bash
python scripts/30_run_yaml_fig4_experiment.py \
    --yaml configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --policy major_blocks \
    --n-tasksets 50 \
    --max-real-profiles 500
```

### 4. Plot results

```bash
python scripts/31_plot_fig4.py --results results/yaml_fig4/
```

## Algorithms

| Flag | Name | Description |
|------|------|-------------|
| `ss:tol-fb` | SS_ours | SS RTA + greedy tolerance-feedback splitting |
| `ss:opt` | SS_Buttazzo | SS RTA + BFS-optimal splitting |
| `uni:tol-fb` | UNI_ours | UNI RTA + greedy tolerance-feedback splitting |
| `uni:opt` | UNI_Buttazzo | UNI RTA + BFS-optimal splitting |

## Dependencies

- Python ≥ 3.8 (no PyYAML required — YAML parser is built-in)
- PyTorch (for ONNX export)
- TensorRT ≥ 8.6 (for engine build/profile; not required for dry-run)
- ONNX Runtime (optional; for verification)
- matplotlib (for plotting)
- numpy

See `INSTALL.md` for detailed setup instructions.

## Self-contained RTA

`src/rta/` contains the scheduling analysis code ported from the DNNSplitting
research codebase. **No external `../DNNSplitting` sibling repo is required.**
