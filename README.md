# TrtDnnSplitting

TensorRT-backed DNN splitting and schedulability analysis for NVIDIA Jetson
Orin. The repository implements SS and UNI variants with heuristic and
tolerance-feedback splitting, generates model-specific TensorRT artifacts on
the target device, and evaluates schedulability over generated task sets.

The canonical rebuttal experiment is
[`run_hayeonp_no_inflation.sh`](run_hayeonp_no_inflation.sh). It evaluates six
models over eight workload configurations using raw measured interval timing,
without monotonic timing inflation.

## Reproduce the canonical experiment

1. Prepare the Jetson and the `trt` Conda environment as described in
   [`INSTALL.md`](INSTALL.md).
2. Clone this repository on the target device.
3. Run a small smoke test:

   ```bash
   CONFIGS_OVERRIDE=1_base.yaml \
   MODELS_OVERRIDE=alexnet \
   UTILIZATIONS_OVERRIDE=0.5 \
   NUM_TASKSETS=1 \
   RUN_LABEL=smoke_no_inflation \
   ./run_hayeonp_no_inflation.sh
   ```

4. Run the full experiment:

   ```bash
   ./run_hayeonp_no_inflation.sh
   ```

The runner builds `table4_runner` if needed, checks the Python/TensorRT
environment, creates ONNX and TensorRT engines lazily, profiles missing masks,
and writes all generated data below `artifacts/` and `results/`. Generated
artifacts and results are intentionally excluded from Git.

See [`docs/RUN_HAYEONP_NO_INFLATION.md`](docs/RUN_HAYEONP_NO_INFLATION.md) for
the complete procedure, cache semantics, environment overrides, output layout,
and recovery instructions.

Two additional, opt-in model keys are available without changing the canonical
six-model run:

```bash
CONFIGS_OVERRIDE=1_base.yaml \
MODELS_OVERRIDE="vit_tiny yolo11s" \
UTILIZATIONS_OVERRIDE=0.5 \
NUM_TASKSETS=1 \
RUN_LABEL=smoke_vit_tiny_yolo11s \
./run_hayeonp_no_inflation.sh
```

Their deterministic PyTorch constructors, DAG-aligned baseline metadata, and
`trt_fusion_safe` policies are versioned in the repository. A fresh target
therefore generates its own ONNX chunks, TensorRT engines, and timing cache in
the same way as the original models; no timm or Ultralytics package is needed.
See [`docs/VIT_TINY_YOLO11S.md`](docs/VIT_TINY_YOLO11S.md) for the exact indexed
boundaries and grouping rationale.

## Experiment scope

- Models: AlexNet, ResNet18, ViT-B/16, VGG19, InceptionV3, MobileNetV3-Small
- Workloads: eight YAML configurations in `configs/yaml/gpu_util_configs/`
- Utilizations: 0.5, 0.6, 0.7, 0.8, 0.9
- Task sets: 50 per utilization by default
- Precision: FP32
- Split policy: `trt_fusion_safe`
- Algorithms: SS-heuristic, SS-tolerance-feedback, UNI-heuristic,
  UNI-tolerance-feedback
- WCET metric: measured maximum CPU wall time around synchronized TensorRT
  execution

## Repository layout

```text
configs/                       workload and split-policy definitions
artifacts/split_configs/       versioned baseline split metadata
cpp_runtime/                   C++ TensorRT wall-time profiler
scripts/                       experiment, plotting, export, and build tools
src/                           RTA, integration, splitting, and optimization code
tests/                         unit and regression tests
artifacts/                     generated ONNX/engine/interval caches (ignored)
results/                       generated experiment outputs (ignored)
```

The RTA implementation under `src/rta/` is self-contained; no sibling
DNNSplitting repository is required.

## Documentation

- [`INSTALL.md`](INSTALL.md): target prerequisites and installation
- [`docs/RUN_HAYEONP_NO_INFLATION.md`](docs/RUN_HAYEONP_NO_INFLATION.md): canonical experiment
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md): runner controls and outputs
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): experiment semantics
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md): common failures
- [`docs/E2E_ARCHITECTURE.md`](docs/E2E_ARCHITECTURE.md): source and artifact data flow
- [`docs/report_trt_fusion_safe_model_summary_ko.md`](docs/report_trt_fusion_safe_model_summary_ko.md): split-policy rationale
- [`docs/VIT_TINY_YOLO11S.md`](docs/VIT_TINY_YOLO11S.md): ViT-Tiny and YOLO11s boundaries and fresh-device usage

## Tests

```bash
conda run --no-capture-output -n trt python -m pytest -q
```
