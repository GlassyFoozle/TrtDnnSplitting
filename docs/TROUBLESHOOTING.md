# Troubleshooting

## Conda environment is missing or incomplete

The runner checks that `torch`, `torchvision`, `onnx`, `numpy`, and
`matplotlib` import in `CONDA_ENV`.

```bash
conda run --no-capture-output -n trt python -c \
  "import torch, torchvision, onnx, numpy, matplotlib"
```

Install NVIDIA's JetPack-compatible PyTorch and torchvision wheels before
installing `requirements.txt`.

## `trtexec` not found

The default is `/usr/src/tensorrt/bin/trtexec`. For another installation:

```bash
TRTEXEC=/absolute/path/to/trtexec ./run_hayeonp_no_inflation.sh
```

Do not reuse engines built by an incompatible TensorRT version.

## CMake configure or build fails

Confirm that CUDA, TensorRT headers/libraries, a compiler, CMake, and network
access for `nlohmann/json` are available:

```bash
nvcc --version
ls /usr/include/aarch64-linux-gnu/NvInfer.h
cmake -S cpp_runtime -B cpp_runtime/build_sync_wall \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_runtime/build_sync_wall \
  --target table4_runner --parallel "$(nproc)"
```

## `setcap`, `SCHED_FIFO`, or permission failure

Install `libcap2-bin` and grant the profiler `cap_sys_nice`:

```bash
sudo apt install -y libcap2-bin
sudo setcap cap_sys_nice+ep cpp_runtime/build_sync_wall/table4_runner
getcap cpp_runtime/build_sync_wall/table4_runner
```

Rebuilding the executable may remove the capability.

## ONNX export or TensorRT engine build fails

Inspect the newest files under `artifacts/logs/`. Common causes are an
incompatible PyTorch/torchvision wheel, insufficient memory, insufficient disk
space, or a TensorRT parser/build failure.

Run a one-model smoke test before the full matrix:

```bash
CONFIGS_OVERRIDE=1_base.yaml \
MODELS_OVERRIDE=alexnet \
UTILIZATIONS_OVERRIDE=0.5 \
NUM_TASKSETS=1 \
RUN_LABEL=smoke_no_inflation \
./run_hayeonp_no_inflation.sh
```

## Disk-space guard aborts

The full observed FP32 interval cache is approximately 134 GB and the eight
experiment result directories add several GB. Start with at least 160 GB free;
200 GB is recommended. `MIN_FREE_GB` is a safety reserve, not an estimate of
total required storage.

Generated data is ignored by Git. Review it before using the cleanup helper:

```bash
conda run -n trt python scripts/22_clean_generated_artifacts.py --help
du -sh artifacts/* results/* 2>/dev/null
```

## A run was interrupted

Use the same cache paths to reuse completed ONNX exports, engines, and measured
mask results. Use a new `RUN_LABEL` if the previous output directory must be
preserved. Do not use `--force-profile` unless intentionally rebuilding and
remeasuring artifacts.

## Results differ between devices

Check JetPack/CUDA/TensorRT versions, power mode, clocks, temperature, builder
optimization level, precision, WCET metric, Git commit, and cache provenance.
TensorRT tactic selection and synchronized wall-time measurements can differ
across these conditions.

## Python tests fail

Run tests from the repository root in the configured environment:

```bash
conda run --no-capture-output -n trt python -m pytest -q
```

No external sibling repository is required.
