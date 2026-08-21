# Installation

## Supported target

The live experiment requires an NVIDIA system with CUDA and TensorRT. The
reference setup used for this repository is:

- Jetson AGX Orin, Jetson Linux R36.3 / JetPack 6.0
- CUDA 12.2
- TensorRT 8.6.2 (`trtexec` at `/usr/src/tensorrt/bin/trtexec`)
- Python 3.10
- NVIDIA PyTorch 2.4 and torchvision 0.19 builds for JetPack 6

Other Orin/JetPack combinations can build their own engines, but results from
different CUDA, TensorRT, power-mode, or clock configurations should not be
treated as directly comparable. Non-NVIDIA systems can run unit tests and
inspect the CLI, but cannot execute `run_hayeonp_no_inflation.sh`. Analytical
dry runs additionally require populated timing metadata.

Plan for at least 160 GB of free space; 200 GB or more is recommended for the
six-model FP32 run. TensorRT engines are generated locally and are not portable
across arbitrary TensorRT or device versions.

## System packages

Install CMake, a C++ compiler, Git, and Linux capability tools. CUDA and
TensorRT normally come from JetPack.

```bash
sudo apt update
sudo apt install -y build-essential cmake git libcap2-bin
```

The first CMake configure needs network access to fetch the pinned
`nlohmann/json` v3.11.3 dependency.

## Python environment

Create the environment and install the NVIDIA-provided PyTorch/torchvision
wheels that match the installed JetPack release. NVIDIA changes wheel URLs
between JetPack releases, so use the matching Jetson PyTorch installation
instructions rather than a generic PyPI CPU wheel.

```bash
conda create -n trt python=3.10 -y
conda activate trt

# Install matching NVIDIA Jetson torch and torchvision wheels first.
python -m pip install -r requirements.txt
python -m pip install pytest
```

The optional `vit_tiny` and `yolo11s` experiments use local
PyTorch model definitions. They do not require timm, Ultralytics, or a network
download of model weights.

Verify the environment:

```bash
conda run --no-capture-output -n trt python -c \
  "import torch, torchvision, onnx, numpy, matplotlib; print(torch.__version__, torchvision.__version__)"
/usr/src/tensorrt/bin/trtexec --version
```

If `trtexec` is installed elsewhere, export its absolute path before running:

```bash
export TRTEXEC=/path/to/trtexec
```

## Build the wall-time profiler

The experiment runner builds this target automatically when it is missing. To
build it explicitly:

```bash
cmake -S cpp_runtime -B cpp_runtime/build_sync_wall \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_runtime/build_sync_wall \
  --target table4_runner --parallel "$(nproc)"
sudo setcap cap_sys_nice+ep cpp_runtime/build_sync_wall/table4_runner
getcap cpp_runtime/build_sync_wall/table4_runner
```

`cap_sys_nice` is required because the profiler uses `SCHED_FIFO`. Rebuilds can
replace the binary and remove its capability, so the runner checks it on every
start and invokes `sudo setcap` only when necessary.

## Device preparation

Use the same power mode and clock policy on every device. The reference device
used MAXN. Changing the mode requires administrator access and can affect
timings.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q
sudo jetson_clocks --show
```

Do not run another GPU workload concurrently with profiling.

## Verify the checkout

From the repository root:

```bash
conda run --no-capture-output -n trt python -m pytest -q

CONFIGS_OVERRIDE=1_base.yaml \
MODELS_OVERRIDE=alexnet \
UTILIZATIONS_OVERRIDE=0.5 \
NUM_TASKSETS=1 \
RUN_LABEL=smoke_no_inflation \
./run_hayeonp_no_inflation.sh
```

The smoke run is live: on an empty checkout it exports ONNX, builds TensorRT
engines, profiles missing intervals, and writes generated data locally.
