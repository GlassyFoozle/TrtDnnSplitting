# Installation

## Target platform

NVIDIA Jetson AGX Orin (JetPack 5.x / 6.x, TensorRT 8.6+, CUDA 11.4+).
Dry-run mode (schedulability analysis only) works on any Linux machine with Python ≥ 3.8.

## Python environment

```bash
conda create -n trt python=3.10
conda activate trt

# PyTorch for ONNX export (Jetson wheel — adjust version as needed)
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# Core Python deps
pip install numpy matplotlib onnx
```

> PyYAML is NOT required — the YAML parser in `scripts/30_run_yaml_fig4_experiment.py`
> is a lightweight built-in implementation.

## C++ profiler

```bash
cd cpp_runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binary: cpp_runtime/build/table4_runner
```

Requires: CMake ≥ 3.18, TensorRT headers and libraries on the system path,
CUDA toolkit matching the TensorRT version.

## Verify install

```bash
# Check Python imports (no GPU needed)
python -c "from src.rta import get_SS_R, get_UNI_R_and_K; print('RTA OK')"

# Dry-run smoke test
python scripts/30_run_yaml_fig4_experiment.py \
    --yaml configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet \
    --policy major_blocks \
    --n-tasksets 2 \
    --dry-run
```

## Directory permissions

Ensure the process has write access to `artifacts/` and `results/`:

```bash
mkdir -p artifacts results
```

Both directories contain a `.gitkeep` and are tracked by git; their generated
contents are excluded via `.gitignore`.
