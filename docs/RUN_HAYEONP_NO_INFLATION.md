# `run_hayeonp_no_inflation` 재현 가이드

이 문서는 artifact가 없는 새 Jetson Orin에서 저장소 clone부터 FP32
no-inflation 실험까지 수행하는 기준 절차다.

## 1. 필요한 장치와 공간

- 기준 환경: Jetson AGX Orin, Jetson Linux R36.3 / JetPack 6.0
- CUDA 12.2, TensorRT 8.6.2
- Python 3.10, JetPack용 PyTorch 2.4/torchvision 0.19 계열
- 최소 160GB, 권장 200GB 이상의 여유 공간
- 최초 package/CMake dependency 설치를 위한 인터넷 연결
- `table4_runner`에 `cap_sys_nice`를 부여할 수 있는 sudo 권한

다른 JetPack/TensorRT 버전에서도 artifact를 새로 빌드할 수 있지만 서로
다른 환경에서 측정한 WCET와 schedulability 결과를 직접 비교하면 안 된다.

## 2. 저장소와 Python 환경 준비

```bash
git clone https://github.com/GlassyFoozle/TrtDnnSplitting.git
cd TrtDnnSplitting

conda create -n trt python=3.10 -y
conda activate trt

# 먼저 설치된 JetPack과 맞는 NVIDIA PyTorch/torchvision wheel을 설치한다.
python -m pip install -r requirements.txt
python -m pip install pytest
```

세부 system package, PyTorch wheel, build 조건은 루트의 `INSTALL.md`를
따른다.

## 3. 장치 상태 고정

모든 장치에서 동일한 power mode와 clock 설정을 사용한다. 기준 장치는
MAXN을 사용했다.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q
sudo jetson_clocks --show
```

실험 중 다른 CUDA/TensorRT workload를 동시에 실행하지 않는다.

## 4. 테스트와 소규모 live smoke run

```bash
conda run --no-capture-output -n trt python -m pytest -q

CONFIGS_OVERRIDE=1_base.yaml \
MODELS_OVERRIDE=alexnet \
UTILIZATIONS_OVERRIDE=0.5 \
NUM_TASKSETS=1 \
RUN_LABEL=smoke_no_inflation \
./run_hayeonp_no_inflation.sh
```

runner는 다음을 자동으로 확인하거나 수행한다.

1. `trt` Conda 환경과 필수 Python import 확인
2. `trtexec` 확인
3. `cpp_runtime/build_sync_wall/table4_runner` 자동 build
4. `cap_sys_nice`가 없을 때만 `sudo setcap` 실행
5. K=1과 필요한 candidate interval의 ONNX export
6. FP32 TensorRT engine build
7. synchronized wall-time profiling과 cache 기록
8. 결과 CSV/JSON 및 plot 생성

smoke run이 실패하면 full run을 시작하지 말고
`docs/TROUBLESHOOTING.md`를 확인한다.

## 5. 전체 실험

```bash
./run_hayeonp_no_inflation.sh
```

기본 실험 범위는 다음과 같다.

- YAML 8종
- model 6종: AlexNet, ResNet18, ViT-B/16, VGG19, InceptionV3,
  MobileNetV3-Small
- utilization 0.5, 0.6, 0.7, 0.8, 0.9
- utilization별 task set 50개
- SS-heu, SS-tol-fb, UNI-heu, UNI-tol-fb
- FP32, TensorRT builder optimization level 1
- `trt_fusion_safe` split policy
- max synchronized CPU wall-time WCET

## 6. no-inflation 설정

runner의 기본값은 다음과 같다.

```text
TRT_RAW_INTERVAL_TIMING=1
TRT_DISABLE_MONOTONIC_INFLATION=1
MONOTONIC_K_SPLIT_CACHE=0
K_SPLIT_CACHE_PATH=results/optimization/measured_k_split_cache_no_inflation.json
EVALUATION_CACHE_DIR=results/evaluations_no_inflation
PROFILING_CACHE_PATH=results/optimization/.profiling_cache_no_inflation.json
```

raw interval timing을 사용하고 interval inflation과 legacy monotonic K-cache
envelope를 비활성화한다. 일반 measured best-K cache 재사용은 유지된다.

## 7. 생성되는 파일

- `artifacts/chunk_cache/`: interval ONNX, FP32 engine, timing metadata
- `artifacts/split_configs/`: 실행 중 생성된 mask별 config
- `artifacts/logs/`: TensorRT build log
- `results/evaluations_no_inflation/`: mask별 측정 결과
- `results/optimization/*no_inflation*`: profiling 및 measured-K cache
- `results/dnn_experiments/<run-name>/`: task set, 집계 CSV/JSON, summary, plot

모두 `.gitignore` 대상이며 GitHub에 올리지 않는다.

## 8. 일부 범위만 실행

```bash
CONFIGS_OVERRIDE="1_base.yaml 5_task1.yaml" \
MODELS_OVERRIDE="alexnet resnet18" \
UTILIZATIONS_OVERRIDE="0.5 0.7" \
NUM_TASKSETS=5 \
RUN_LABEL=partial_no_inflation \
./run_hayeonp_no_inflation.sh
```

중단 후 같은 cache 경로를 사용하면 완성된 interval과 mask 측정을
재사용한다. 기존 결과 디렉터리를 보존해야 하면 `RUN_LABEL`만 새 값으로
바꾼다.

## 9. 결과와 함께 기록할 정보

- Git commit hash와 `RUN_LABEL`
- Jetson model, JetPack/Jetson Linux, CUDA, TensorRT 버전
- Python, PyTorch, torchvision 버전
- power mode와 clock 설정
- cold-cache/warm-cache 여부
- 기본값과 다르게 지정한 모든 환경변수
