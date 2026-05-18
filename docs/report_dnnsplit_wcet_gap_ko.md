# DNNSplit 논문과 현재 workspace 측정 시간 차이 분석

## 결론

`DNNSplit.pdf`의 Table IV와 현재 workspace의 non-split / full-split 시간을 그대로 비교하면 큰 차이가 나는 것이 정상에 가깝다. 가장 큰 이유는 **같은 하드웨어를 썼더라도 같은 대상을 측정한 것이 아니기 때문**이다.

현재 차이를 크게 만드는 원인은 우선순위대로 다음 네 가지다.

1. 논문은 **CPU가 관측한 wall-clock WCET**를 사용하고, 현재 workspace의 주 경로는 **CUDA event 기반 GPU 시간**을 사용한다.
2. 논문은 NVIDIA **Caffe** 모델을 쓰고, 현재 workspace는 torchvision/ONNX 기반 모델을 쓴다.
3. 논문의 full split과 현재 workspace의 full split은 **split point 수와 위치가 크게 다르다**.
4. TensorRT 버전, builder tactic, precision, engine build 조건이 완전히 동일하다는 근거가 없다.

따라서 지금 숫자는 “논문 재현 실패”라기보다, **서로 다른 실험 정의의 결과**로 보는 것이 맞다.

## 확인한 숫자

### 논문 Table IV

`DNNSplit.pdf` Table IV는 Jetson AGX Orin에서 측정한 값을 다음처럼 보고한다.

| 모델 | 논문 full split chunk 수 | 논문 full split Sum (ms) | 논문 non-split (ms) |
|---|---:|---:|---:|
| AlexNet | 12 | 4.802 | 4.469 |
| InceptionV4 | 7 | 9.129 | 8.670 |
| ResNet18 | 12 | 3.751 | 2.533 |
| VGG19 | 6 | 11.425 | 6.615 |

### 현재 workspace에서 바로 비교 가능한 값

현재 `trt_fusion_safe` K-cache 곡선과 최근 `results/table4/*_k1_fp32.json` 기준 값은 아래와 같다.

| 모델 | 현재 K=1 non-split GPU max 합 (ms) | 현재 `trt_fusion_safe` max-K G (ms) | 현재 max-K | 논문 대비 주의점 |
|---|---:|---:|---:|---|
| AlexNet | 2.680 | 1.777 | 11 | 모델 그래프와 split 정의가 다름 |
| ResNet18 | 1.777 | 1.005 | 11 | 모델 그래프와 split 정의가 다름 |
| VGG19 | 9.563 | 6.459 | 12 | 논문 full split은 6 chunks, 현재는 12 chunks |

이 표만 봐도 full split 쪽은 논문과 현재 workspace가 같은 의미가 아니다. 예를 들어 VGG19는 논문에서 6개 chunk로 나누지만, 현재 `trt_fusion_safe` 정책은 최대 12개 chunk를 허용한다. 더 잘게 나눈 현재 값이 오히려 작게 나오는 것은, 현재 값이 논문식 CPU wall-clock WCET가 아니라 GPU-side timing이고, 현재 그래프/engine fusion 결과도 다르기 때문이다.

### 현재 `dag_aligned_full` 정의도 논문과 다름

현재 `artifacts/split_configs/*/dag_aligned_full.json`의 chunk 수는 다음과 같다.

| 모델 | 현재 `dag_aligned_full` chunk 수 | 논문 full split chunk 수 |
|---|---:|---:|
| AlexNet | 22 | 12 |
| ResNet18 | 14 | 12 |
| VGG19 | 46 | 6 |
| Inception 계열 | `inception_v3` 22 | `InceptionV4` 7 |

특히 VGG19는 `46 vs 6`이라서, “split point 위치가 조금 다르다” 수준이 아니다. 두 실험의 full split은 구조적으로 다른 설정이다.

## 왜 이렇게 차이가 나는가

## 1. 측정 대상이 다르다

논문은 chunk WCET를 정의할 때 “CPU 관점에서 관측한 total latency”를 사용해야 한다고 명시한다. 논문 Sec. VI와 Sec. VII는 CPU wall-clock profiling, inference setup, CPU-GPU communication, synchronization, shared resource management를 포함하는 방향으로 설명한다.

반면 현재 workspace의 핵심 측정 경로는 `cpp_runtime/src/chunk_pipeline.cpp`와 `cpp_runtime/src/main_table4.cpp`다.

- `ChunkPipeline::run()`은 CUDA event로 `full_engine_gpu_*`, `total_chunked_gpu_*`, `chunk gpu_*`를 잰다.
- chunk별 CPU 시간도 남기지만, 현재 measured-K / schedulability 쪽에서 주로 소비하는 값은 `per_chunk_gpu_max_ms` 계열이다.
- `scripts/21_profile_base_chunks.py`도 요약 출력에서 GPU mean/p99/max 합을 중심으로 쓴다.

즉 논문은 “스케줄러가 실제로 부담하는 실행 시간”을 보고 있고, 현재 workspace는 “GPU가 실제 kernel 실행에 쓴 시간”에 더 가까운 값을 보고 있다.

이 차이는 full split에서 특히 커진다. split이 많아질수록 chunk마다 다음 비용이 반복된다.

- TensorRT enqueue / execute 호출
- scheduler dispatch
- resource acquire / release
- CPU-GPU synchronization
- chunk boundary 처리

논문은 이런 비용을 WCET에 넣는다. 현재 GPU event timing은 대부분 넣지 않는다. 그래서 논문에서는 full split 합이 non-split보다 커지는데, 현재 workspace에서는 full split 총 G가 더 작게 보이는 경우도 생긴다.

## 2. 모델 그래프가 다르다

논문 Sec. VIII-A는 NVIDIA pretrained **Caffe** 모델을 사용했다고 밝힌다.

- AlexNet
- ResNet18
- VGG19
- InceptionV4

현재 workspace는 torchvision 모델을 ONNX로 내보낸 그래프를 사용한다.

- 같은 이름의 AlexNet / ResNet18 / VGG19라도 Caffe graph와 PyTorch/ONNX graph는 동일하지 않을 수 있다.
- layer folding, BatchNorm 처리, ReLU 위치, ONNX export 형태가 달라질 수 있다.
- TensorRT가 선택하는 fusion과 tactic도 달라진다.
- 논문은 `InceptionV4`, 현재 workspace는 `inception_v3`다. 이건 직접 비교 대상이 아니다.

같은 GPU 위에서도 그래프가 다르면 non-split 시간부터 달라지는 것이 자연스럽다.

## 3. full split 정의가 다르다

논문은 일부 모델에서 split point 수를 의도적으로 줄였다. 본문은 VGG19와 InceptionV4에서 major architectural block 경계만 남겼다고 설명한다. 그 결과 Table IV의 full split은 VGG19 6 chunks, InceptionV4 7 chunks다.

현재 workspace는 `dag_aligned_full` 기준으로 더 세밀한 split 후보를 갖고, 실험용 `trt_fusion_safe` 정책도 별도의 허용 boundary 집합을 쓴다.

따라서 아래 셋은 서로 다른 개념이다.

- 논문 Table IV의 fully split
- 현재 `dag_aligned_full`
- 현재 `trt_fusion_safe`에서의 max-K split

이 셋을 이름만 보고 같은 full split으로 놓으면 해석이 틀어진다.

## 4. build / runtime 조건이 완전히 같지 않다

현재 engine build 경로는 `trtexec`에 `--noDataTransfers`, `--iterations=100`, 필요 시 `--builderOptimizationLevel`, FP16일 때 `--fp16`을 준다. 논문 본문만으로는 아래가 현재와 정확히 같은지 확인되지 않는다.

- TensorRT / JetPack / CUDA / cuDNN 버전
- builder optimization level
- tactic source와 workspace 설정
- FP32 / FP16 사용 여부
- 입력 shape와 batch
- engine rebuild 시점의 tactic 선택
- 클럭 고정, fan mode, thermal 상태

사용한 장비가 둘 다 AGX Orin 64GB여도, 이 조건들이 다르면 10~수십 % 차이는 충분히 난다. 다만 지금 관측되는 큰 차이의 1순위 원인은 여전히 측정 정의 차이다.

## 현재 workspace 숫자를 어떻게 해석해야 하나

현재 workspace 안에서도 숫자 계층을 구분해야 한다.

1. `K=1` all-zero mask 측정값  
   실제 non-split에 가장 가까운 현재 workspace 값이다.
2. `dag_aligned_full` chunk 합  
   split candidate metadata에 가깝다. non-split runtime으로 보면 안 된다.
3. `trt_fusion_safe` max-K G  
   현재 정책 하에서 허용된 가장 잘게 나눈 measured split이다. 논문 full split과 동일하지 않다.
4. CPU wall-clock chunk 측정값  
   논문 Table IV와 비교하려면 이 축을 써야 한다.

실제로 기존 archived FP32 결과에서도 `dag_aligned_full` 합은 K=1 측정값보다 컸다.

| 모델 | archived `dag_aligned_full` GPU p99 합 (ms) | archived K=1 GPU p99 (ms) |
|---|---:|---:|
| AlexNet | 1.770 | 1.609 |
| ResNet18 | 1.058 | 0.859 |
| VGG19 | 7.560 | 6.231 |

즉 현재 repo에서도 “가장 잘게 쪼갠 base chunk 합”은 “실제 한 덩어리 engine 실행시간”과 같은 값이 아니다.

## 논문과 비슷한 결과를 만들려면

### A. 먼저 비교 목표를 둘로 나눈다

**논문 수치 재현**이 목표라면:

- 논문과 같은 Caffe model을 사용한다.
- Inception은 `inception_v3`가 아니라 `InceptionV4`를 쓴다.
- 논문 Table II / III에 맞는 split point 집합을 직접 구성한다.
- FP32부터 맞춘다.
- CPU wall-clock latency를 기준으로 Table IV를 다시 만든다.

**현재 시스템의 공정한 대응 비교**가 목표라면:

- 현재 torchvision/ONNX 모델은 유지한다.
- 대신 논문식 측정 정의만 가져온다.
- 즉 GPU max가 아니라 runtime-level CPU wall-clock WCET로 non-split / split을 다시 잰다.
- 이 경우 “논문과 숫자가 같아지는 것”보다 “왜 차이가 나는지 분해 가능한 것”이 목표다.

### B. 측정 코드를 논문식으로 바꾼다

현재 `chunk_pipeline.cpp`의 `cpu_max_ms`는 `execute()` 호출 전후만 재서, 논문이 말한 전체 runtime wall-clock과 완전히 같다고 보기 어렵다. 논문과 더 비슷하게 하려면 다음이 필요하다.

1. 스케줄러가 chunk 요청을 받는 지점부터 완료를 확인하는 지점까지 wall-clock을 잰다.
2. 각 chunk 완료 시 synchronization을 포함한다.
3. resource manager 경로를 포함한다.
4. non-split도 같은 runtime 경로로 잰다.
5. 각 설정을 충분히 반복하고 max 또는 논문과 동일한 WCET 산정 방식을 쓴다.

이렇게 하면 full split 합이 커지는 방향이 자연스럽게 드러날 가능성이 높다.

### C. 실험 조건을 고정한다

AGX Orin에서 최소한 아래는 로그로 남겨야 한다.

- `nvpmodel` 모드
- `jetson_clocks` 적용 여부
- JetPack / TensorRT / CUDA / cuDNN 버전
- precision
- batch size / input shape
- builder optimization level
- engine 파일 생성 시점과 해시
- warmup / iteration 수
- 온도와 throttling 여부

논문은 maximum performance mode를 썼다고 했으므로, 현재도 그 조건을 명시적으로 맞춰야 한다.

## 추천 실험 순서

1. **현재 모델 그대로** K=1과 `trt_fusion_safe` max-K에 대해 GPU timing과 CPU wall-clock timing을 둘 다 뽑는다.
2. 두 시간의 차이를 모델별로 표로 만든다.  
   여기서 full split에서 CPU/GPU gap이 크게 벌어지면, 논문과의 주된 차이가 실제로 dispatch/sync overhead임을 확인할 수 있다.
3. 그 다음 논문 split 개수와 비슷한 정책을 새로 만들어, 현재 모델에서만 split granularity 효과를 분리한다.
4. 마지막으로 정말 논문 수치에 접근하고 싶다면 Caffe model + 논문 split points + CPU wall-clock runtime 경로로 별도 재현 실험을 한다.

## 실무적으로 가장 중요한 판단

지금 연구 목적이 schedulability analysis라면, **논문 숫자와 비슷해 보이는 것**보다 **RTA에 넣는 시간이 어떤 의미인지 일관적인 것**이 더 중요하다.

- 논문식 RTA를 따를 거면 CPU wall-clock WCET를 넣는다.
- 현재 GPU-side measured-K 기반 분석을 유지할 거면, 논문 Table IV와 직접 수치 비교하지 않는다.
- 둘을 모두 보여줄 거면 표를 분리해서 `GPU execution time`과 `runtime-observed WCET`라고 명확히 이름 붙인다.

현재 workspace 기준으로는 이 셋을 섞지 않는 것이 가장 중요하다.

- `K=1 measured timing`
- `dag_aligned_full` metadata
- `trt_fusion_safe` max-K measured timing

이 구분만 지켜도, 논문과 현재 결과가 왜 다르게 보이는지는 훨씬 덜 수상해진다.
