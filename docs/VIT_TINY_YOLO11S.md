# ViT-Tiny 및 YOLO11s split 지원

이 문서는 새 장치에서 아래 두 모델을 기존 실험 경로에 그대로 넣어 실행할 때 필요한
모델 키, 전체 boundary index, `trt_fusion_safe` allowlist와 grouping 규칙을 정리한다.

- `vit_tiny`: 입력 `(1,3,224,224)`, 출력 `(1,1000)`, 14 chunks
- `yolo11s`: 입력 `(1,3,640,640)`, 출력 `(1,84,8400)`, 24 chunks

두 모델은 `src/models/registry.py`에 등록되어 있고, baseline split metadata는
`artifacts/split_configs/<model>/dag_aligned_full.json`, policy는
`configs/split_point_policies.json`에 versioning된다. 장치별 ONNX, TensorRT engine,
측정 timing과 실험 결과는 Git에 포함하지 않으며 첫 실행에서 자동 생성한다.

## 바로 실행하기

repository root에서 다음과 같이 실행한다.

```bash
CONFIGS_OVERRIDE=1_base.yaml \
MODELS_OVERRIDE="vit_tiny yolo11s" \
UTILIZATIONS_OVERRIDE=0.5 \
NUM_TASKSETS=1 \
RUN_LABEL=smoke_vit_tiny_yolo11s \
./run_hayeonp_no_inflation.sh
```

기존 canonical 여섯 모델의 default 목록은 변경하지 않았다. 새 두 모델은
`MODELS_OVERRIDE`로 명시할 때만 실행된다. 별도 timm/Ultralytics 설치나 weight
download는 필요 없다.

## Boundary 표기법

`boundary k`는 `chunk k`와 `chunk k+1` 사이를 뜻한다. `all`은 구조상 열거 가능한
모든 boundary이고, `trt_fusion_safe`는 그중 TensorRT에 넘길 복합 연산과 route
stage 내부를 쪼개지 않는 allowlist다. allowlist에 없는 boundary bit는 항상 0으로
고정된다.

## ViT-Tiny/16 (224)

Chunk 0은 patch projection, class token, position embedding을 하나로 묶는다.
Chunk 1–12는 transformer encoder block 0–11이고, chunk 13은 final LayerNorm,
class-token select, classifier head다. Multi-head attention, MLP, residual 및
LayerNorm 내부는 분할하지 않는다.

| Boundary | 왼쪽 chunk → 오른쪽 chunk | `trt_fusion_safe` |
|---:|---|:---:|
| 0 | token preparation → encoder block 0 | yes |
| 1 | encoder block 0 → 1 | yes |
| 2 | encoder block 1 → 2 | yes |
| 3 | encoder block 2 → 3 | yes |
| 4 | encoder block 3 → 4 | yes |
| 5 | encoder block 4 → 5 | yes |
| 6 | encoder block 5 → 6 | yes |
| 7 | encoder block 6 → 7 | yes |
| 8 | encoder block 7 → 8 | yes |
| 9 | encoder block 8 → 9 | yes |
| 10 | encoder block 9 → 10 | yes |
| 11 | encoder block 10 → 11 | yes |
| 12 | encoder block 11 → final norm/head | yes |

```text
all             = [0,1,2,3,4,5,6,7,8,9,10,11,12]
trt_fusion_safe = [0,1,2,3,4,5,6,7,8,9,10,11,12]
```

따라서 최대 허용 분할은 14개의 singleton group이다. 어떤 boundary bit를 끄면
서로 인접한 encoder-block group이 하나의 ONNX/TensorRT interval로 병합된다.

## YOLO11s

YOLO는 backbone의 P3/P4/P5 feature와 PAN neck의 skip route를 동시에 유지해야 하는
DAG다. 기존 runtime의 single-input/single-output 계약을 유지하기 위해 각 boundary의
live feature들을 정해진 순서로 flatten/concatenate한 하나의 packed state tensor로
전달한다. 각 chunk는 필요한 feature를 unpack하고 갱신한 뒤 다시 pack한다. 이 방식은
route를 제거하거나 근사하지 않으며 전체 PyTorch graph와 chunk chain의 출력을 동일하게
유지한다.

| Boundary | 왼쪽 chunk → 오른쪽 chunk | `trt_fusion_safe` | 이유 |
|---:|---|:---:|---|
| 0 | model.0 Conv → model.1 Conv | yes | top-level block boundary |
| 1 | model.1 Conv → model.2 C3k2 | yes | top-level block boundary |
| 2 | model.2 C3k2 → model.3 Conv | yes | top-level block boundary |
| 3 | model.3 Conv → model.4 C3k2 | yes | top-level block boundary |
| 4 | model.4 C3k2 (P3) → model.5 Conv | yes | backbone stage exit |
| 5 | model.5 Conv → model.6 C3k2 | yes | top-level block boundary |
| 6 | model.6 C3k2 (P4) → model.7 Conv | yes | backbone stage exit |
| 7 | model.7 Conv → model.8 C3k2 | yes | top-level block boundary |
| 8 | model.8 C3k2 → model.9 SPPF | yes | top-level block boundary |
| 9 | model.9 SPPF → model.10 C2PSA | yes | top-level block boundary |
| 10 | model.10 C2PSA (P5) → model.11 Upsample | yes | backbone/neck handoff |
| 11 | model.11 Upsample → model.12 Concat | no | route stage 내부 |
| 12 | model.12 Concat → model.13 C3k2 | no | route stage 내부 |
| 13 | model.13 C3k2 → model.14 Upsample | yes | atomic route stage exit |
| 14 | model.14 Upsample → model.15 Concat | no | route stage 내부 |
| 15 | model.15 Concat → model.16 C3k2 | no | route stage 내부 |
| 16 | model.16 C3k2 (Detect P3) → model.17 Conv | yes | atomic route stage exit |
| 17 | model.17 Conv → model.18 Concat | no | route stage 내부 |
| 18 | model.18 Concat → model.19 C3k2 | no | route stage 내부 |
| 19 | model.19 C3k2 (Detect P4) → model.20 Conv | yes | atomic route stage exit |
| 20 | model.20 Conv → model.21 Concat | no | route stage 내부 |
| 21 | model.21 Concat → model.22 C3k2 | no | route stage 내부 |
| 22 | model.22 C3k2 (Detect P5) → model.23 Detect | yes | Detect handoff |

```text
all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
trt_fusion_safe = [0,1,2,3,4,5,6,7,8,9,10,13,16,19,22]
```

최대 허용 분할에서 실제 grouping은 다음 16개다.

```text
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
[11,12,13] [14,15,16] [17,18,19] [20,21,22] [23]
```

즉 Upsample/또는 downsample Conv → Concat → C3k2로 이루어진 네 route stage를
각각 하나의 TensorRT interval로 유지한다. C3k2, SPPF, C2PSA, Detect의 내부
operator boundary도 애초에 top-level chunk 후보로 노출하지 않는다.

## 재현 범위와 weight 의미

두 constructor는 기존 repository 모델들과 동일하게 네트워크 접근 없이
deterministic하게 초기화된다. 따라서 이 지원의 목적은 해당 architecture의 split,
TensorRT timing 및 schedulability 실험이며 pretrained semantic accuracy 비교가 아니다.
`yolo11s` 출력은 NMS 전 Detect+DFL decode 결과이고 NMS는 포함하지 않는다.

새 장치는 자체 CUDA/TensorRT 조합으로 engine과 timing을 다시 만들어야 한다.
다른 장치에서 생성한 engine이나 timing을 복사해 재사용하는 것은 지원하지 않는다.
