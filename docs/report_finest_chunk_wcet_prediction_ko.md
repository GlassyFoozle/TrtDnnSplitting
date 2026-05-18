# Finest chunk WCET 합 기반 chunk 길이 예측 실험

## 실험 방법

- precision은 `fp32`이고, WCET은 각 JSON의 `gpu_p99_ms`를 사용했다.
- finest granularity 기준은 각 모델의 `dag_aligned_full` profile이다.
- 특정 splitting configuration의 chunk 예측값은 해당 chunk가 포함하는 `source_chunk_ids`의 finest chunk WCET 합으로 계산했다.
- random configuration 선택은 seed `20260516`로 고정했다. 후보는 이미 측정된 table4 JSON 중 k=1..5를 사용했다.

## 데이터 가용성

- `mobilenet_v3_small`: finest profile JSON이 없어 제외함.
- `inception_v3`: finest profile JSON이 없어 제외함.
- `vit_b_16`: finest profile JSON이 없어 제외함.
- `vit_l_16`: finest profile JSON이 없어 제외함.

## alexnet

- finest chunk 수: 22
- finest profile: `results_0512_no_vit/table4/alexnet_cpp_dag_aligned_full_fp32.json`

### no-split: `alexnet_mask_b1565b4d_k1`

- 측정 JSON: `results_0512_no_vit/table4/alexnet_cpp_alexnet_mask_b1565b4d_k1_fp32.json`
- split config: `artifacts/split_configs/alexnet/alexnet_mask_b1565b4d_k1.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..21 (22개) | 1.608689 | 1.769933 | -0.161244 | -9.11% |
| **합계** | - | **1.608689** | **1.769933** | **-0.161244** | **-9.11%** |

### 2 split: `alexnet_mask_539fd571_k2`

- 측정 JSON: `results_0512_no_vit/table4/alexnet_cpp_alexnet_mask_539fd571_k2_fp32.json`
- split config: `artifacts/split_configs/alexnet/alexnet_mask_539fd571_k2.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..7 (8개) | 0.200478 | 0.266019 | -0.065541 | -24.64% |
| 1 | 8..21 (14개) | 1.456090 | 1.503915 | -0.047825 | -3.18% |
| **합계** | - | **1.656568** | **1.769933** | **-0.113366** | **-6.41%** |

### 3 split: `alexnet_mask_8a6facf5_k3`

- 측정 JSON: `results_0512_no_vit/table4/alexnet_cpp_alexnet_mask_8a6facf5_k3_fp32.json`
- split config: `artifacts/split_configs/alexnet/alexnet_mask_8a6facf5_k3.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..5 (6개) | 0.168131 | 0.208291 | -0.040160 | -19.28% |
| 1 | 6..7 (2개) | 0.049568 | 0.057728 | -0.008160 | -14.13% |
| 2 | 8..21 (14개) | 1.457348 | 1.503915 | -0.046566 | -3.10% |
| **합계** | - | **1.675048** | **1.769933** | **-0.094886** | **-5.36%** |

### 4 split: `alexnet_mask_911c9c9c_k4`

- 측정 JSON: `results_0512_no_vit/table4/alexnet_cpp_alexnet_mask_911c9c9c_k4_fp32.json`
- split config: `artifacts/split_configs/alexnet/alexnet_mask_911c9c9c_k4.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..2 (3개) | 0.104851 | 0.118466 | -0.013616 | -11.49% |
| 1 | 3..7 (5개) | 0.112416 | 0.147553 | -0.035136 | -23.81% |
| 2 | 8..17 (10개) | 0.958348 | 1.003972 | -0.045625 | -4.54% |
| 3 | 18..21 (4개) | 0.484119 | 0.499942 | -0.015823 | -3.16% |
| **합계** | - | **1.659734** | **1.769933** | **-0.110200** | **-6.23%** |

### 5 split: `alexnet_mask_db240274_k5`

- 측정 JSON: `results_0512_no_vit/table4/alexnet_cpp_alexnet_mask_db240274_k5_fp32.json`
- split config: `artifacts/split_configs/alexnet/alexnet_mask_db240274_k5.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..5 (6개) | 0.166998 | 0.208291 | -0.041292 | -19.82% |
| 1 | 6..14 (9개) | 0.143937 | 0.192354 | -0.048418 | -25.17% |
| 2 | 15..17 (3개) | 0.851564 | 0.869346 | -0.017781 | -2.05% |
| 3 | 18..20 (3개) | 0.387749 | 0.398822 | -0.011072 | -2.78% |
| 4 | 21 | 0.100353 | 0.101121 | -0.000767 | -0.76% |
| **합계** | - | **1.650602** | **1.769933** | **-0.119331** | **-6.74%** |

## resnet18

- finest chunk 수: 14
- finest profile: `results_0512_no_vit/table4/resnet18_cpp_dag_aligned_full_fp32.json`

### no-split: `resnet18_mask_af53aaf8_k1`

- 측정 JSON: `results_0512_no_vit/table4/resnet18_cpp_resnet18_mask_af53aaf8_k1_fp32.json`
- split config: `artifacts/split_configs/resnet18/resnet18_mask_af53aaf8_k1.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..13 (14개) | 0.858990 | 1.057966 | -0.198976 | -18.81% |
| **합계** | - | **0.858990** | **1.057966** | **-0.198976** | **-18.81%** |

### 2 split: `resnet18_mask_6d06790a_k2`

- 측정 JSON: `results_0512_no_vit/table4/resnet18_cpp_resnet18_mask_6d06790a_k2_fp32.json`
- split config: `artifacts/split_configs/resnet18/resnet18_mask_6d06790a_k2.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..11 (12개) | 0.835746 | 1.029902 | -0.194156 | -18.85% |
| 1 | 12..13 (2개) | 0.025536 | 0.028064 | -0.002528 | -9.01% |
| **합계** | - | **0.861283** | **1.057966** | **-0.196684** | **-18.59%** |

### 3 split: `resnet18_mask_18e328d0_k3`

- 측정 JSON: `results_0512_no_vit/table4/resnet18_cpp_resnet18_mask_18e328d0_k3_fp32.json`
- split config: `artifacts/split_configs/resnet18/resnet18_mask_18e328d0_k3.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..3 (4개) | 0.147150 | 0.265572 | -0.118423 | -44.59% |
| 1 | 4..11 (8개) | 0.696674 | 0.764329 | -0.067656 | -8.85% |
| 2 | 12..13 (2개) | 0.025506 | 0.028064 | -0.002559 | -9.12% |
| **합계** | - | **0.869329** | **1.057966** | **-0.188637** | **-17.83%** |

### 4 split: `resnet18_mask_abeeb268_k4`

- 측정 JSON: `results_0512_no_vit/table4/resnet18_cpp_resnet18_mask_abeeb268_k4_fp32.json`
- split config: `artifacts/split_configs/resnet18/resnet18_mask_abeeb268_k4.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..3 (4개) | 0.149542 | 0.265572 | -0.116031 | -43.69% |
| 1 | 4..5 (2개) | 0.148288 | 0.153506 | -0.005218 | -3.40% |
| 2 | 6..7 (2개) | 0.146722 | 0.155808 | -0.009086 | -5.83% |
| 3 | 8..13 (6개) | 0.466370 | 0.483080 | -0.016709 | -3.46% |
| **합계** | - | **0.910922** | **1.057966** | **-0.147044** | **-13.90%** |

### 5 split: `resnet18_mask_2995fada_k5`

- 측정 JSON: `results_0512_no_vit/table4/resnet18_cpp_resnet18_mask_2995fada_k5_fp32.json`
- split config: `artifacts/split_configs/resnet18/resnet18_mask_2995fada_k5.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..3 (4개) | 0.147151 | 0.265572 | -0.118421 | -44.59% |
| 1 | 4..5 (2개) | 0.147328 | 0.153506 | -0.006177 | -4.02% |
| 2 | 6..11 (6개) | 0.569666 | 0.610824 | -0.041158 | -6.74% |
| 3 | 12 | 0.007297 | 0.007712 | -0.000416 | -5.39% |
| 4 | 13 | 0.019168 | 0.020352 | -0.001184 | -5.82% |
| **합계** | - | **0.890610** | **1.057966** | **-0.167356** | **-15.82%** |

## vgg19

- finest chunk 수: 46
- finest profile: `results_0512_no_vit/table4/vgg19_cpp_dag_aligned_full_fp32.json`

### no-split: `vgg19_mask_e412b169_k1`

- 측정 JSON: `results_0512_no_vit/table4/vgg19_cpp_vgg19_mask_e412b169_k1_fp32.json`
- split config: `artifacts/split_configs/vgg19/vgg19_mask_e412b169_k1.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..45 (46개) | 6.231463 | 7.560377 | -1.328913 | -17.58% |
| **합계** | - | **6.231463** | **7.560377** | **-1.328913** | **-17.58%** |

### 2 split: `vgg19_mask_b09863df_k2`

- 측정 JSON: `results_0512_no_vit/table4/vgg19_cpp_vgg19_mask_b09863df_k2_fp32.json`
- split config: `artifacts/split_configs/vgg19/vgg19_mask_b09863df_k2.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..27 (28개) | 3.019715 | 4.246962 | -1.227247 | -28.90% |
| 1 | 28..45 (18개) | 3.210889 | 3.313415 | -0.102526 | -3.09% |
| **합계** | - | **6.230604** | **7.560377** | **-1.329773** | **-17.59%** |

### 3 split: `vgg19_mask_5ddd5221_k3`

- 측정 JSON: `results_0512_no_vit/table4/vgg19_cpp_vgg19_mask_5ddd5221_k3_fp32.json`
- split config: `artifacts/split_configs/vgg19/vgg19_mask_5ddd5221_k3.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..18 (19개) | 2.093743 | 3.195983 | -1.102240 | -34.49% |
| 1 | 19..36 (18개) | 1.351683 | 1.527716 | -0.176033 | -11.52% |
| 2 | 37..45 (9개) | 2.796065 | 2.836678 | -0.040614 | -1.43% |
| **합계** | - | **6.241490** | **7.560377** | **-1.318886** | **-17.44%** |

### 4 split: `vgg19_mask_dcafab35_k4`

- 측정 JSON: `results_0512_no_vit/table4/vgg19_cpp_vgg19_mask_dcafab35_k4_fp32.json`
- split config: `artifacts/split_configs/vgg19/vgg19_mask_dcafab35_k4.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..4 (5개) | 0.611809 | 1.083935 | -0.472126 | -43.56% |
| 1 | 5..9 (5개) | 0.549762 | 0.783749 | -0.233988 | -29.85% |
| 2 | 10..37 (28개) | 2.340842 | 2.862093 | -0.521252 | -18.21% |
| 3 | 38..45 (8개) | 2.817798 | 2.830598 | -0.012800 | -0.45% |
| **합계** | - | **6.320210** | **7.560377** | **-1.240166** | **-16.40%** |

### 5 split: `vgg19_mask_84567e93_k5`

- 측정 JSON: `results_0512_no_vit/table4/vgg19_cpp_vgg19_mask_84567e93_k5_fp32.json`
- split config: `artifacts/split_configs/vgg19/vgg19_mask_84567e93_k5.json`

| chunk | finest source chunk | 실제 WCET p99 (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |
|---:|---|---:|---:|---:|---:|
| 0 | 0..4 (5개) | 0.628215 | 1.083935 | -0.455720 | -42.04% |
| 1 | 5..9 (5개) | 0.550048 | 0.783749 | -0.233701 | -29.82% |
| 2 | 10..27 (18개) | 1.935631 | 2.379277 | -0.443645 | -18.65% |
| 3 | 28..37 (10개) | 0.420098 | 0.482817 | -0.062719 | -12.99% |
| 4 | 38..45 (8개) | 2.910275 | 2.830598 | 0.079676 | +2.81% |
| **합계** | - | **6.444267** | **7.560377** | **-1.116110** | **-14.76%** |

## 해석

finest chunk WCET 합은 split으로 인한 TensorRT fusion 변화, kernel launch 구성 변화, 중간 텐서 materialization 비용을 반영하지 않는다. 따라서 실제 chunk WCET과 차이가 날 수 있으며, 위 표의 오차는 그 차이를 정량화한 값이다.
