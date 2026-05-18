# 현재 apply-K cache와 full-split WCET 정리

## 기준

- apply-K curve는 현재 `results/optimization/measured_k_split_cache.json`에 실제로 남아 있는 엔트리만 사용했다.
- 없는 K는 새로 프로파일링하거나 보간하지 않았다.
- curve 출력은 아래 파일로 다시 생성했다.
  - `results/best_k_split_curves/fp16_trt_fusion_safe_cache/`
  - `results/best_k_split_curves/fp32_trt_fusion_safe_cache/`
- 아래 WCET 표는 `fp32`, CPU wall-time, `trt_fusion_safe` full split 기준이다.
- wall-time 측정 구간은 각 chunk의 `execute()` 직전부터 `cudaStreamSynchronize()` 완료 직후까지다.

## 현재 apply-K cache 커버리지

| precision | model | available K |
|---|---|---|
| fp16 | alexnet | 2-11 |
| fp16 | inception_v3 | 2-7, 17 |
| fp16 | mobilenet_v3_small | 2-16 |
| fp16 | resnet18 | 11 |
| fp16 | vgg19 | 2-12 |
| fp16 | vit_b_16 | 2-14 |
| fp32 | alexnet | 11 |
| fp32 | inception_v3 | 2, 3, 17 |
| fp32 | mobilenet_v3_small | 2, 16 |
| fp32 | resnet18 | 11 |
| fp32 | vgg19 | 2-12 |
| fp32 | vit_b_16 | 2-9, 14 |

## Non-split / full-split WCET

| model | config | chunks | per-chunk WCET (ms) | sum of chunk WCETs (ms) | measured total wall max (ms) |
|---|---|---:|---|---:|---:|
| alexnet | non-split | 1 | 2.298 | 2.298 | 2.298 |
| alexnet | full-split | 11 | 0.369, 0.113, 0.286, 0.126, 0.313, 0.298, 0.244, 0.125, 1.428, 0.767, 0.296 | 4.364 | 3.536 |
| resnet18 | non-split | 1 | 1.682 | 1.682 | 1.682 |
| resnet18 | full-split | 11 | 0.455, 0.226, 0.232, 0.378, 0.351, 0.393, 0.413, 0.501, 0.444, 0.261, 0.102 | 3.757 | 3.063 |
| vgg19 | non-split | 1 | 9.806 | 9.806 | 9.806 |
| vgg19 | full-split | 12 | 1.830, 1.078, 1.072, 1.259, 1.077, 1.190, 0.778, 0.798, 0.134, 2.886, 0.712, 0.394 | 13.208 | 12.689 |

## 해석 메모

- `sum of chunk WCETs`는 각 chunk의 worst case를 따로 더한 값이라, schedulability 분석에서 보수적으로 쓰기 좋은 값이다.
- `measured total wall max`는 한 번의 전체 실행에서 관찰된 end-to-end max다. 각 chunk의 개별 worst case가 같은 반복에서 동시에 나오지 않으므로, 보통 `sum of chunk WCETs`보다 작다.
- 현재 fp32 apply-K cache는 `alexnet`과 `resnet18`에서 full split 점만 남아 있다. 따라서 해당 curve는 현재 cache 기준으로는 한 점짜리다.
