# Finest chunk WCET 합 기반 chunk 길이 예측 실험

## 실험 방법

- precision은 `fp16`이고, WCET은 각 JSON의 `gpu_max_ms`를 사용했다.
- finest granularity 기준은 각 모델의 `dag_aligned_full` profile이다.
- 특정 splitting configuration의 chunk 예측값은 해당 chunk가 포함하는 `source_chunk_ids`의 finest chunk WCET 합으로 계산했다.
- random configuration 선택은 seed `20260516`로 고정했다. 후보는 이미 측정된 table4 JSON 중 k=1..5를 사용했다.

## 데이터 가용성

- `alexnet`: finest profile JSON이 없어 제외함.
- `resnet18`: finest profile JSON이 없어 제외함.
- `vgg19`: finest profile JSON이 없어 제외함.
- `mobilenet_v3_small`: finest profile JSON이 없어 제외함.
- `inception_v3`: finest profile JSON이 없어 제외함.
- `vit_b_16`: finest profile JSON이 없어 제외함.
- `vit_l_16`: finest profile JSON이 없어 제외함.

## 해석

finest chunk WCET 합은 split으로 인한 TensorRT fusion 변화, kernel launch 구성 변화, 중간 텐서 materialization 비용을 반영하지 않는다. 따라서 실제 chunk WCET과 차이가 날 수 있으며, 위 표의 오차는 그 차이를 정량화한 값이다.
