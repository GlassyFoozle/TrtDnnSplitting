# measured K split best-mask cache

## 목적

measured K-search는 같은 모델과 같은 split policy에서 K별 후보 mask를 모두 평가해서 가장 balanced한 measured configuration을 고른다. TensorRT profile 자체는 cache hit가 나더라도, 후보 mask를 매번 전부 순회하면 design-time overhead가 크다.

이를 줄이기 위해 K-search 결과를 persistent cache로 저장하도록 수정했다.

## 저장 위치

```text
results/optimization/measured_k_split_cache.json
```

## cache key

다음 조건이 같으면 이전에 찾은 best mask를 재사용한다.

- model name
- precision
- wcet metric
- split policy
- boundary count
- enabled boundary set
- K

즉 `resnet18 / fp32 / p99 / major_blocks / K=7`에서 한 번 찾은 best mask는 이후 같은 조건에서 바로 로드된다.

## 동작 방식

`src/integration/mask_applicator.py::apply_k_chunks()`에서 처리한다.

1. cache key를 만든다.
2. `measured_k_split_cache.json`에 best mask가 있으면 그 mask 하나만 `evaluate_and_apply_mask()`로 적용한다.
3. cache가 없거나 cached mask 적용이 실패하면 기존처럼 모든 K 후보를 평가한다.
4. best measured mask를 찾으면 JSON cache에 저장한다.

cache hit 후에도 `evaluate_and_apply_mask()`는 호출한다. 따라서 실제 timing은 기존 TensorRT evaluation cache에서 검증/로드되고, task에는 measured chunk timing이 patch된다.

## force / dry-run 처리

- `force=True`이면 best-mask cache를 우회한다.
- `dry_run=True`이면 measured timing이 없으므로 best-mask cache를 쓰지 않는다.

## best mask score

best mask는 다음 tuple을 최소화하는 방식으로 고른다.

```text
(max_chunk, total_gpu, spread)
```

즉 우선순위는 `max_chunk -> total_gpu -> spread`다.

## 검증

테스트 추가:

```text
tests/test_measured_k_split.py::test_apply_k_chunks_reuses_persistent_best_mask_cache
```

확인한 동작:

- 첫 실행: 모든 K 후보 mask를 평가하고 best mask 저장
- 두 번째 실행: 저장된 best mask 하나만 평가

검증 결과:

```text
pytest -q tests/test_measured_k_split.py tests/test_k1_accounting.py tests/test_uni_conversion_alignment.py tests/test_split_policy.py
20 passed
```
