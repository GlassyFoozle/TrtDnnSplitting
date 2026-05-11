# task generation range sampling / K=1 timing 수정

## 내 판단

`period` 역산 자체는 버그라고 보지 않는다. real DNN에서는 GPU 실행시간 `G`가 모델/profile에 의해 고정되므로, target utilization과 `G_ratio`를 맞추려면 `T`를 역산하는 방식이 자연스럽다. 다만 이전 simulation과 같은 분포는 아니므로, plot/report에는 이 차이를 명시해야 한다.

`per_splitting_overhead`도 실제 TRT profile을 사용하는 실험에서는 별도로 더하지 않는 것이 맞다. split으로 인해 발생하는 overhead는 export/build된 실제 engine profile에 포함된다. 단, 이전 simulation은 synthetic overhead `5`를 명시적으로 더하는 모델이므로 simulation과 TRT의 절대 ratio가 달라질 수는 있다.

반대로 YAML range lower-bound mapping과 K=1 timing은 수정해야 하는 코드 이슈였다.

## 수정 내용

### 1. YAML integer range random sampling

수정 파일:

- `scripts/30_run_yaml_fig4_experiment.py`
- `src/integration/dnn_workload_generator.py`

이전에는 `number_of_tasks_per_cpu_range: [1, 3]` 같은 YAML range가 lower bound `1`로 고정됐다. 이제 range를 generator까지 전달하고 taskset 생성 시 random sampling한다.

현재 동작:

- `number_of_cpu_range`: taskset마다 `randint(lo, hi)`
- `number_of_tasks_per_cpu_range`: CPU마다 `randint(lo, hi)`
- `number_of_inference_segments_range`: taskset마다 sample 후 metadata 기록
- `max_block_count_range`: taskset마다 sample 후 metadata 기록

주의: real-DNN TRT 경로에서는 actual split candidate count가 DNN graph/policy에서 정해지므로 `max_block_count_range`가 ResNet18 chunk 수를 직접 바꾸지는 않는다. 하지만 YAML semantic을 보존하기 위해 sampled value는 taskset metadata에 남긴다.

### 2. task generation K=1 timing

수정 파일:

- `src/integration/dnn_workload_generator.py`
- `scripts/15_compare_k1_timing_semantics.py`

이전에는 workload generation의 `real_G`가 `dag_aligned_full` per-chunk timing 합산을 먼저 사용했다. 이 값은 fully split metadata이지 measured K=1 timing이 아니다.

이제 `_get_base_gpu_wcet_ms()`는 다음 순서로 동작한다.

1. all-zero mask K=1 evaluation JSON을 읽는다.
2. live YAML run에서는 K=1 evaluation이 없으면 `evaluate_mask(mask=[0,...,0])`로 profile/cache 생성을 시도한다.
3. dry-run/reference 상황에서만 `_DRY_RUN_BASE_WCET_MS`를 사용한다.

`dag_aligned_full` chunk sum은 더 이상 workload execution time 산정 fallback으로 쓰지 않는다.

## 확인 결과

ResNet18 fp32 기준:

| source | value |
| --- | ---: |
| measured K=1 all-zero | `0.858990 ms` |
| dag_aligned_full chunk sum | `1.057966 ms` |

smoke check에서 config 1의 `[1,3]` task range가 실제로 CPU별 random count로 생성되는 것도 확인했다.

예:

```text
taskset_000: cpu_task_counts=[1, 1, 1, 3, 1, 3, 3, 3], n_tasks=16, real_G=0.85899
taskset_001: cpu_task_counts=[1, 3, 2, 1, 1, 1, 1, 3], n_tasks=13, real_G=0.85899
taskset_002: cpu_task_counts=[1, 2, 1, 3, 3, 1, 3, 1], n_tasks=15, real_G=0.85899
```

## 남은 해석 이슈

이 수정 후에는 이전보다 simulation과 TRT의 양상이 더 가까워질 가능성이 있다. 특히 config 1/5/6은 task count randomization 때문에 기존 TRT 저장 결과보다 더 어려워질 수 있다.

그래도 완전히 같은 ratio를 기대하면 안 된다. 이유는 다음이다.

- TRT는 DNN별 실제 measured timing을 쓰고, simulation은 synthetic integer block timing을 쓴다.
- TRT split timing은 additive/monotonic하지 않다.
- split overhead는 simulation에서는 고정 상수, TRT에서는 measured engine 결과에 내재된다.
- 현재 저장된 `fig4_resnet_*` 결과는 이번 수정 전 결과라 재사용하면 안 된다.

따라서 config 1~8은 이번 수정 후 다시 돌려야 한다.
