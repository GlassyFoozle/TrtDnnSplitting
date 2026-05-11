# 이전 시뮬레이션 vs 현재 TRT 실험 차이 분석

## 결론

현재 `results/dnn_experiments/fig4_resnet_*` 결과는 이전 `prev/schedulability_ratio`와 직접 비교하기 어렵다. RTA 수식 자체는 거의 동일하지만, taskset generation과 TRT timing 입력이 이전 시뮬레이션과 다르게 동작한다. 그래서 schedulability ratio와 method 간 상대 순서/상관관계가 달라지는 것이 자연스럽다.

특히 현재 저장된 TRT 결과가 거의 schedulable하게 나오는 가장 큰 이유는 다음이다.

1. YAML의 `period_range: [1000, 10000]`이 무시되고, 실제 ResNet18 GPU WCET에서 period가 역산된다.
2. `[1, 3]` 같은 YAML range가 랜덤 샘플링되지 않고 lower bound로 고정된다.
3. 현재 run은 `models=['resnet18']` 단일 모델만 사용한다.
4. simulation의 `per_splitting_overhead=5`는 TRT RTA adapter에서 실행시간 overhead로 적용되지 않는다.
5. current generator의 base GPU WCET는 아직 `dag_aligned_full` chunk 합산을 사용한다. 그런데 cache에는 실제 K=1 all-zero TRT timing이 따로 있고, ResNet18 기준 `dag_aligned_full sum=1.057966 ms`, actual K=1 measured `0.858990 ms`다.

따라서 현재 결과는 “ResNet18 단일 모델, real-DNN period 역산, period range 무시, range lower-bound mapping, measured split cache 사용” 조건의 결과로는 의미가 있지만, 이전 simulation config 1~8과 같은 실험이라고 결론내리면 안 된다.

## 관측된 차이

대표적으로 U=0.9에서 이전 simulation과 현재 TRT 결과가 크게 다르다.

| config | method | prev ratio | TRT ratio |
| --- | --- | ---: | ---: |
| 4 `G_ratio=1.0` | SS-heu | 0.01 | 0.74 |
| 4 `G_ratio=1.0` | SS-tol-fb | 0.02 | 0.96 |
| 4 `G_ratio=1.0` | UNI-heu | 0.08 | 0.98 |
| 4 `G_ratio=1.0` | UNI-tol-fb | 0.12 | 0.74 |
| 8 `singleCPU_task8` | SS-heu | 0.00 | 0.98 |
| 8 `singleCPU_task8` | SS-tol-fb | 0.17 | 1.00 |
| 8 `singleCPU_task8` | UNI-heu | 0.20 | 0.94 |
| 8 `singleCPU_task8` | UNI-tol-fb | 0.32 | 1.00 |

현재 TRT config 8 U=0.9의 generated taskset은 period가 대체로 매우 작다.

- p10/p50/p90/p99/max period: `5.323 / 15.947 / 102.633 / 871.557 / 2888.542 ms`
- YAML의 `[1000, 10000]` period 분포와 전혀 다르다.
- 모델은 전부 `resnet18`이고, task마다 `G ~= 1.057966 ms`로 고정된다.

## RTA analysis 비교

`prev/analysis.py`와 `src/rta/analysis.py`의 핵심 함수는 동일하다.

- `get_SS_R_req`, `get_SS_R_job`, `get_SS_R`
- `get_UNI_R_and_K`, `get_UNI_tolerance`
- `RTA_SS_tol_fb`, `RTA_UNI_tol_fb`
- `RTA_SS_heu/opt`, `RTA_UNI_heu/opt`

차이는 import 경로와 이전 simulation에 있던 SS R 선택 통계 helper 제거 정도다. RTA 수식이나 interference 계산이 크게 바뀐 흔적은 없다.

CPU별 interference도 의도대로 보인다. `get_SS_R_req()`는 higher-priority CPU execution interference를 `task_h.cpu != task_i.cpu`이면 skip한다. 반면 GPU blocking/interference는 CPU와 무관하게 global resource 기준으로 계산한다. 이는 이전 코드와 같은 구조다.

따라서 schedulability ratio 차이의 1차 원인은 RTA core가 아니라 RTA에 들어가는 taskset과 timing 분포다.

## generate_task_set 로직 차이

### 1. UUniFast

이전:

- `prev/generate_task_set.py::UUniFast()`
- utilization을 만든 뒤 shuffle하지 않는다.

현재:

- `src/integration/dnn_workload_generator.py::uunifast()`
- 마지막에 `rng.shuffle(utils)`를 수행한다.

분포 자체는 같은 simplex 분포지만, seed와 CPU/task 배치 재현성은 이전과 달라질 수 있다. 큰 schedulability 차이의 주원인은 아니다.

### 2. period 생성 방식

이전 simulation:

```text
T = random.randint(period_range)
G_plus_C = target_U * T
G_ratio = random.uniform(G_ratio_range)
G = round(G_plus_C * G_ratio)
C = G_plus_C - G
```

현재 TRT `dnnsplitting` mode:

```text
real_G = TensorRT model WCET
G_ratio = random.uniform(G_ratio_range)
total_exec = real_G / G_ratio
C = total_exec - real_G
T = total_exec / target_U
```

즉 이전은 `T`가 실험 입력이고 `G/C`가 따라오지만, 현재는 `G`가 모델에서 고정되고 `T`가 역산된다. 이 차이 때문에 deadline/period 분포가 완전히 달라진다.

현재 YAML runner는 기본적으로 `--ignore-period-range=True`라서 `period_range: [1000,10000]`을 사용하지 않는다. 이건 이전 simulation과 가장 큰 semantic 차이다.

### 3. YAML range 처리

이전:

- `number_of_tasks_per_cpu_range: [1, 3]`이면 CPU마다 `random.randint(1, 3)`을 수행한다.
- `number_of_cpu_range`, `number_of_inference_segments_range`, `max_block_count_range`도 같은 원리다.

현재:

- `scripts/30_run_yaml_fig4_experiment.py::int_range_choice()`가 range의 lower bound를 선택한다.
- 예: `[1, 3] -> 1`

그 결과 현재 `config 1_base`는 “CPU당 task 수 1~3 랜덤”이 아니라 “항상 CPU당 task 1개”가 된다. 실제로 `fig4_resnet_1_base`는 taskset당 8 tasks이고, `fig4_resnet_5_task1`과 거의 같은 구조가 된다.

이건 config 1/5/6의 의미를 바꾸는 실질적인 issue다.

### 4. max_block_count / inference segment

이전:

- `G`를 integer budget으로 만든 뒤 `split_int(G, n_segments)`로 segment별 GPU budget을 나눈다.
- `max_block_count`도 `G_segment`와 YAML range에 의해 정해진다.

현재:

- DNN graph의 candidate split point 수가 모델별로 고정된다.
- ResNet18은 현재 `N=14` base chunks다.
- YAML의 `max_block_count_range: [20,20]`은 실제 ResNet18 candidate count를 20으로 만들지 못한다.

즉 simulation의 “최대 20개 block”과 TRT의 “ResNet18 DAG-aligned 14 chunks”는 같은 split granularity가 아니다.

### 5. per_splitting_overhead

이전:

- `per_splitting_overhead=5`가 `InferenceSegment._compute_block_list()`에서 split마다 execution time에 더해진다.

현재 TRT adapter:

- `src/integration/dnnsplitting_adapter.py`에서 `per_splitting_overhead=0.0`으로 고정한다.
- 이유는 measured TRT chunk timing이 실제 profile 결과라 별도 synthetic overhead를 더하지 않기 위해서다.

이건 real TRT 실험으로는 합리적일 수 있지만, 이전 simulation과 같은 overhead 모델은 아니다. 특히 simulation에서는 split이 많아질수록 total GPU work가 증가하지만, TRT에서는 measured config에 따라 total time이 오히려 줄 수도 있다.

### 6. CPU pre/post

이전:

- `C_list = split_int(C, number_of_inference_segments + 1)`로 CPU pre/post가 integer budget에서 나뉜다.

현재:

- `_split_cpu_budget(cpu_total)`이 continuous float budget을 random cut으로 pre/post로 나눈다.

1 inference segment 조건에서는 원리는 유사하다. 최근 수정 후 UNI 변환에서도 CPU pre/post는 GPU chunk 수에 포함하지 않고, SS/UNI 변환에서 별도 block source로 유지된다. 이 부분은 현재 차이의 주원인으로 보이지 않는다.

### 7. priority

이전:

- `priority = 1 / T`
- `sort_task_set()`이 큰 priority를 먼저 정렬하므로 짧은 period가 higher priority다.

현재:

- generator가 deadline 순서로 `priority=1,2,...`를 부여한다.
- adapter가 이를 `-priority`로 변환한다.
- 결과적으로 `sort_task_set()`에서는 짧은 deadline task가 higher priority가 된다.

priority 방향은 맞다.

### 8. G_utilization_threshold

이전 코드도 threshold를 random으로 뽑고 log에 남기지만, taskset reject/regenerate 조건으로 쓰지 않는다. 현재도 실질적으로 사용되지 않는다. 이건 두 실험 간 차이 원인은 아니다.

## timing semantic issue

현재 `_get_base_gpu_wcet_ms()`는 task generation의 `real_G`를 만들 때 먼저 `dag_aligned_full` per-chunk p99 합을 사용한다.

ResNet18 fp32 p99 기준:

| timing | value |
| --- | ---: |
| `dag_aligned_full` chunk sum | `1.057966 ms` |
| measured all-zero K=1 TRT config | `0.858990 ms` |

즉 generation은 period를 `1.057966 ms` 기준으로 만들고, 현재 measured-only RTA path는 cache가 있으면 K=1 all-zero 측정값을 쓸 수 있다. 이 상태로 새 실험을 돌리면 taskset의 target utilization과 RTA 실제 execution time이 어긋난다.

이전 “base timing 합산은 어디에서도 쓰면 안 된다”는 원칙을 엄밀히 적용하려면 task generation의 `real_G`도 all-zero K=1 measured timing을 우선해야 한다. `dag_aligned_full` sum은 split candidate search용 metadata로는 쓸 수 있지만, workload utilization을 정하는 execution time으로 쓰면 안 된다.

## 현재 TRT 결과가 거의 schedulable한 이유

1. `period_range`를 무시해서 이전 simulation의 `[1000,10000]` period 분포와 다른 taskset이 생성됐다.
2. config 1의 `[1,3]` tasks-per-CPU range가 lower bound 1로 고정되어 workload가 가벼워졌다.
3. 단일 ResNet18만 사용해서 task별 GPU WCET가 거의 동일하고 작다.
4. split overhead model이 simulation과 다르다.
5. stored result 기준 많은 method가 no-split 또는 cache-hit split만으로 끝난다. 예를 들어 config 1 U=0.9에서 SS-heu/SS-opt/SS-tol-fb는 `split_triggered_pct=0.0`인데 ratio가 `1.0`이다.
6. ratio가 1.0 근처로 saturate되면 method 간 correlation/order가 쉽게 왜곡된다. 이전 simulation처럼 method별 차이가 크게 드러나는 regime이 아니다.

## 타당성 판단

현재 결과는 “현재 코드와 현재 real-DNN mapping 조건에서의 TRT schedulability 결과”로는 해석할 수 있다. 하지만 이전 `prev/schedulability_ratio`와 동일한 config 1~8 실험이라고 보기는 어렵다.

직접 비교를 위해서는 최소한 다음을 맞춰야 한다.

1. YAML range를 lower bound가 아니라 이전처럼 taskset마다 random sampling한다.
2. `period_range`를 무시하지 말고 이전처럼 period를 먼저 sample하는 mode를 별도로 둔다. 단, real DNN은 `G`가 고정이므로 이 경우 target `G_ratio`와 utilization을 동시에 정확히 만족시키기 어렵다. 비교 목적이면 simulation-compatible synthetic timing mode와 real-DNN mode를 분리해서 표시해야 한다.
3. task generation의 K=1 `real_G`는 measured all-zero TRT timing을 사용한다. `dag_aligned_full` sum을 utilization 산정에 쓰지 않는다.
4. config 1~8별 task 수, CPU 수, G ratio, split granularity가 실제로 이전 YAML 의미와 같은지 run report에서 검증한다.
5. 최근 SS<->UNI 변환 및 measured-only K=1 수정 이후 기존 `fig4_resnet_*` 결과는 재사용하지 말고 다시 생성한다.

## 권장 수정 항목

우선순위는 다음과 같다.

1. `scripts/30_run_yaml_fig4_experiment.py::int_range_choice()`를 없애거나, YAML range를 taskset generation 단계까지 전달해서 이전처럼 random sampling하게 바꾼다.
2. `_get_base_gpu_wcet_ms()`가 all-zero K=1 measured result를 `dag_aligned_full` sum보다 먼저 사용하게 바꾼다.
3. report에 `period_range ignored`, `tasks_per_cpu actual distribution`, `model distribution`, `K=1 timing source`를 항상 출력한다.
4. simulation-compatible 비교용이면 `taskgen_mode=synthetic_compat` 같은 별도 mode를 만들고, 이전 `generate_task_set.py`와 동일하게 `T -> C+G -> C/G split -> block split` 순서로 생성한다.
5. real-DNN 결과와 simulation 결과는 plot에서 같은 legend로 직접 겹치지 말고, “synthetic simulation”과 “TRT real-DNN mapped workload”로 분리한다.
