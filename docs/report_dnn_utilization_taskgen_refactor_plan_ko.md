# DNN Utilization 중심 Taskset 생성 리팩토링 계획

작성일: 2026-05-13

## 결론

현재 이해는 맞다. `scripts/30_run_yaml_fig4_experiment.py`의 YAML 기반 Fig.4 workflow는 YAML config를 읽고, 이를 `WorkloadConfig`로 매핑한 뒤 `src/integration/dnn_workload_generator.py`에서 taskset JSON을 생성하고, 생성된 taskset별로 RTA 알고리즘을 실행해 CSV/summary/plot 입력을 만든다.

현재 taskset 생성의 utilization 의미는 "total utilization", 즉 각 task의 `(GPU time + CPU time) / period` 합이다. 반면 새 요구사항은 `configs/yaml/260511_configs/9_base_1.25.yaml`처럼 `dnn_utilization_range`를 실험 x축으로 삼고, 각 task의 `GPU time / period` 합이 이 값이 되도록 period를 먼저 정해야 한다. 이후 `C_ratio_range`에서 CPU/GPU 시간 비율을 샘플링해 CPU 시간을 만들고, 기존처럼 pre/post CPU segment로 랜덤 분할하면 된다.

따라서 리팩토링의 핵심은 `U_i`의 의미를 total utilization에서 DNN/GPU utilization으로 바꾸는 새 생성 경로를 추가하는 것이다. `31_plot_fig4.py`의 x축 라벨도 이 경로에서는 `Total utilization U`가 아니라 `GPU utilization U`가 되어야 한다.

## 현재 Workflow

대상 엔트리포인트는 `scripts/30_run_yaml_fig4_experiment.py`다.

1. `parse_simple_yaml()`이 flat YAML을 파싱한다.
2. `build_mapping()`이 YAML 값을 내부 mapping으로 변환한다.
3. `generate_yaml_tasksets()`가 mapping으로 `WorkloadConfig`를 만들고 `generate_tasksets()`를 호출한다.
4. 생성된 taskset JSON에 대해 `run_dnn_rta_algorithm()`을 알고리즘별로 실행한다.
5. `per_taskset_results.csv`, `schedulability_ratio.csv`, `split_activity.csv`, `summary.md`, `yaml_mapping_report.md`를 쓴다.
6. `scripts/31_plot_fig4.py`는 `schedulability_ratio.csv`의 `utilization` column을 x축으로 사용한다.

관련 코드:

- `scripts/30_run_yaml_fig4_experiment.py:355`: 현재 `utilization_list()`는 `utilization_range`/`utilization_step`만 읽는다.
- `scripts/30_run_yaml_fig4_experiment.py:389`: 현재 mapping은 `G_ratio_range`를 읽는다.
- `scripts/30_run_yaml_fig4_experiment.py:445`: 현재 mapping의 `utilization_basis`는 `"total"`로 고정된다.
- `scripts/30_run_yaml_fig4_experiment.py:477`: `WorkloadConfig` 생성 시 `utilization_basis="total"`로 고정된다.
- `scripts/31_plot_fig4.py:23`: x축 라벨은 `Total utilization U`로 고정되어 있다.

## 현재 Taskset 생성 Logic

실제 생성기는 `src/integration/dnn_workload_generator.py`다.

`dnnsplitting` mode에서 현재 흐름은 다음과 같다.

1. K=1 real DNN GPU WCET `G_i`를 모델별 profiling/evaluation JSON 또는 dry-run fallback에서 가져온다.
2. 각 taskset마다 CPU 수와 CPU별 task 수를 YAML range에서 샘플링한다.
3. 전체 target utilization `U`를 CPU별로 나눈다.
   - `uniform_cpu_utilization=true`: 각 CPU가 `U / num_cpus`.
   - `false`: UUniFast로 CPU별 utilization을 샘플링한다.
4. 각 CPU의 utilization을 해당 CPU의 task들에 나눈다.
   - `uniform_task_utilization=true`: 균등 분배.
   - `false`: UUniFast로 task별 `U_i`를 샘플링한다.
5. 각 task에 대해 모델을 샘플링하고, fixed GPU WCET `G_i`를 가져온다.
6. `G_ratio_range`에서 `sampled_g_ratio`를 샘플링한다.
7. `utilization_basis == "total"`이므로 다음 식으로 total execution budget과 period를 역산한다.

```text
total_budget_i = G_i / sampled_g_ratio
C_i            = total_budget_i - G_i
T_i = D_i      = total_budget_i / U_i
```

8. `C_i`는 `_split_cpu_budget()`에서 random 비율로 `cpu_pre_ms`와 `cpu_post_ms`로 나뉜다.
9. deadline-monotonic 순서로 priority를 부여한다.
10. metadata에 actual GPU/CPU/total utilization과 actual G ratio를 기록한다.

관련 코드:

- `src/integration/dnn_workload_generator.py:544`: total utilization을 CPU별로 분배.
- `src/integration/dnn_workload_generator.py:558`: CPU별 utilization을 task별로 분배.
- `src/integration/dnn_workload_generator.py:570`: `G_ratio_range` 샘플링.
- `src/integration/dnn_workload_generator.py:576`: total-utilization 기반 공식.
- `src/integration/dnn_workload_generator.py:604`: CPU budget을 pre/post로 분할.
- `src/integration/dnn_workload_generator.py:641`: actual GPU/CPU/total utilization 계산.

즉 현재 구조에서 `U_i`는 task의 total utilization이고, 실제 GPU utilization은 대략 `sum(U_i * sampled_g_ratio_i)`가 된다. 이 때문에 YAML의 `utilization_range`가 0.7이라도 실제 GPU utilization은 `G_ratio` 평균에 따라 더 작아진다.

## 새 YAML의 의미

새 config `configs/yaml/260511_configs/9_base_1.25.yaml`에는 다음 키가 있다.

```yaml
dnn_utilization_range: [0.5, 1.0]
dnn_utilization_step: 0.1
C_ratio_range: [0.0, 0.5]
```

이 요구사항에서 `U_i`는 task의 DNN/GPU utilization이다.

새 생성 공식은 다음이 되어야 한다.

```text
u_gpu_i = UUniFast(...), sum(u_gpu_i) = selected_dnn_utilization
G_i     = real model GPU WCET
T_i     = D_i = G_i / u_gpu_i

c_ratio_i = uniform(C_ratio_range)
C_i       = G_i * c_ratio_i
cpu_pre_i, cpu_post_i = random split of C_i
```

이때 actual utilization은 다음 관계를 가진다.

```text
gpu_util_i   = G_i / T_i = u_gpu_i
cpu_util_i   = C_i / T_i = c_ratio_i * u_gpu_i
total_util_i = (G_i + C_i) / T_i = (1 + c_ratio_i) * u_gpu_i
```

따라서 `C_ratio_range: [0.0, 0.5]`이면 task별 total utilization은 GPU utilization의 1.0배에서 1.5배 사이가 된다. config 이름 `9_base_1.25.yaml`은 평균 `C_ratio=0.25`일 때 total utilization이 GPU utilization의 약 1.25배가 되는 의도로 보인다.

기존 `G_ratio`와 연결하면 다음과 같다.

```text
G_ratio = G_i / (G_i + C_i) = 1 / (1 + c_ratio_i)
```

하지만 `C_ratio`를 균등 샘플링하는 것과 `G_ratio`를 균등 샘플링하는 것은 분포가 다르다. 따라서 `C_ratio_range`를 단순히 `G_ratio_range = [1/(1+C_max), 1/(1+C_min)]`로 변환해 기존 코드에 넣는 방식은 요구사항과 정확히 일치하지 않는다. 새 code path에서 `C_ratio` 자체를 샘플링해야 한다.

## 변경 계획

### 1. YAML mapping 확장

파일: `scripts/30_run_yaml_fig4_experiment.py`

변경 내용:

- `utilization_list()`를 일반화해 다음 두 모드를 지원한다.
  - 기존 모드: `utilization_range`/`utilization_step`
  - 새 모드: `dnn_utilization_range`/`dnn_utilization_step`
- `build_mapping()`에서 `dnn_utilization_range` 존재 여부로 generation basis를 판별한다.
- 새 모드에서는 `C_ratio_range`를 필수 또는 명시적 default로 파싱한다.
- mapping에 다음 필드를 추가한다.
  - `utilization_basis: "gpu"`
  - `utilization_kind: "dnn_gpu"`
  - `c_ratio_min`, `c_ratio_max`
  - `utilization_label: "GPU utilization U"`
- 기존 config와의 호환을 위해 `utilization_range` 기반 config는 지금처럼 `utilization_basis: "total"`과 `G_ratio_range`를 유지한다.

권장 validation:

- `dnn_utilization_range`와 `utilization_range`가 동시에 있으면 명시적으로 error 처리한다.
- `dnn_utilization_step`은 양수여야 한다.
- `C_ratio_range`는 `0 <= min <= max`여야 한다.
- 새 모드에서는 `G_ratio_range`가 없어도 정상 동작해야 한다.

### 2. WorkloadConfig 확장

파일: `src/integration/dnn_workload_generator.py`

변경 내용:

- `WorkloadConfig`에 다음 필드를 추가한다.

```python
c_ratio_range: Optional[Tuple[float, float]] = None
utilization_kind: str = "total"
```

- docstring에서 기존 `dnnsplitting total` 경로와 새 `dnn/gpu utilization` 경로의 공식을 분리해 설명한다.

### 3. Generator에 DNN utilization 경로 추가

파일: `src/integration/dnn_workload_generator.py`

선호 구현:

- 기존 `_generate_dnnsplitting_taskset()` 내부의 `utilization_basis == "gpu"` branch를 `c_ratio_range` 기반으로 명확히 바꾼다.
- 또는 더 명확하게 `taskgen_mode == "dnnsplitting_dnn_utilization"`를 추가한다.

권장 방식은 기존 함수의 CPU/task 분배 구조를 재사용하되, task별 budget 계산만 분기하는 것이다. 이유는 CPU 수, task 수, uniform/UUniFast 옵션, priority 부여, metadata 출력 구조가 거의 동일하기 때문이다.

새 branch pseudo-code:

```python
if config.utilization_basis == "gpu" and config.c_ratio_range is not None:
    T_i = G_i / u_i
    sampled_c_ratio = rng.uniform(c_min, c_max)
    cpu_total = G_i * sampled_c_ratio
    actual_c_ratio = sampled_c_ratio
    actual_g_ratio = 1.0 / (1.0 + sampled_c_ratio)
```

기존 total-utilization branch는 그대로 유지한다.

주의점:

- 현재 `utilization_basis == "gpu"` branch는 `T_i = G_i / u_i`를 쓰지만, CPU 시간은 여전히 `G_ratio_range`로부터 `G_i / sampled_g_ratio - G_i`를 계산한다. 새 요구사항은 `C_ratio_range`를 균등 샘플링해야 하므로 이 branch를 그대로 쓰면 안 된다.
- `period_range`는 현재 default로 무시된다. 새 formula에서도 real DNN WCET이 ms 단위라 `period_range: [1000, 10000]`를 validity filter로 강제하면 taskset이 많이 reject될 수 있다. 기존 `--ignore-period-range` default는 유지하는 편이 맞다.

### 4. Metadata와 diagnostics 보강

파일:

- `src/integration/dnn_workload_generator.py`
- `scripts/30_run_yaml_fig4_experiment.py`

추가할 task-level metadata:

- `target_gpu_utilization`
- `sampled_c_ratio`
- `actual_c_ratio`
- 기존 호환용 `target_utilization`은 유지하되, 새 모드에서는 의미가 GPU utilization임을 notes에 남긴다.

추가할 taskset-level metadata:

- `_utilization_kind: "dnn_gpu"`
- `_selected_dnn_utilization`
- `_c_ratio_range`
- `_actual_c_ratio: {min, max, avg}`
- `_actual_gpu_partition_utilization` 가능하면 추가
- `_generation_formula`를 새 공식으로 갱신

`taskset_diagnostics()`와 `taskset_diag_by_util()`에는 `_actual_c_ratio`를 읽는 logic을 추가한다. `yaml_mapping_report.md`, `summary.md`, `schedulability_ratio.csv`에는 최소한 다음을 보여주는 것이 좋다.

- `avg_gpu_util`
- `avg_cpu_util`
- `avg_total_util`
- `avg_actual_c_ratio`
- 기존 호환용 `avg_actual_g_ratio`

### 5. CSV x축 의미 정리

파일: `scripts/30_run_yaml_fig4_experiment.py`

현재 CSV의 `utilization` column은 plot x축으로 쓰인다. 새 모드에서는 이 값이 selected DNN/GPU utilization이다. 하위 호환을 위해 column 이름은 유지하되, 추가 column을 넣는 방식이 가장 안전하다.

권장 추가 column:

- `utilization_kind`: `total` 또는 `dnn_gpu`
- `target_gpu_utilization`
- `target_total_utilization`은 새 모드에서는 없거나 실제 평균 total과 구분해야 하므로, 혼동을 피하려면 `avg_total_util`만 유지한다.

기존 plot/script 호환성 때문에 `schedulability_ratio.csv`의 `utilization`은 계속 유지한다.

### 6. Plot label 변경

파일: `scripts/31_plot_fig4.py`

최소 변경:

```python
_X_AXIS_LABEL = "GPU utilization U"
```

더 안전한 변경:

- `--x-axis-label` CLI 옵션을 추가한다.
- `--run-dir`가 주어진 경우 sibling `run_config.json`의 `mapped_values.utilization_kind`를 읽어서 자동으로 라벨을 선택한다.
  - `dnn_gpu`면 `GPU utilization U`
  - 기존 total mode면 `Total utilization U`
- 사용자가 `--x-axis-label`을 주면 그 값을 우선한다.

사용자가 요청한 새 Fig.4 기준으로만 보면 최소 변경도 충분하지만, 기존 total-utilization 결과를 다시 plot할 가능성이 있으면 자동 선택 방식이 더 안전하다.

### 7. Test 계획

파일: `tests/test_taskgen.py`

추가할 단위 테스트:

1. `dnn_utilization` mode에서 `_actual_gpu_utilization`이 target에 가깝거나 거의 동일한지 확인.
2. task별 `sampled_c_ratio`가 `C_ratio_range` 안에 있는지 확인.
3. task별 `period_ms == real_gpu_wcet_ms / target_gpu_utilization` 관계를 허용 오차 내에서 확인.
4. `_actual_total_utilization >= _actual_gpu_utilization`이고, `C_ratio_range=[0.0, 0.5]`이면 대략 `total <= 1.5 * gpu`인지 확인.
5. 기존 `utilization_range` + `G_ratio_range` config가 이전과 동일하게 total-utilization mode로 동작하는지 regression test.

간단 smoke command:

```bash
python scripts/30_run_yaml_fig4_experiment.py \
  --config configs/yaml/260511_configs/9_base_1.25.yaml \
  --dry-run \
  --num-tasksets-override 2 \
  --utilizations 0.5 0.6 \
  --algorithm-set main4 \
  --run-name dnn_util_taskgen_smoke
```

확인할 output:

- `generated_tasksets/u0p50/taskset_*.json`
- `_actual_gpu_utilization`이 0.5 근처인지
- `_actual_total_utilization`이 GPU utilization보다 큰지
- task-level `sampled_c_ratio`가 `[0.0, 0.5]` 안인지
- `schedulability_ratio.csv`의 `utilization`이 GPU utilization x축 값인지

## 구현 순서

1. `WorkloadConfig`에 `c_ratio_range`와 `utilization_kind`를 추가한다.
2. `dnn_workload_generator.py`에서 `utilization_basis="gpu" + c_ratio_range` 경로를 구현한다.
3. `30_run_yaml_fig4_experiment.py`의 YAML mapping을 `dnn_utilization_range`/`C_ratio_range` aware하게 바꾼다.
4. diagnostics/report/CSV에 `utilization_kind`와 `actual_c_ratio`를 추가한다.
5. `31_plot_fig4.py`의 x축 라벨을 `GPU utilization U`로 바꾸거나 run_config 기반 자동 선택을 추가한다.
6. unit test와 smoke run으로 기존 config와 새 config를 모두 확인한다.

## Open Questions

1. `C_ratio_range`는 현재 요구 설명상 `C / G`로 이해했다. 즉 `C_ratio=0.5`면 CPU 총 시간이 GPU 시간의 50%다. 이 해석이 맞으면 위 공식 그대로 구현하면 된다.
2. 기존 total-utilization 실험을 앞으로도 plot해야 한다면 `31_plot_fig4.py`는 자동 라벨 선택 방식이 좋다. 새 Fig.4만 유지할 계획이면 상수 라벨만 바꿔도 된다.
3. 새 YAML의 `period_range`는 기존과 같이 기본 무시가 맞아 보인다. real DNN WCET 기반 생성에서는 `T=G/u_gpu`로 이미 결정되므로, `period_range`는 선택적 validity filter로만 유지하는 것이 일관적이다.
