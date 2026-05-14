# DNN Utilization 기반 Taskset 생성 구현 보고서

작성일: 2026-05-13

## 변경 요약

`30_run_yaml_fig4_experiment.py`의 기존 total-utilization 기반 taskset 생성 경로는 유지하면서, `dnn_utilization_range`와 `C_ratio_range`를 사용하는 GPU-utilization 기반 경로를 추가했다.

새 경로의 task별 생성 공식은 다음과 같다.

```text
u_gpu_i = UUniFast(...), sum(u_gpu_i) = selected dnn_utilization
G_i     = real model GPU WCET
T_i     = D_i = G_i / u_gpu_i
c_i     = uniform(C_ratio_range)
C_i     = G_i * c_i
cpu_pre_i, cpu_post_i = random split of C_i
```

따라서 metadata의 `_actual_gpu_utilization`은 YAML의 selected `dnn_utilization`과 맞고, `_actual_total_utilization`은 CPU ratio만큼 더 커진다.

## 수정 파일

### `src/integration/dnn_workload_generator.py`

- `WorkloadConfig`에 `c_ratio_range`, `utilization_kind`를 추가했다.
- 기존 `_generate_dnnsplitting_taskset()` 내부에 `utilization_basis == "gpu" and c_ratio_range is not None` branch를 추가했다.
- 새 branch는 `G_ratio_range`로 CPU 시간을 역산하지 않고, `C_ratio_range`에서 직접 `C/G`를 샘플링한다.
- task-level metadata를 추가했다.
  - `target_gpu_utilization`
  - `sampled_c_ratio`
  - `actual_c_ratio`
- taskset-level metadata를 추가했다.
  - `_utilization_kind`
  - `_selected_dnn_utilization`
  - `_c_ratio_range`
  - `_actual_c_ratio`
  - `_actual_gpu_partition_utilization`
- 기존 total-utilization + `G_ratio_range` 경로는 그대로 유지했다.

### `scripts/30_run_yaml_fig4_experiment.py`

- YAML mapping에서 `dnn_utilization_range`/`dnn_utilization_step`를 인식하도록 확장했다.
- `dnn_utilization_range`가 있으면 mapping은 다음처럼 설정된다.
  - `utilization_basis = "gpu"`
  - `utilization_kind = "dnn_gpu"`
  - `c_ratio_min/max = C_ratio_range`
- `utilization_range`와 `dnn_utilization_range`가 동시에 있으면 error 처리한다.
- diagnostics, `per_taskset_results.csv`, `schedulability_ratio.csv`, `yaml_mapping_report.md`, `summary.md`에 `utilization_kind`와 C-ratio diagnostics를 추가했다.
- `run_config.json`의 taskset path serialization을 repo-relative 실패 시 absolute path로 fallback하게 했다.

### `scripts/31_plot_fig4.py`

- `--x-axis-label` override 옵션을 추가했다.
- `--run-dir`의 `run_config.json`에서 `mapped_values.utilization_kind == "dnn_gpu"`이면 x축 라벨을 자동으로 `GPU utilization U`로 바꾼다.
- 기존 total-utilization run은 기본값 `Total utilization U`를 유지한다.

### `run_tests.sh`

- 기본 `CONFIG_DIR`를 `configs/yaml/gpu_util_configs`로 바꿨다.
- `CONFIG_DIR` 환경변수로 기존 config dir를 override할 수 있게 했다.
- 기본 config 목록의 2~4번을 새 C-ratio 실험 이름으로 바꿨다.

### `tests/test_taskgen.py`

- GPU-utilization mode 단위 테스트를 추가했다.
- 확인 항목:
  - `_actual_gpu_utilization`이 target GPU utilization과 일치
  - task별 `sampled_c_ratio`가 `C_ratio_range` 안에 있음
  - `period_ms == real_gpu_wcet_ms / target_gpu_utilization`
  - total utilization이 GPU utilization보다 크고 C-ratio bound를 넘지 않음

## 새 YAML 세트

새 디렉터리: `configs/yaml/gpu_util_configs`

생성한 파일:

- `1_base.yaml`: `C_ratio_range: [0.0, 0.5]`, 8 CPUs, tasks per CPU `[1, 3]`
- `2_C_ratio_00.yaml`: `C_ratio_range: [0.0, 0.0]`
- `3_C_ratio_25.yaml`: `C_ratio_range: [0.25, 0.25]`
- `4_C_ratio_50.yaml`: `C_ratio_range: [0.5, 0.5]`
- `5_task1.yaml`: tasks per CPU `[1, 1]`
- `6_task3.yaml`: tasks per CPU `[3, 3]`
- `7_singleCPU_task4.yaml`: 1 CPU, 4 tasks
- `8_singleCPU_task8.yaml`: 1 CPU, 8 tasks

공통 설정:

```yaml
dnn_utilization_range: [0.5, 1.0]
dnn_utilization_step: 0.1
period_range: [1000, 10000]
number_of_inference_segments_range: [1, 1]
max_block_count_range: [20, 20]
G_utilization_threshold_range: [1.0, 1.0]
per_splitting_overhead: 5
uniform_cpu_utilization: false
uniform_task_utilization: false
```

## 검증 결과

실행한 검사:

```bash
python -m py_compile scripts/30_run_yaml_fig4_experiment.py scripts/31_plot_fig4.py src/integration/dnn_workload_generator.py
pytest -q tests/test_taskgen.py
```

결과:

```text
5 passed
```

추가 smoke:

```bash
python scripts/30_run_yaml_fig4_experiment.py \
  --config configs/yaml/gpu_util_configs/1_base.yaml \
  --models alexnet resnet18 \
  --dry-run \
  --num-tasksets-override 1 \
  --utilizations 0.5 \
  --algorithms ss:tol-fb:SS-tol-fb \
  --run-name gpu_util_smoke \
  --allow-equal-wcet-fallback
```

확인한 생성값:

```text
_utilization_kind: dnn_gpu
_actual_gpu_utilization: 0.5
_actual_cpu_utilization: 0.144549
_actual_total_utilization: 0.644549
_c_ratio_range: [0.0, 0.5]
_actual_c_ratio avg: 0.266935
```

smoke는 CSV/report 작성까지 완료했다. 로컬 profiling cache가 부족한 개발 환경이라 알고리즘 결과 row에는 analysis error가 기록될 수 있지만, taskset 생성과 새 utilization/C-ratio metadata 경로는 정상 동작함을 확인했다.

## 실행 방법

기본값 그대로 실행하면 새 GPU-utilization config 세트를 사용한다.

```bash
./run_tests.sh
```

기존 total-utilization config를 다시 쓰려면 다음처럼 override할 수 있다.

```bash
CONFIG_DIR=configs/yaml/260511_configs ./run_tests.sh
```
