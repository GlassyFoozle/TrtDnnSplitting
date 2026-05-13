# metadata-only live CandidateSpace 구현 리포트

작성일: 2026-05-12

## 목표

`major_blocks` live 실험에서 `dag_aligned_full` metadata는 계속 필수로 사용하되, `results/table4/<model>_cpp_dag_aligned_full_fp32.json` 같은 canonical singleton/base chunk profiling JSON이 없어도 `run_tests.sh`가 시작될 수 있게 했다.

핵심 원칙은 다음이다.

- `dag_aligned_full.json`은 boundary universe, mask length, interval config 생성을 위해 계속 필요하다.
- `dag_aligned_full` singleton timing은 live `major_blocks` 실험의 prerequisite가 아니다.
- RTA/schedulability는 placeholder timing을 절대 사용하지 않는다.
- K=1은 `dag_aligned_full` sum이 아니라 all-zero mask의 single merged interval `int_0_(N-1)` measured `gpu_max_ms`를 사용한다.

## 수정 내용

### 1. CandidateSpace live placeholder 허용

`src/optimization/candidate_space.py`에 `allow_missing_timing_for_live` 옵션을 추가했다.

동작:

- `artifacts/split_configs/<model>/dag_aligned_full.json`은 여전히 필수다.
- canonical timing source가 있으면 기존처럼 사용한다.
- timing source가 없고 `allow_missing_timing_for_live=True`이면 `chunk_gpu_*_ms`에 positive placeholder `1.0`을 넣는다.
- 이 CandidateSpace에는 `timing_is_placeholder=True`, `timing_source="live_placeholder_missing_base_timing"`을 기록한다.
- `allow_missing_timing_for_live=False`이면 기존처럼 fatal error를 유지한다.

positive placeholder를 쓴 이유는 UNI 변환 코드가 `base_block_list`에서 `<=0` block을 버리기 때문이다. `1.0`은 GPU block 구조와 boundary index를 보존하기 위한 값이며, 실제 timing으로 쓰면 안 된다.

### 2. Loader/runner live path 연결

다음 경로에 옵션을 전달했다.

- `run_dnn_rta_algorithm(..., dry_run=False)`
  → `generate_dnn_taskset(..., allow_missing_base_timing_for_live=True)`
- `generate_dnn_taskset()`
  → `load_dnn_taskset()`
- `load_dnn_taskset()`
  → `load_candidate_space(..., allow_missing_timing_for_live=True)`

dry-run mode에서는 이 옵션을 켜지 않는다. 따라서 정확한 timing source가 없으면 기존처럼 error가 나거나, 명시적으로 `--allow-equal-wcet-fallback`을 준 경우에만 fallback을 사용한다.

### 3. Placeholder guard

`DNNBackedTask`에 다음 상태를 추가했다.

- `base_timing_placeholder`
- `current_timing_measured`

`dnnsplitting_adapter`는 이 상태를 `InferenceSegment`에도 복사한다.

RTA 진입 전 guard:

- `_apply_no_split_measured_to_all()`는 모든 task에 K=1 measured timing을 적용한다.
- patch 성공 시 `current_timing_measured=True`가 된다.
- placeholder task가 measured timing으로 patch되지 않은 상태면 fatal error로 중단한다.

즉 live placeholder는 mask length와 boundary structure를 만들기 위한 bootstrap 값일 뿐, RTA가 소비할 수 없다.

### 4. K=1 semantics 명확화

K=1은 다음 경로를 사용한다.

```text
apply_no_split_mask()
  -> evaluate_and_apply_mask(mask=[0, ..., 0])
  -> evaluate_mask()
  -> artifacts/chunk_cache/<model>/int_0_(N-1)/chunk_fp32.engine
  -> per_chunk_gpu_max_ms[0]
```

따라서 K=1 schedulability check는 `dag_aligned_full` singleton sum이 아니다. all-zero mask로 merge된 단일 TensorRT interval의 measured max timing이다.

### 5. split 후보 evaluation 순서

`evaluate_and_apply_mask()`의 cache 순서를 보강했다.

1. exact mask evaluation JSON은 `evaluate_mask()` early cache hit에서 처리한다.
2. exact mask JSON이 없고 필요한 interval timing이 모두 있으면 `assemble_from_intervals()`로 cache-only result를 만든다.
3. 둘 다 없으면 기존처럼 live export/build/profile로 간다.

이로써 `major_blocks` 실험에서 실제로 필요한 interval만 build/profile되고, 필요하지 않은 모든 singleton `int_i_i` timing을 요구하지 않는다.

### 6. run_tests.sh smoke 실행 옵션

`run_tests.sh`에 smoke/debug용 환경변수 override를 추가했다.

- `CONFIGS_OVERRIDE`
- `NUM_TASKSETS_OVERRIDE`
- `UTILIZATIONS`
- `ALGORITHM_SET`
- `ALGORITHMS`
- `SKIP_PLOTS`

기본 4-model full run은 그대로 가능하다.

## vit_l_16 policy

`vit_l_16 major_blocks`는 유지했다.

```json
"major_blocks": [0, 6, 12, 18, 24]
```

모든 encoder block boundary는 `transformer_blocks` 정책에 따로 남아 있다.

## 검증

### 1. 전체 테스트

```bash
conda run --no-capture-output -n trt python -m pytest tests/ -q
```

결과:

```text
68 passed
```

### 2. dag_aligned_full profiling JSON 없이 load_candidate_space 통과 확인

현재 `results/table4/*dag_aligned_full*`가 없는 상태에서 확인했다.

```text
alexnet   N=22 placeholder=True live_placeholder_missing_base_timing
resnet18  N=14 placeholder=True live_placeholder_missing_base_timing
vgg19     N=46 placeholder=True live_placeholder_missing_base_timing
vit_l_16  N=26 placeholder=True live_placeholder_missing_base_timing
```

기본 모드에서는 기존처럼 error가 유지된다.

```text
Missing dag_aligned_full profiling data for '<model>' (fp32).
```

### 3. run_tests.sh smoke

실행:

```bash
CONFIGS_OVERRIDE='1_base.yaml' \
NUM_TASKSETS_OVERRIDE=1 \
UTILIZATIONS='0.70' \
ALGORITHMS='ss:single:SS-single-smoke' \
RUN_LABEL='smoke_c_metadata_live_skipplot' \
SKIP_PLOTS=1 \
./run_tests.sh
```

결과:

- 4개 모델 taskset 로딩 성공
- K=1 all-zero mask cache hit 확인
- real profile 0, cache hit 16
- result rows with errors: 0
- policy violations: 0

## 실행 권장

full8 major_blocks 실험은 이제 canonical `dag_aligned_full` singleton profiling 없이 시작할 수 있다.

```bash
./run_tests.sh
```

짧은 smoke:

```bash
CONFIGS_OVERRIDE='1_base.yaml' \
NUM_TASKSETS_OVERRIDE=1 \
UTILIZATIONS='0.70' \
ALGORITHMS='ss:single:SS-single-smoke' \
RUN_LABEL='smoke_4models' \
SKIP_PLOTS=1 \
./run_tests.sh
```

주의: full8에서 split이 실제로 필요해지는 taskset은 major_blocks interval들을 live로 export/build/profile할 수 있다. 이 동작이 이번 C 방식의 의도된 동작이다.
