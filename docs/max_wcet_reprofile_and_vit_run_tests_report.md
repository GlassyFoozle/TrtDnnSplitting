# max WCET 전환, interval 재프로파일링, vit_l_16 run_tests 리포트

작성일: 2026-05-12

## 요약

이번 변경은 profiling 결과에서 `p99`를 계속 측정/저장하되, schedulability check와 RTA에서 사용하는 실제 WCET timing source를 `max`로 전환하는 것이다. 기존 `p99` 값은 비교, 리포트, legacy cache 호환을 위해 유지한다.

추가로 interval cache에 있는 기존 TensorRT engine을 다시 실행해서 `gpu_max_ms_*`와 interval별 profiling wall time을 채울 수 있는 `scripts/27_reprofile_cached_intervals.py`를 추가했다. `warmup`과 `iters`가 달라지면 profiling loop 자체의 실행 횟수가 달라지므로 `profile_wall_s_*`도 달라질 수 있다. 따라서 새 스크립트는 `profile_warmup_*`, `profile_iters_*`를 함께 저장한다.

## p99 로직을 max 로 전환한 내용

실제 분석 경로는 이제 다음 규칙을 따른다.

- `wcet_metric="max"`: `per_chunk_gpu_max_ms` / `chunk_gpu_max_ms` 사용
- `wcet_metric="p99"`: 분석 경로에서는 deprecated alias로 취급하고 max 사용
- `wcet_metric="mean"`: 개발/비교용 optimistic path로만 mean 사용

주요 수정 파일:

- `src/integration/mask_applicator.py`: mask 적용 시 선택하는 chunk timing을 max로 변경
- `src/integration/dnn_taskset_loader.py`: taskset 로딩 시 기본 WCET metric을 max로 변경
- `src/integration/dnn_taskset_generator.py`: overlay evaluation timing을 max 우선으로 변경
- `src/integration/dnn_workload_generator.py`: K=1 WCET lookup을 max 기준으로 변경
- `src/integration/dnn_algorithm_runner.py`: 알고리즘 기본 `wcet_metric`을 max로 변경
- `src/optimization/candidate_space.py`: `chunk_gpu_max_ms`를 CandidateSpace에 추가하고, table4에 max가 없을 때 singleton interval cache timing(`int_i_i/timing.json`)도 읽도록 보완
- `src/optimization/config_evaluator.py`: interval assembly/cache validity를 max timing 존재 기준으로 강화

`p99` 필드는 삭제하지 않았다. C++ runner, Python parser, profiling DB, JSON result에는 mean/p99/max가 함께 남는다.

## compile/profile 시간 기록 보완

기존 구현 상태:

- ONNX export 시간은 interval별 `timing.json`에 `export_wall_s`로 저장된다.
- TensorRT engine build 시간은 precision별 `build_fp32_wall_s` 또는 `build_fp16_wall_s`로 저장된다.
- mask 전체 profile 시간은 evaluation result의 `profile_wall_s`로 저장되고 design-time 통계에 합산된다.

이번 보완:

- 일반 mask 평가에서 interval timing을 backfill할 때, 해당 mask profile 전체 wall time은 `source_eval_profile_wall_s_<precision>`로 남긴다. 이 값은 여러 interval을 한 번에 profile한 wall time이므로 interval 단독 wall time으로 해석하면 안 된다.
- 새 `scripts/27_reprofile_cached_intervals.py`는 interval engine 하나를 1-chunk config로 따로 실행하므로, 여기서 저장하는 `profile_wall_s_<precision>`가 interval별 profile wall time이다.
- 새 스크립트는 함께 사용한 `profile_warmup_<precision>`, `profile_iters_<precision>`도 저장한다.

interval timing 예시:

```json
{
  "gpu_mean_ms_fp32": 0.12,
  "gpu_p99_ms_fp32": 0.15,
  "gpu_max_ms_fp32": 0.18,
  "profile_wall_s_fp32": 3.42,
  "profile_warmup_fp32": 20,
  "profile_iters_fp32": 200
}
```

## max 재프로파일링 스크립트

추가 파일:

- `scripts/27_reprofile_cached_intervals.py`

동작:

- `artifacts/chunk_cache/<model>/int_<start>_<end>/chunk_<precision>.engine`를 스캔한다.
- ONNX export와 engine build는 하지 않는다.
- 기존 engine을 `cpp_runtime/build/table4_runner`로 다시 실행한다.
- interval `timing.json`에 mean/p99/max와 profiling wall time, warmup, iters를 기록한다.

dry-run:

```bash
conda run -n trt python scripts/27_reprofile_cached_intervals.py \
  --models alexnet resnet18 vgg19 \
  --precision fp32 \
  --warmup 20 \
  --iters 200 \
  --dry-run
```

실행:

```bash
conda run --no-capture-output -n trt python scripts/27_reprofile_cached_intervals.py \
  --models alexnet resnet18 vgg19 \
  --precision fp32 \
  --warmup 20 \
  --iters 200 \
  --force
```

`--force`를 빼면 같은 `warmup`/`iters`로 이미 `gpu_max_ms_*`가 있는 interval은 건너뛴다. `vit_l_16`은 아직 cache가 없다면 나중에 live runtime에서 engine이 생긴 뒤 같은 스크립트로 재실행하면 된다.

base chunk timing은 `int_0_0`, `int_1_1` 같은 singleton interval의 `timing.json`에서도 읽을 수 있게 했다. 따라서 기존 table4 JSON이 mean/p99만 가진 상태여도, script 27로 singleton interval들의 max가 채워지면 CandidateSpace가 max timing을 사용할 수 있다.

## vit_l_16 split policy 확인

`vit_l_16`의 `major_blocks`는 모든 encoder block boundary를 켜는 정책이 아니라 coarse split policy로 유지하는 것이 맞다. 따라서 다음으로 수정했다.

```json
"major_blocks": [0, 6, 12, 18, 24]
```

의미:

- 0: patch/class/position prep 뒤
- 6, 12, 18: encoder block 6개 단위 그룹 경계
- 24: final norm/head 앞

모든 encoder block boundary를 쓰고 싶을 때는 별도 정책인 `transformer_blocks`를 사용한다.

## run_tests.sh 에 vit_l_16 포함해서 다시 실행하는 방법

`run_tests.sh`는 기본값을 기존 3개 모델로 유지하면서, 환경변수로 모델 목록과 run label을 바꿀 수 있게 했다.

기존 결과 백업:

```bash
ts=$(date +%Y%m%d_%H%M%S)
mkdir -p "backups/run_tests_${ts}"
cp -a results "backups/run_tests_${ts}/results"
```

chunk cache까지 보존하려면 용량을 확인한 뒤 추가로 백업한다.

```bash
cp -a artifacts/chunk_cache "backups/run_tests_${ts}/chunk_cache"
```

vit 포함 실행:

```bash
MODELS="alexnet resnet18 vgg19 vit_l_16" \
RUN_LABEL="fig4_4models_vit_l_16_${ts}" \
./run_tests.sh
```

이 명령은 각 config에 대해 `scripts/30_run_yaml_fig4_experiment.py --models alexnet resnet18 vgg19 vit_l_16` 형태로 실행한다. `vit_l_16`은 full engine이 큰 모델이므로 디스크 여유 공간과 `--min-free-gb` 조건을 먼저 확인하는 것이 좋다.

## 검증 결과

실행한 검증:

```bash
python -m py_compile scripts/15_compare_k1_timing_semantics.py scripts/27_reprofile_cached_intervals.py src/optimization/candidate_space.py src/optimization/config_evaluator.py src/integration/mask_applicator.py src/integration/dnn_taskset_loader.py src/integration/dnn_taskset_generator.py src/integration/dnn_workload_generator.py src/integration/dnn_algorithm_runner.py
python -m json.tool configs/split_point_policies.json
python scripts/27_reprofile_cached_intervals.py --models alexnet resnet18 vgg19 --precision fp32 --dry-run --limit 5
conda run --no-capture-output -n trt python -m pytest -q
```

결과:

- py_compile 통과
- split policy JSON 유효성 통과
- reprofile script dry-run으로 cached interval 탐색 확인
- 전체 테스트 통과: `68 passed`
