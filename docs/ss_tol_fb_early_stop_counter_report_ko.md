# SS-tol-fb early stop 및 profiling counter 분석 리포트

## 요약

- `src/rta/analysis.py` 기준으로 `RTA_SS_tol`에는 optimistic early stop이 없다.
- early stop 메커니즘은 `_RTA_SS_tol_fb_impl(task_set, early_stop=True)`에만 있으며, 공개 wrapper로는 `RTA_SS_tol_fb_early()`가 이 경로를 호출한다.
- 기존 `src/integration/dnn_algorithm_runner.py`의 SS `tol-fb` 구현은 tolerance-fit 및 fallback 흐름은 대체로 맞았지만, fallback 직후 `get_optimistic_SS_R()`로 더 쪼개도 deadline을 만족할 수 없는지 검사하는 early stop 단계가 빠져 있었다.
- 이번 수정으로 DNN runner의 `ss:tol-fb`는 `_RTA_SS_tol_fb_impl(..., early_stop=True)`에 맞춰 fallback 후 optimistic R 검사를 수행한다.
- `apply_k_chunks()` 호출 수와 K-split 탐색에서 암묵적으로 평가해야 하는 candidate mask/chunk profile/inference run 수를 카운트하도록 instrumentation을 추가했다.

## analysis.py 알고리즘 확인

`RTA_SS_tol`은 `src/rta/analysis.py:589`에서 시작한다.

1. 우선순위 순서로 task를 정렬하고, task index `i`를 0부터 끝까지 한 번 순회한다.
2. 각 task에 대해 `get_SS_R()`로 현재 response time `R_i`를 계산한다.
3. 마지막 task가 아니면 `get_SS_tolerance()`로 tolerance를 갱신한다.
4. `R_i <= D_i`이면 다음 task로 넘어간다.
5. deadline miss가 발생하면 lower-priority task 중 현재 target tolerance보다 큰 GPU block을 가진 task를 `find_splitting_target()`으로 찾는다.
6. 찾은 target에 대해 기존 analytical `split_segment(..., current_n + 1)`을 호출한다.
7. split 후 `update_SS_R_list_and_tolerance_list()`로 이미 본 prefix task들의 R/tolerance를 갱신한다.
8. tolerance-fit이 실패하면 즉시 unschedulable로 종료한다.

이 함수에는 fallback도 없고, optimistic early stop도 없다.

`_RTA_SS_tol_fb_impl`은 `src/rta/analysis.py:669`에서 시작한다.

1. Step 1은 `RTA_SS_tol`과 동일하게 SS RTA와 tolerance를 계산한다.
2. Step 2도 tolerance-fit split을 먼저 시도한다.
3. tolerance-fit이 성공하면 `i += 1`로 다음 task를 본다.
4. tolerance-fit이 실패하면 Step 3 fallback으로 `split_largest_block_excluding_highest()`를 호출한다. 즉, highest-priority task를 제외하고 현재 가장 큰 GPU block을 하나 더 쪼갠다.
5. fallback split 이후 restart index를 `min(i, splitted_task_idx)` 의미로 잡아, 영향받을 수 있는 앞쪽 task부터 다시 본다.
6. Step 4 early stop은 `early_stop=True`일 때만 실행된다. `get_optimistic_SS_R()`로 모든 lower-priority task가 full split까지 갔다고 가정한 optimistic R을 계산하고, 이 optimistic R조차 `D_k`를 넘으면 즉시 unschedulable로 반환한다.

중요한 wrapper 차이는 다음과 같다.

- `RTA_SS_tol_fb()`는 `early_stop=False`로 호출한다 (`src/rta/analysis.py:793`).
- `RTA_SS_tol_fb_early()`는 `early_stop=True`로 호출한다 (`src/rta/analysis.py:796`).

따라서 원본 analytical API 명명만 보면 `tol-fb`와 `tol-fb-early`가 분리되어 있다. 다만 DNN 실험 runner에는 별도의 `ss:tol-fb-early` 알고리즘명이 없고, 사용자는 `ss:tol`과 `ss:tol-fb` 비교로 early stop 효과를 보려는 실험을 의도하고 있다. 그래서 DNN runner의 `ss:tol-fb`를 early-stop 포함 버전으로 맞추는 것이 현재 실험 목적에는 맞다.

## dnn_algorithm_runner.py 구현 확인

`_run_ss_tol`은 `src/integration/dnn_algorithm_runner.py:847`에서 시작한다.

- 시작 전에 `_apply_no_split_measured_to_all()`로 모든 task의 K=1 measured timing을 적용한다.
- 원본 `RTA_SS_tol`의 Step 1과 같이 `get_SS_R()`, `get_SS_tolerance()`를 계산한다.
- deadline miss 시 `does_all_lower_meet_tolerance()`, `find_splitting_target()` 순서로 target을 찾는다.
- 원본의 `split_segment(..., current_n + 1)` 자리는 DNN measured K-search인 `apply_k_chunks(..., current_n + 1, ...)`로 대체되어 있다.
- split 후 `update_SS_R_list_and_tolerance_list()`로 R/tolerance를 갱신한다.
- fallback과 optimistic early stop은 없다. 이는 `RTA_SS_tol`과 일치한다.

`_run_ss_tol_fb`는 `src/integration/dnn_algorithm_runner.py:960`에서 시작한다.

- Step 1과 Step 2는 `_RTA_SS_tol_fb_impl`의 SS RTA 및 tolerance-fit split 흐름과 대응한다.
- tolerance-fit split 호출은 `apply_k_chunks()`로 대체된다.
- Step 3 fallback은 `_dnn_split_largest_excluding_highest()`로 구현되어 있고, highest-priority task를 제외한 가장 큰 `max_G_block`을 `cur_n + 1`로 split한다.
- 이번 수정 전에는 fallback 이후 optimistic early stop 단계가 없었다.
- 이번 수정 후에는 fallback 직후 `get_optimistic_SS_R(sorted_task_list)`를 호출하고, 하나라도 `R_k > D_k`이면 `is_schedulable=False`로 루프를 중단한다 (`src/integration/dnn_algorithm_runner.py:1092`).

주의할 점은 `get_optimistic_SS_R()`가 원본 analytical 함수 그대로라는 점이다. 즉, optimistic check 자체는 새 profiling을 하지 않고 `split_segment(..., max_block_count)`를 deepcopy 상에서 수행해 full-split lower-bound 성격의 R을 계산한다. base fine-grained timing이 정확히 준비되어 있어야 이 early stop 판정도 의미가 있다.

## 추가한 counter 정의

`ProfilingStats`에 다음 필드를 추가했다 (`src/integration/dnn_algorithm_runner.py:141`).

- `k_split_calls`: `apply_k_chunks()` 호출 횟수. 사용자가 말한 split 함수 호출 수에 대응한다.
- `k_split_cache_hits`: measured-best K cache가 hit되어 candidate enumeration을 우회한 횟수.
- `k_split_candidate_masks`: 각 K-search 호출에서 가능한 candidate mask 수의 누적합.
- `k_split_candidate_chunk_profiles`: candidate mask마다 K개 chunk를 profile한다고 볼 때의 chunk profile 수. 한 호출당 `candidate_masks * K`.
- `k_split_candidate_inference_runs`: chunk profile 수에 `(warmup + iters)`를 곱한 inference 실행 수 추정치.
- `early_stop_optimistic_checks`: optimistic R check에서 검사한 task 수 누적합.
- `early_stop_optimistic_deadline_misses`: optimistic R이 deadline을 넘어서 early stop을 유발한 횟수.

`apply_k_chunks()`의 candidate 수 계산은 다음과 같다 (`src/integration/mask_applicator.py:406`).

- enabled split boundary 수를 `E`라고 하고, 목표 chunk 수를 `K`라고 하면 candidate mask 수는 `C(E, K-1)`이다.
- 각 candidate mask는 K개의 chunk interval을 만든다.
- 따라서 한 번의 K-search에서 chunk profile 수는 `C(E, K-1) * K`이다.
- warmup=20, iters=200이면 inference 실행 수는 `C(E, K-1) * K * 220`이다.

사용자가 든 예시처럼 split point가 2개이고 K=2이면 `E=2`, `C(2,1)=2`이다. candidate는 `[1,0]`, `[0,1]` 두 개이고, 각 candidate마다 2개 chunk를 profile하므로 `2 * 2 = 4` chunk profile이다. warmup=20, iters=200이면 `4 * 220 = 880` inference runs로 카운트된다.

이 counter들은 cache hit 여부와 별개로 "그 K-search를 실제로 cold하게 수행했다면 필요했을 profiling workload"를 센다. 기존 `real_profiles`는 cache miss로 실제 새 profiling/build가 일어난 mask 평가 수이고, `masks_evaluated`는 runner가 `evaluate_and_apply_mask()`를 호출한 횟수에 가깝다. early stop 효과를 보려면 `k_split_calls`, `k_split_candidate_chunk_profiles`, `k_split_candidate_inference_runs`를 같이 보는 것이 더 직접적이다.

## 출력 파일 반영

새 counter들은 `summarize_result()`에 추가되어 per-taskset row에 저장된다 (`scripts/internal_fig4_helpers.py:461`).

`scripts/30_run_yaml_fig4_experiment.py`의 CSV 필드에도 추가했다.

- `per_taskset_results.csv`: raw per-run counter 값
- `schedulability_ratio.csv`: utilization/algorithm별 평균 counter
- `split_activity.csv`: split activity와 함께 평균 counter

early stop 실험에서 boxplot과 histogram을 그릴 때는 `per_taskset_results.csv`를 쓰는 것이 가장 적합하다.

## 추천 실험 방식

1_base YAML config로 `ss:tol`과 `ss:tol-fb`만 실행한다.

```bash
python scripts/30_run_yaml_fig4_experiment.py \
  --config configs/yaml/260511_configs/1_base.yaml \
  --algorithms ss:tol:SS-tol ss:tol-fb:SS-tol-fb \
  --split-policy major_blocks \
  --wcet-metric max \
  --live \
  --run-name early_stop_ss_tol_vs_tolfb
```

분석할 주요 컬럼은 다음이다.

- `k_split_calls`: split 함수 호출 횟수 비교
- `k_split_candidate_masks`: K-search candidate mask 수 비교
- `k_split_candidate_chunk_profiles`: 실제 서로 다른 chunk profile workload 비교
- `k_split_candidate_inference_runs`: warmup/iters까지 반영한 profiling 실행량 비교
- `early_stop_optimistic_deadline_misses`: `ss:tol-fb`에서 early stop이 실제로 걸린 횟수
- `real_profiles`: cache를 고려한 실제 새 profiling 횟수
- `cache_hits`, `k_split_cache_hits`: cache가 실험 비용을 얼마나 줄였는지 확인

## 검증

문법 검사는 통과했다.

```bash
python -m py_compile \
  src/integration/dnn_algorithm_runner.py \
  src/integration/mask_applicator.py \
  scripts/internal_fig4_helpers.py \
  scripts/30_run_yaml_fig4_experiment.py
```
