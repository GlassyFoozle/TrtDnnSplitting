# Measured K-Split Search 변경 리포트

## 배경

기존 `K` split 경로는 requested K에 대해 `base_block_list`의 합이 가장 균형 잡히도록 DP balanced split을 수행했다.

```text
base_block_list -> balanced_split / policy_aware_balanced_split -> mask 1개 선택
mask 1개만 TRT evaluate/profile
```

이 방식은 가능한 boundary 조합을 모두 compile/profile하기 어렵다는 전제에서 만든 근사다. 하지만 현재 실험은 `major_blocks`처럼 enabled boundary 수가 제한된 split policy를 쓰고, interval/chunk cache도 사용할 수 있으므로 K개 chunk를 만드는 모든 policy-allowed mask를 실제 measured timing으로 비교할 수 있다.

## 수정 내용

`src/integration/mask_applicator.py::apply_k_chunks()`를 measured K-search 방식으로 변경했다.

변경 전:

- `balanced_split()` 또는 `policy_aware_balanced_split()`로 base timing 기준 mask 1개 선택
- 선택된 mask만 `evaluate_and_apply_mask()`로 measured timing patch

변경 후:

- active boundary 수가 `K-1`인 모든 policy-allowed mask를 생성
- 각 mask를 `evaluate_and_apply_mask()`로 평가
- 성공한 measured 후보 중 최선의 mask를 선택
- 선택된 mask를 다시 적용해 task의 `G_block_list`를 최종 measured timing으로 patch

선택 기준:

1. measured `max(chunk_times)` 최소
2. 동률이면 measured spread, 즉 `max - min` 최소
3. 그래도 동률이면 measured total GPU time 최소

즉 RTA의 blocking에 직접 영향을 주는 max chunk를 우선 최소화하고, 같은 max라면 더 균등한 measured split을 고른다.

## 적용 범위

`apply_k_chunks()`를 사용하는 SS K-split 경로가 모두 바뀐다.

- `SS-tol`
- `SS-tol-fb`
- `SS-heu-k`
- `SS-opt-k`
- SS fallback largest-block split

UNI K-split도 `_uni_apply_k_chunks()`가 `apply_k_chunks()`를 호출하도록 바뀌었다.

- `UNI-tol`
- `UNI-tol-fb`
- UNI fallback largest-block split

`UNI-heu` / `UNI-opt` paper-style search는 원래부터 boundary subset 후보를 직접 measured 평가하므로 기존 raw-mask search를 유지한다.

## split policy 처리

`policy_name != "all"`이면 `get_enabled_boundaries()`로 enabled boundary만 가져온다. 예를 들어 `alexnet/major_blocks`에서는 다음 boundary만 후보가 된다.

```text
[2, 5, 7, 9, 12, 14, 17, 20]
```

K=2이면 위 boundary 각각을 하나만 켠 mask를 전부 measured 평가한다. K=3이면 enabled boundary 중 2개를 고르는 모든 조합을 평가한다.

요청 K가 policy로 표현 가능한 최대 chunk 수보다 크면 기존 정책과 같이 `len(enabled_boundaries) + 1`로 clamp한다.

## 통계 처리

`apply_k_chunks(..., search_stats=result.stats)`를 넘기는 경로에서는 내부 후보 mask 평가도 `ProfilingStats`에 반영된다. 따라서 K-search가 실제로 여러 mask를 profile/cache-hit한 비용이 통계에 잡힌다.

최종 선택 mask는 task state를 확정하기 위해 한 번 재적용한다. 이 재적용은 대부분 cache hit이어야 한다.

## measured-only 정책과의 관계

각 후보는 `evaluate_and_apply_mask()`를 통해 measured timing이 있는 경우에만 성공한다.

- cache hit measured timing 사용 가능
- cache miss면 export/build/profile 수행
- dry-run, evaluation failure, timing 없음, chunk count mismatch는 실패
- base timing 합산 fallback은 사용하지 않음

따라서 K-search도 RTA에 들어가는 `G_block_list`에는 measured timing만 넣는다.

## 추가 테스트

추가 파일:

- `tests/test_measured_k_split.py`

검증 항목:

- K=2에서 가능한 모든 mask를 evaluate하는지
- base timing 기준으로 균등하지 않아도 measured max chunk가 가장 낮은 mask를 선택하는지
- `major_blocks` policy가 enabled boundary 후보를 제한하는지

