# TRT UNI/SS 변환 정렬 리포트

## 배경

`prev` 시뮬레이션 실험의 UNI 변환은 SS task를 하나의 UNI inference segment로 합치되, CPU와 GPU를 서로 다른 UNI block으로 유지한다.

예를 들어 SS task가 다음 구조라면:

```text
C_pre, G_0, G_1, ..., G_N-1, C_post
```

시뮬레이션 UNI의 `base_block_list`와 no-split `G_block_list`는 다음 구조를 갖는다.

```text
[C_pre, G_0 + G_1 + ... + G_N-1, C_post]
```

GPU 내부 split이 적용되면 CPU block은 계속 별도 fixed block으로 남는다.

```text
[C_pre, GPU_chunk_0, GPU_chunk_1, ..., GPU_chunk_K-1, C_post]
```

`src/rta/task.py`의 `convert_SS_to_UNI()`는 C/G boundary를 `fixed_one_indices`로 고정하고, 해당 boundary에는 split overhead도 붙이지 않는다. 따라서 CPU pre/post는 첫 GPU chunk 또는 마지막 GPU chunk에 합쳐지지 않는다.

## 확인된 차이

TRT 실험 경로의 UNI split 적용 함수는 measured GPU chunk timing을 UNI task에 직접 덮어쓰면서 다음 형태로 재구성하고 있었다.

```text
[C_pre + GPU_chunk_0, GPU_chunk_1, ..., GPU_chunk_K-1 + C_post]
```

해당 경로는 다음 두 함수였다.

- `src/integration/dnn_algorithm_runner.py::_uni_apply_k_chunks()`
- `src/integration/dnn_algorithm_runner.py::_uni_apply_raw_mask()`

반면 `_uni_config_from_trt_mask()`는 이미 C/G boundary를 fixed split으로 확장하고 있었다. 즉 `splitting_config`는 시뮬레이션 UNI 방식인데, `G_block_list`만 CPU와 GPU를 합친 형태였다.

이 불일치 때문에 TRT UNI RTA에서 다음 값들이 시뮬레이션 기준과 달라질 수 있었다.

- `max_G_block`
- `get_UNI_last_segment()`가 보는 마지막 block
- `get_UNI_R_and_K()`의 `C_i_last`
- `get_UNI_tolerance()`의 tolerance window
- tol/tol-fb fallback의 largest-block 선택

## 수정 내용

TRT measured timing을 UNI task에 반영할 때도 시뮬레이션 UNI 구조를 유지하도록 변경했다.

새 helper:

- `_uni_g_blocks_from_measured_chunks(dnn_task, measured_chunks)`
- `_patch_uni_measured_blocks(dnn_task, ut_target, s_idx, trt_mask, measured_chunks)`

변경 후 UNI measured `G_block_list`는 다음 형태다.

```text
[C_pre?, measured_GPU_chunk_0, ..., measured_GPU_chunk_K-1, C_post?]
```

`?`는 값이 0 이하이면 `convert_SS_to_UNI()`의 `append_block()`과 동일하게 block을 만들지 않는다는 의미다.

변경된 함수:

- `_uni_apply_k_chunks()`는 더 이상 `C_pre`/`C_post`를 measured GPU chunk에 더하지 않고 `_patch_uni_measured_blocks()`를 호출한다.
- `_uni_apply_raw_mask()`도 같은 helper를 사용한다.
- 파일 상단 UNI 구현 설명과 `_run_uni_tol_fb()` 주석도 새 block 구조에 맞게 수정했다.

추가로 `convert_UNI_to_SS()`도 measured timing을 보존하도록 수정했다.

기존 `convert_UNI_to_SS()`는 UNI의 현재 `G_block_list`가 아니라 `base_block_list`를 다시 그룹화해서 SS segment를 복원했다. TRT 경로에서는 `G_block_list`에 measured chunk timing이 직접 들어가므로, 이 방식은 SS view를 만들 때 measured timing을 잃는다.

변경 후 동작:

- SS 복원 시 원래 GPU base block list와 `max_block_count`는 유지한다.
- 현재 UNI `splitting_config`가 나타내는 GPU grouping을 SS `splitting_config`로 복원한다.
- 현재 UNI `G_block_list`에서 CPU block은 `C_list`로, GPU block은 SS segment의 measured `G_block_list`로 옮긴다.

따라서 UNI tol/tol-fb가 내부적으로 임시 SS view를 만들 때도 measured `max_G_block`, 현재 chunk 수, 추가 split 가능 범위가 유지된다.

## 기대 효과

TRT 실험의 UNI 표현이 `prev` 시뮬레이션 실험의 UNI/SS 변환 원리와 일치한다.

특히 기존처럼 마지막 block이 `last_gpu_chunk + C_post`가 아니라 `C_post` 단독 block이 되므로, UNI RTA에서 `C_i_last`와 tolerance 계산이 시뮬레이션 기준으로 돌아간다.

또한 `max_G_block` 계산에서도 CPU pre/post가 GPU chunk blocking에 합쳐지지 않는다.

## Measured-only Timing 정책

추가로 SS/UNI 모두 RTA에 들어가는 `G_block_list`는 measured timing만 사용하도록 바꿨다.

변경 전에는 K=1 no-split이 `base_block_list` 합산값을 사용했고, dry-run/evaluation failure/timing count mismatch/live-budget skip에서도 base timing 합산 fallback이 `G_block_list`에 patch될 수 있었다.

변경 후:

- K=1도 `evaluate_mask()` 경로를 사용한다. cache가 있으면 cache measured timing을 쓰고, 없으면 export/build/profile을 수행한다.
- `dry_run=True`는 measured timing을 만들 수 없으므로 `success=False`를 반환하고 task timing을 patch하지 않는다.
- evaluator 실패, measured timing 없음, measured chunk count mismatch는 모두 `success=False`이며 base timing 합산 fallback을 적용하지 않는다.
- UNI `single`, `max`, `tol`, `tol-fb`, `heu`, `opt`와 SS 대응 알고리즘은 RTA 전에 필요한 no-split/full-split mask를 measured path로 적용한다.
- 기존 초기 task loader가 갖고 있는 mask-derived timing은 알고리즘 시작 전 overload guard 등에 사용하지 않도록 했다.

## 추가 검증

다음 단위 테스트를 추가했다.

- `tests/test_uni_conversion_alignment.py`

검증 항목:

- measured UNI block list가 `[C_pre, GPU chunks, C_post]`로 생성되는지
- 0인 CPU block은 `convert_SS_to_UNI()`처럼 생략되는지
- 0인 measured GPU chunk는 chunk 수 보존을 위해 생략하지 않는지
- TRT mask `[1, 0]`이 UNI config `[1, 1, 0, 1]`로 확장되고, measured block list가 CPU/GPU 분리 형태로 patch되는지
- UNI→SS 변환 후 measured GPU chunks, original base blocks, split capacity가 유지되는지
