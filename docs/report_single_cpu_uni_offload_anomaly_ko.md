# single CPU에서 UNI가 Offload보다 낮게 나온 원인

## 결론

이번 config 8 재실험에서 `UNI-*`가 `SS/Offload-*`보다 낮게 나온 것은 이론적 현상이 아니라 코드 버그다.

single CPU 조건에서 UNI가 Offload보다 구조적으로 더 나빠지는 것은 말이 안 된다는 지적이 맞다. 현재 결과는 UNI 변환 과정에서 measured K=1 GPU timing을 잃고 `dag_aligned_full` 합산 timing으로 되돌아가서 생긴 문제였다.

## 증상

config 8 / U=0.9 / `taskset_000`에서 저장 결과는 다음과 같았다.

- `SS-tol-fb`: schedulable
- `UNI-tol-fb`: unschedulable

그런데 task detail을 보면 timing이 다르게 들어가 있었다.

SS:

```text
tau2: C=0.4377, G=0.8590, chunks=[0.85899]
```

UNI:

```text
tau2: C=0.0, G=1.4957, chunks=[0.2038, 1.057966, 0.2339]
```

UNI의 GPU block이 measured K=1 `0.85899`가 아니라 `dag_aligned_full` 합산 `1.057966`으로 들어갔다. 여기에 `C_pre/C_post`까지 UNI block으로 붙으면서 total UNI execution이 더 커졌다.

## 원인

실험 runner는 UNI 알고리즘 시작 전에 `_apply_no_split_measured_to_all()`로 SS task에 measured K=1 timing을 patch한다.

즉 SS task 내부는 정상적으로:

```text
segment.G_block_list = [0.85899]
```

가 된다.

하지만 그 다음 `convert_SS_to_UNI()`가 UNI task를 만들 때 `segment.G_block_list`를 보존하지 않고, `segment.base_block_list`와 `splitting_config`로 다시 `_compute_block_list()`를 호출했다.

그 결과:

```text
measured K=1:  [0.85899]
```

가

```text
base chunk sum: [1.057966]
```

으로 되돌아갔다.

즉 이전에 고쳤던 `UNI -> SS` measured timing 보존 문제의 반대 방향 문제가 남아 있었다.

## 수정

수정 파일:

- `src/rta/task.py`
- `tests/test_uni_conversion_alignment.py`

`SegInfTask.convert_SS_to_UNI()`가 현재 SS segment의 `G_block_list`를 UNI group에 매핑해서 보존하도록 바꿨다.

이제 K=1 SS task:

```text
C_pre=0.2038, G_block_list=[0.85899], C_post=0.2339
```

는 UNI 변환 후:

```text
UNI G_block_list=[0.2038, 0.85899, 0.2339]
```

가 된다.

## 재현 확인

같은 config 8 / U=0.9 / `taskset_000`을 수정 후 단독 재실행하면:

- `SS-tol-fb`: schedulable
- `UNI-single`: schedulable
- `UNI-tol-fb`: schedulable

수정 후 UNI task detail:

```text
tau2: chunks=[0.2038, 0.8590, 0.2339], R=2.1557, D=3.6592
tau5: chunks=[0.0230, 0.8590, 0.0605], R=3.0982, D=4.1468
...
```

기존처럼 `1.057966`이 들어가지 않는다.

## 검증

```text
python -m py_compile src/rta/task.py src/integration/dnn_algorithm_runner.py
pytest -q tests/test_k1_accounting.py tests/test_uni_conversion_alignment.py tests/test_measured_k_split.py tests/test_split_policy.py
```

결과:

```text
19 passed
```

## 영향

기존 `fig4_resnet_1_base`, `fig4_resnet_8_singleCPU_task8` 결과는 이 버그가 섞여 있으므로 다시 사용하면 안 된다.

특히 UNI 계열은 measured K=1보다 큰 `dag_aligned_full` 합산 timing으로 분석된 케이스가 있어 schedulability가 과도하게 낮게 나왔다.
