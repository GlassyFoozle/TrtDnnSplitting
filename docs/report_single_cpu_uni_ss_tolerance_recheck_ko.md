# single CPU config 7/8 UNI vs SS/offload 재점검

## 결론

single CPU에서 `UNI-tol`이 `SS-tol`보다 구조적으로 낮게 나오는 현상은 현재 최신 코드 기준으로는 보이지 않는다.

문제가 있어 보였던 비교는 대부분 다음 두 가지가 섞인 것이다.

1. plot/report의 "Tolerance"가 순수 tolerance가 아니라 `tol-fb` fallback 포함 알고리즘을 가리킨다.
2. `SS-tol-fb`와 `UNI-heu/opt`는 같은 search objective가 아니다.

최신 코드로 기존 generated taskset을 직접 재실행한 결과, U=0.9 기준은 다음과 같다.

| config | method | schedulable / 50 |
|---|---:|---:|
| 7 single CPU task4 | `SS-tol-fb` | 29 |
| 7 single CPU task4 | `UNI-tol-fb` | 28 |
| 7 single CPU task4 | `UNI-heu` | 17 |
| 7 single CPU task4 | `UNI-opt` | 17 |
| 7 single CPU task4 | `SS-heu` | 11 |
| 7 single CPU task4 | `SS-opt` | 11 |
| 8 single CPU task8 | `SS-tol-fb` | 25 |
| 8 single CPU task8 | `UNI-tol-fb` | 28 |
| 8 single CPU task8 | `UNI-heu` | 19 |
| 8 single CPU task8 | `UNI-opt` | 19 |
| 8 single CPU task8 | `SS-heu` | 11 |
| 8 single CPU task8 | `SS-opt` | 12 |

즉 config 8은 최신 코드 기준으로 `UNI-tol-fb > SS-tol-fb`다. config 7은 `SS-tol-fb`가 1 taskset 차이로 높다.

저장 CSV의 순수 tolerance도 같은 방향이다.

| config / U=0.9 | `SS-tol` | `UNI-tol` |
|---|---:|---:|
| 7 | 17/50 | 20/50 |
| 8 | 14/50 | 19/50 |

따라서 “single CPU인데 offload tolerance가 UNI tolerance보다 본질적으로 우월하다”는 해석은 맞지 않는다. 순수 tolerance끼리 비교하면 UNI가 더 높다.

## 왜 `SS-tol-fb`가 `UNI-heu/opt`보다 높을 수 있나

`tol-fb`는 pure tolerance 알고리즘이 아니다.

`SS-tol-fb`는 tolerance 조건을 만족하지 못하면 fallback으로 lower-priority side의 largest block을 계속 split한다. 반면 `UNI-heu`/`UNI-opt`는 paper-style search다.

- full-split feasibility check를 통과해야 한다.
- task별 tolerance를 기준으로 candidate mask를 고른다.
- fallback으로 계속 쪼개서 최종 RTA를 맞추는 절차가 없다.
- `opt`도 전체 taskset/mask 조합에 대한 global optimum이 아니라, 현재 paper-style boundary subset search 안에서의 optimum이다.

따라서 `SS-tol-fb > UNI-heu/opt`는 “SS/offload RTA가 UNI RTA보다 좋아서”라기보다 “fallback 포함 알고리즘과 fallback 없는 paper-style 알고리즘을 비교했기 때문”이다.

공정한 비교는 다음처럼 봐야 한다.

- pure tolerance 비교: `SS-tol` vs `UNI-tol`
- fallback 포함 비교: `SS-tol-fb` vs `UNI-tol-fb`
- paper baseline 비교: `SS-heu/opt` vs `UNI-heu/opt`

## 왜 config 7에서 `SS-tol-fb`가 `UNI-tol-fb`보다 1개 높나

config 7 U=0.9에서 최신 코드 기준 `SS-tol-fb`만 성공하고 `UNI-tol-fb`는 실패한 taskset은 7개 있었다.

예: `taskset_002`.

`SS-tol-fb` 최종 상태:

| task | R/D | K | max block | GPU sum |
|---|---:|---:|---:|---:|
| tau2 | 1.590 / 2.380 | 1 | 0.859 | 0.859 |
| tau4 | 3.115 / 6.312 | 3 | 0.307 | 0.897 |
| tau1 | 6.437 / 8.061 | 3 | 0.307 | 0.897 |
| tau3 | 12.664 / 40.717 | 3 | 0.307 | 0.897 |

`UNI-tol-fb` 최종 상태:

| task | R/D | K | max UNI block | UNI sum |
|---|---:|---:|---:|---:|
| tau2 | 1.744 / 2.380 | 1 | 0.859 | 1.283 |
| tau4 | 4.147 / 6.312 | 7 | 0.269 | 1.121 |
| tau1 | 11.780 / 8.061 | 7 | 0.461 | 1.428 |
| tau3 | 18.871 / 40.717 | 7 | 0.269 | 0.964 |

UNI 변환은 simulation 기준과 동일하게 `C_pre`, measured GPU chunks, `C_post`를 모두 UNI block으로 둔다. 따라서 UNI task에서는 `C=0`, `G=C_pre + measured GPU chunks + C_post`가 된다.

이 케이스는 split 탐색 차이만으로 설명되지 않는다. `SS-tol-fb`가 찾은 동일한 final mask를 UNI RTA에 강제로 넣어도 UNI는 실패한다.

| final mask source | SS RTA | UNI RTA |
|---|---:|---:|
| `SS-tol-fb` masks | schedulable | unschedulable |
| `UNI-tol-fb` masks | schedulable | unschedulable |

`SS-tol-fb` final mask를 그대로 썼을 때 `tau1`의 상세 값은 다음과 같다.

SS request-driven:

| term | value |
|---|---:|
| `C_i + G_i` | 1.409107 |
| GPU blocking `B_i` | 2.921457 |
| CPU interference `I_i` | 2.106064 |
| `R_i` | 6.436628 |
| `D_i` | 8.061065 |

UNI:

| term | value |
|---|---:|
| `C_i + G_i` after UNI conversion | 1.409107 |
| lower blocking `B_i` | 0.306852 |
| `K_i` | 2 |
| `R_i` | 11.743006 |
| `D_i` | 8.061065 |

차이는 high-priority work를 세는 범위다.

SS request-driven bound에서는 GPU demand가 response window 전체에 CPU interference처럼 들어가지 않는다. GPU demand는 `get_B_i_req()`의 GPU blocking fixed point 안에서만 계산된다.

`tau1` 기준 SS의 high-priority contribution은 대략 다음처럼 분리된다.

| HP task | CPU jobs in `I_i` | CPU contribution | GPU jobs in `B_i` | GPU contribution |
|---|---:|---:|---:|---:|
| `tau2` | 4 | 1.694704 | 2 | 1.717980 |
| `tau4` | 2 | 0.411360 | 1 | 0.896626 |

반면 UNI는 `get_UNI_R_and_K()`에서 high-priority task의 전체 demand를 다음 값으로 한 번에 센다.

```python
C_h = task_h.C + task_h.G
```

`tau1`의 `k=2` response 계산에서는 다음처럼 잡힌다.

| HP task | UNI jobs | `C+G` per job | contribution |
|---|---:|---:|---:|
| `tau2` | 5 | 1.282666 | 6.413329 |
| `tau4` | 2 | 1.102306 | 2.204612 |

즉 single CPU라고 해서 SS와 UNI가 같은 interference를 세는 것은 아니다. CPU interference는 single CPU에서 모두 잡히지만, SS request-driven bound는 GPU interference를 별도 GPU blocking interval로 제한한다. UNI는 high-priority task의 `C+G`를 unified demand로 start-time recurrence에 넣는다.

따라서 "single CPU라서 CPU를 합치고 말고는 영향이 없다"는 말은 CPU 항에 대해서는 맞지만, GPU 항까지 포함한 RTA dominance에는 충분하지 않다. single CPU여도 CPU와 GPU는 여전히 서로 다른 resource이고, SS request-driven analysis는 그 분리를 이용한다.

이 때문에 현재 구현의 수식상 `R_UNI <= R_SS`는 항상 성립하지 않는다. 이전 simulation aggregate에서는 UNI가 더 좋은 경향을 보였지만, per-taskset strict dominance는 보장되지 않는다.

## 현재 타당한 해석

- 순수 tolerance 결과는 직관과 맞다: config 7/8 U=0.9에서 `UNI-tol > SS-tol`.
- config 8의 fallback 포함 결과도 직관과 맞다: `UNI-tol-fb > SS-tol-fb`.
- config 7의 `SS-tol-fb > UNI-tol-fb`는 1개 taskset 차이이며, fallback split trajectory와 SS/UNI RTA interference-counting 차이에서 나온 경계 케이스다.
- `SS-tol-fb > UNI-heu/opt`는 RTA 우위가 아니라 알고리즘 비교가 섞인 결과다.

따라서 그림 label은 `Tolerance` 대신 `Tolerance-FB` 또는 `Tolerance+Fallback`으로 분리해서 표시하는 편이 맞다. simulation의 pure tolerance와 맞춰 해석하려면 `SS-tol`/`UNI-tol`을 별도 plot으로 봐야 한다.

## 왜 prev에서는 잘 안 보였나

prev simulation은 integer timing과 synthetic period/task generation을 사용했다. 현재 TRT config 7/8은 ResNet18 p99 measured K=1이 약 `0.859 ms`이고, U=0.9에서 짧은 period task가 많이 생긴다.

예시 `taskset_002`는 high-priority `tau2` period가 `2.380 ms`이고, UNI 변환 후 `C+G=1.283 ms`다. 이 값이 `tau1`의 UNI start-time recurrence에서 5 jobs까지 잡히면서 `6.413 ms`를 차지한다.

prev의 aggregate 결과에서 UNI가 더 좋았던 것은 이 상황이 자주/강하게 발생하지 않았기 때문으로 보는 게 맞다. `number_of_inference_segment=1` 자체가 `UNI <= SS`를 보장하는 조건은 아니다.
