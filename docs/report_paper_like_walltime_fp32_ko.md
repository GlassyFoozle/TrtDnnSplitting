# Paper-like CPU wall-clock WCET

측정 구간: `enqueue 직전 -> cudaStreamSynchronize()` 완료 직후.
TensorRT engine은 기존 파일을 그대로 재사용했고 새로 빌드하지 않았다.
full-split은 `trt_fusion_safe` 정책에서 허용되는 max-K split을 사용했다.

## 기존 측정 방식: GPU event timing

기존 measured-K / schedulability 경로가 주로 쓰던 값은 `gpu_max_ms` 계열이다. 측정 방식은 CUDA stream에 start/stop event를 기록하고, `cudaEventElapsedTime()`으로 두 event 사이의 GPU 실행 시간을 구하는 방식이다.

```cpp
timers[c].record_start(stream);
execute(chunk);
timers[c].record_stop(stream);
gpu_ms = timers[c].elapsed_ms();
```

이 방식은 GPU가 해당 구간을 실행한 시간은 잘 재지만, CPU가 chunk 요청을 내고 완료를 확인할 때까지의 전체 wall time은 포함하지 않는다. chunk별 dispatch, host-side wait, per-chunk synchronization overhead를 논문식 WCET처럼 누적해서 보려는 목적에는 부족하다.

## 기존 GPU 측정값

아래 값은 이번 실험과 **같은 엔진 / 같은 split / 같은 반복**에서 얻은 GPU event 기준 `gpu_max_ms`다.

| model | config | chunks | per-chunk GPU max (ms) | sum of chunk GPU WCETs (ms) | measured total GPU max (ms) |
|---|---|---:|---|---:|---:|
| alexnet | non-split | 1 | 2.096 | 2.096 | 2.310 |
| alexnet | full-split | 11 | 0.248, 0.093, 0.238, 0.102, 0.263, 0.247, 0.218, 0.107, 1.199, 0.576, 0.240 | 3.531 | 3.574 |
| resnet18 | non-split | 1 | 1.524 | 1.524 | 1.692 |
| resnet18 | full-split | 11 | 0.302, 0.190, 0.211, 0.243, 0.217, 0.245, 0.376, 0.356, 0.277, 0.064, 0.089 | 2.570 | 3.076 |
| vgg19 | non-split | 1 | 9.728 | 9.728 | 9.826 |
| vgg19 | full-split | 12 | 1.638, 0.917, 0.870, 1.078, 0.880, 1.020, 0.622, 0.619, 0.120, 2.725, 0.540, 0.227 | 11.256 | 12.717 |

## 새 측정 방식: paper-like CPU wall time

이번에 추가한 실험 모드는 각 chunk마다 `execute()` 직전부터 `cudaStreamSynchronize()`가 끝난 직후까지를 CPU steady clock으로 잰다.

```cpp
wall0 = wall_now();
execute(chunk);
cudaStreamSynchronize(stream);
wall1 = wall_now();
cpu_wall_ms = wall1 - wall0;
```

즉 CPU가 실제로 chunk 완료를 확인할 때까지의 시간을 재므로, 논문이 말한 “CPU 관점 total latency”에 훨씬 가깝다.

| model | config | chunks | per-chunk CPU wall max (ms) | sum of chunk WCETs (ms) | measured total CPU wall max (ms) |
|---|---|---:|---|---:|---:|
| alexnet | non-split | 1 | 2.298 | 2.298 | 2.298 |
| alexnet | full-split | 11 | 0.369, 0.113, 0.286, 0.126, 0.313, 0.298, 0.244, 0.125, 1.428, 0.767, 0.296 | 4.364 | 3.536 |
| resnet18 | non-split | 1 | 1.682 | 1.682 | 1.682 |
| resnet18 | full-split | 11 | 0.455, 0.226, 0.232, 0.378, 0.351, 0.393, 0.413, 0.501, 0.444, 0.261, 0.102 | 3.757 | 3.063 |
| vgg19 | non-split | 1 | 9.806 | 9.806 | 9.806 |
| vgg19 | full-split | 12 | 1.830, 1.078, 1.072, 1.259, 1.077, 1.190, 0.778, 0.798, 0.134, 2.886, 0.712, 0.394 | 13.208 | 12.689 |

`sum of chunk WCETs`가 논문 Table IV의 Sum과 가장 직접적으로 대응한다.

## 논문 Table IV와의 비교

| model | metric | paper Table IV (ms) | current paper-like wall time (ms) |
|---|---|---:|---:|
| alexnet | non-split | 4.469 | 2.298 |
| alexnet | full-split sum | 4.802 | 4.364 |
| resnet18 | non-split | 2.533 | 1.682 |
| resnet18 | full-split sum | 3.751 | 3.757 |
| vgg19 | non-split | 6.615 | 9.806 |
| vgg19 | full-split sum | 11.425 | 13.208 |

## 짧은 해석

- wall-clock + per-chunk sync를 넣자 full-split 합은 논문과 훨씬 가까워졌다. 특히 `resnet18`은 `3.751 ms` 대 `3.757 ms`로 거의 일치한다.
- 같은 엔진에서도 full-split 합은 GPU event 기준보다 wall-clock 기준에서 더 커졌다. `alexnet`은 `3.531 -> 4.364 ms`, `resnet18`은 `2.570 -> 3.757 ms`, `vgg19`는 `11.256 -> 13.208 ms`다.
- 그래도 non-split은 아직 크게 다르다. 이는 측정 방식 외에도 Caffe 대 torchvision/ONNX 그래프 차이, TensorRT engine/tactic 차이가 남아 있다는 뜻이다.
- `vgg19`는 full-split도 논문보다 더 크다. 현재 full-split이 논문 6 chunks가 아니라 `trt_fusion_safe` max-K 12 chunks라서, chunk별 dispatch/sync overhead가 더 많이 누적되는 영향이 있다.
