#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>

// RAII CUDA event pair for GPU timing.
struct CudaTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        if (start) cudaEventDestroy(start);
        if (stop)  cudaEventDestroy(stop);
    }
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void record_start(cudaStream_t s) { cudaEventRecord(start, s); }
    void record_stop(cudaStream_t s)  { cudaEventRecord(stop,  s); }

    // Blocks until stop has fired; returns elapsed ms.
    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

using WallClock = std::chrono::steady_clock;
using WallPoint = std::chrono::time_point<WallClock>;

inline WallPoint wall_now() { return WallClock::now(); }
inline double wall_ms(WallPoint t0, WallPoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
