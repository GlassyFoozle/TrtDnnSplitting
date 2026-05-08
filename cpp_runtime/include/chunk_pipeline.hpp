#pragma once
#include "trt_engine.hpp"
#include "buffer_pool.hpp"
#include "timing.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct ChunkStats {
    int    id          = 0;
    double gpu_mean_ms = 0.0;
    double gpu_p99_ms  = 0.0;
    double cpu_mean_ms = 0.0;
    double cpu_p99_ms  = 0.0;
};

struct PipelineResult {
    double              full_engine_gpu_mean_ms    = 0.0;
    double              full_engine_gpu_p99_ms     = 0.0;
    double              total_chunked_gpu_mean_ms  = 0.0;
    double              total_chunked_gpu_p99_ms   = 0.0;
    int                 n_iters                    = 0;
    std::vector<ChunkStats> chunks;
};

// Loads all engines for a critical_full config, runs them zero-copy in sequence.
class ChunkPipeline {
public:
    ChunkPipeline(const nlohmann::json& cfg,
                  const std::string& repo_root,
                  const std::string& precision,
                  nvinfer1::ILogger& logger);
    ~ChunkPipeline();

    // Run the full pipeline; returns aggregate stats.
    PipelineResult run(int n_warmup, int n_iters);

private:
    void _setup_buffers();
    void _bind_all();
    double _percentile(std::vector<double>& v, double p);

    std::string             repo_root_;
    std::string             precision_;
    nvinfer1::ILogger&      logger_;
    cudaStream_t            stream_ = nullptr;

    // Per-chunk engines (chunks_.size() == n_chunks).
    std::vector<TrtEngine>  chunks_;
    // Intermediate device buffers; buffer[i] is output of chunk i / input of chunk i+1.
    // buffer[0] = input, buffer[n] = final output.
    std::vector<DeviceBuffer> bufs_;

    // Optional full-model engine for baseline timing.
    std::unique_ptr<TrtEngine> full_engine_;

    // Descriptions from config (for logging).
    std::vector<std::string> descriptions_;
};
