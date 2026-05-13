#include "chunk_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>

// ── helpers ───────────────────────────────────────────────────────────────────

static size_t float_bytes(size_t n) { return n * sizeof(float); }

static std::string abs_path(const std::string& repo_root, const std::string& rel) {
    if (!rel.empty() && rel[0] == '/') return rel;
    return repo_root + "/" + rel;
}

// ── ChunkPipeline ─────────────────────────────────────────────────────────────

ChunkPipeline::ChunkPipeline(const nlohmann::json& cfg,
                              const std::string& repo_root,
                              const std::string& precision,
                              nvinfer1::ILogger& logger)
    : repo_root_(repo_root), precision_(precision), logger_(logger)
{
    cudaStreamCreate(&stream_);

    // Load chunk engines.
    std::string eng_key = "engine_" + precision;
    for (auto& c : cfg.at("chunks")) {
        std::string path = abs_path(repo_root_, c.at(eng_key).get<std::string>());
        descriptions_.push_back(c.value("description", ""));
        chunks_.emplace_back(path, logger_);
    }

    // Load full-model engine if present.
    auto& fm = cfg.at("full_model");
    std::string full_path = abs_path(repo_root_, fm.at(eng_key).get<std::string>());
    try {
        full_engine_ = std::make_unique<TrtEngine>(full_path, logger_);
    } catch (...) {
        full_engine_.reset(); // not fatal — will report 0 for full latency
    }

    _setup_buffers();
    _bind_all();
}

ChunkPipeline::~ChunkPipeline() {
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

// Allocate one buffer per boundary (n_chunks + 1 boundaries, but the output
// of chunk[i] IS the input of chunk[i+1], so we need n_chunks+1 buffers).
void ChunkPipeline::_setup_buffers() {
    size_t n = chunks_.size();
    bufs_.reserve(n + 1);

    // Buffer 0: input to chunk 0.
    const char* in0 = chunks_[0].io_name(0); // "input" tensor
    // Find the actual input (TRT names tensors "input" / "output" by ONNX export).
    // We walk all IO tensors of chunk 0 to find the input.
    size_t in_vol = 0;
    for (int32_t j = 0; j < chunks_[0].n_io(); ++j) {
        const char* nm = chunks_[0].io_name(j);
        if (chunks_[0].engine()->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT) {
            in_vol = chunks_[0].tensor_volume(nm);
            break;
        }
    }
    if (in_vol == 0) throw std::runtime_error("Could not determine input volume of chunk 0");
    bufs_.emplace_back(float_bytes(in_vol));

    // Buffers 1..n: output of chunk[i-1] = input of chunk[i].
    for (size_t i = 0; i < n; ++i) {
        size_t out_vol = 0;
        for (int32_t j = 0; j < chunks_[i].n_io(); ++j) {
            const char* nm = chunks_[i].io_name(j);
            if (chunks_[i].engine()->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kOUTPUT) {
                out_vol = chunks_[i].tensor_volume(nm);
                break;
            }
        }
        if (out_vol == 0) throw std::runtime_error("Could not determine output volume of chunk");
        bufs_.emplace_back(float_bytes(out_vol));
    }
}

// Bind bufs_[i] → input of chunk[i], bufs_[i+1] → output of chunk[i].
void ChunkPipeline::_bind_all() {
    for (size_t i = 0; i < chunks_.size(); ++i) {
        for (int32_t j = 0; j < chunks_[i].n_io(); ++j) {
            const char* nm = chunks_[i].io_name(j);
            auto mode = chunks_[i].engine()->getTensorIOMode(nm);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
                chunks_[i].bind(nm, bufs_[i].ptr);
            else
                chunks_[i].bind(nm, bufs_[i + 1].ptr);
        }
    }
    // Bind full engine if available.
    if (full_engine_) {
        for (int32_t j = 0; j < full_engine_->n_io(); ++j) {
            const char* nm = full_engine_->io_name(j);
            auto mode = full_engine_->engine()->getTensorIOMode(nm);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
                full_engine_->bind(nm, bufs_.front().ptr);
            else
                full_engine_->bind(nm, bufs_.back().ptr);
        }
    }
}

double ChunkPipeline::_percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

PipelineResult ChunkPipeline::run(int n_warmup, int n_iters) {
    size_t n_chunks = chunks_.size();

    // ── Full engine warmup + timing ───────────────────────────────────────────
    std::vector<double> full_gpu_ms;
    if (full_engine_) {
        for (int i = 0; i < n_warmup; ++i) full_engine_->execute(stream_);
        cudaStreamSynchronize(stream_);

        CudaTimer t;
        for (int i = 0; i < n_iters; ++i) {
            t.record_start(stream_);
            full_engine_->execute(stream_);
            t.record_stop(stream_);
            full_gpu_ms.push_back(t.elapsed_ms());
        }
    }

    // ── Chunk pipeline warmup + timing ────────────────────────────────────────
    // Warmup.
    for (int w = 0; w < n_warmup; ++w) {
        for (size_t c = 0; c < n_chunks; ++c) chunks_[c].execute(stream_);
        cudaStreamSynchronize(stream_);
    }

    // Per-chunk timers (pre-allocated — no alloc in hot loop).
    std::vector<CudaTimer> timers(n_chunks);
    std::vector<std::vector<double>> chunk_gpu(n_chunks);
    std::vector<std::vector<double>> chunk_cpu(n_chunks);
    std::vector<double> total_gpu(n_iters);

    CudaTimer t_total;

    for (int it = 0; it < n_iters; ++it) {
        t_total.record_start(stream_);
        for (size_t c = 0; c < n_chunks; ++c) {
            auto wall0 = wall_now();
            timers[c].record_start(stream_);
            chunks_[c].execute(stream_);
            timers[c].record_stop(stream_);
            auto wall1 = wall_now();
            chunk_cpu[c].push_back(wall_ms(wall0, wall1));
        }
        t_total.record_stop(stream_);
        total_gpu[it] = t_total.elapsed_ms();

        // Collect per-chunk GPU times (synchronizes internally).
        for (size_t c = 0; c < n_chunks; ++c)
            chunk_gpu[c].push_back(timers[c].elapsed_ms());
    }

    // ── Build result ──────────────────────────────────────────────────────────
    PipelineResult res;
    res.n_iters = n_iters;

    if (!full_gpu_ms.empty()) {
        double sum = 0; for (auto v : full_gpu_ms) sum += v;
        res.full_engine_gpu_mean_ms = sum / full_gpu_ms.size();
        res.full_engine_gpu_p99_ms  = _percentile(full_gpu_ms, 99.0);
        res.full_engine_gpu_max_ms  = *std::max_element(full_gpu_ms.begin(), full_gpu_ms.end());
    }

    {
        double sum = 0; for (auto v : total_gpu) sum += v;
        res.total_chunked_gpu_mean_ms = sum / total_gpu.size();
        res.total_chunked_gpu_p99_ms  = _percentile(total_gpu, 99.0);
        res.total_chunked_gpu_max_ms  = *std::max_element(total_gpu.begin(), total_gpu.end());
    }

    for (size_t c = 0; c < n_chunks; ++c) {
        ChunkStats cs;
        cs.id = (int)c;
        double gs = 0, cs2 = 0;
        for (auto v : chunk_gpu[c]) gs += v;
        for (auto v : chunk_cpu[c]) cs2 += v;
        cs.gpu_mean_ms = gs / chunk_gpu[c].size();
        cs.cpu_mean_ms = cs2 / chunk_cpu[c].size();
        cs.gpu_p99_ms  = _percentile(chunk_gpu[c], 99.0);
        cs.cpu_p99_ms  = _percentile(chunk_cpu[c], 99.0);
        cs.gpu_max_ms  = *std::max_element(chunk_gpu[c].begin(), chunk_gpu[c].end());
        cs.cpu_max_ms  = *std::max_element(chunk_cpu[c].begin(), chunk_cpu[c].end());
        res.chunks.push_back(cs);
    }

    return res;
}
