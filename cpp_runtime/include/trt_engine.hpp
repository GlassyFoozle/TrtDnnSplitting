#pragma once
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>

// Minimal TRT logger — prints only errors.
class SilentLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// Owns one TRT engine + execution context loaded from a serialised .engine file.
class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path, nvinfer1::ILogger& logger);
    ~TrtEngine();

    // Not copyable, movable.
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;
    TrtEngine(TrtEngine&&) noexcept;
    TrtEngine& operator=(TrtEngine&&) noexcept;

    // Number of I/O tensors.
    int32_t n_io() const;
    // Name of i-th I/O tensor.
    const char* io_name(int32_t i) const;
    // Volume (number of floats) of a named tensor.
    size_t tensor_volume(const char* name) const;

    // Bind device pointer to a named tensor (must be called before execute).
    void bind(const char* name, void* device_ptr);

    // Execute on stream; returns false on error.
    bool execute(cudaStream_t stream);

    nvinfer1::ICudaEngine* engine() { return engine_; }
    nvinfer1::IExecutionContext* context() { return context_; }

private:
    nvinfer1::IRuntime*          runtime_  = nullptr;
    nvinfer1::ICudaEngine*       engine_   = nullptr;
    nvinfer1::IExecutionContext* context_  = nullptr;
};
