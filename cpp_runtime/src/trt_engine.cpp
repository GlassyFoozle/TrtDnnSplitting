#include "trt_engine.hpp"
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <cstdint>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
    return v;
}

TrtEngine::TrtEngine(const std::string& path, nvinfer1::ILogger& logger) {
    auto buf = read_file(path);
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_ = runtime_->deserializeCudaEngine(buf.data(), buf.size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + path);
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("createExecutionContext failed: " + path);
}

TrtEngine::~TrtEngine() {
    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy();  engine_  = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
}

TrtEngine::TrtEngine(TrtEngine&& o) noexcept
    : runtime_(o.runtime_), engine_(o.engine_), context_(o.context_)
{
    o.runtime_ = nullptr; o.engine_ = nullptr; o.context_ = nullptr;
}

TrtEngine& TrtEngine::operator=(TrtEngine&& o) noexcept {
    if (this != &o) {
        if (context_) context_->destroy();
        if (engine_)  engine_->destroy();
        if (runtime_) runtime_->destroy();
        runtime_ = o.runtime_; engine_ = o.engine_; context_ = o.context_;
        o.runtime_ = nullptr; o.engine_ = nullptr; o.context_ = nullptr;
    }
    return *this;
}

int32_t TrtEngine::n_io() const { return engine_->getNbIOTensors(); }

const char* TrtEngine::io_name(int32_t i) const { return engine_->getIOTensorName(i); }

size_t TrtEngine::tensor_volume(const char* name) const {
    return volume(engine_->getTensorShape(name));
}

void TrtEngine::bind(const char* name, void* ptr) {
    if (!context_->setTensorAddress(name, ptr))
        throw std::runtime_error(std::string("setTensorAddress failed for: ") + name);
}

bool TrtEngine::execute(cudaStream_t stream) {
    return context_->enqueueV3(stream);
}
