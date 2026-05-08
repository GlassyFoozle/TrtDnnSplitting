#include "buffer_pool.hpp"

DeviceBuffer::DeviceBuffer(size_t n_bytes) : bytes(n_bytes) {
    if (n_bytes == 0) return;
    cudaError_t err = cudaMalloc(&ptr, n_bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
}

void DeviceBuffer::free() {
    if (ptr) { cudaFree(ptr); ptr = nullptr; bytes = 0; }
}

DeviceBuffer::~DeviceBuffer() { free(); }
