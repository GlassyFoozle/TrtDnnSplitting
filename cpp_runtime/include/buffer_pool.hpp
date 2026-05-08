#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

// Manages a single pre-allocated device buffer.
struct DeviceBuffer {
    void*  ptr    = nullptr;
    size_t bytes  = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
        o.ptr = nullptr; o.bytes = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { free(); ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; o.bytes = 0; }
        return *this;
    }

    void free();
};
