/*
    MIT License

    Copyright (c) 2024 Juhana Lankinen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#pragma once

#include <cstdlib>
#include <cuda_runtime.h>

namespace aosoa {
struct CudaAllocator {
    cudaError_t previous_result = {};
    void *operator()(size_t bytes) noexcept {
        void *ptr = nullptr;
        previous_result = cudaMalloc(&ptr, bytes);
        return ptr;
    }
};

struct CudaDeallocator {
    cudaError_t previous_result = {};
    void operator()(void *ptr) noexcept { previous_result = cudaFree(ptr); }
};

template <bool SYNC> struct CudaMemcpy {
    cudaError_t previous_result = {};
    cudaStream_t stream = {};

    CudaMemcpy(cudaStream_t stream) : stream(stream) {}

    void operator()(void *dst, const void *src, size_t bytes,
                    bool synchronize = false) noexcept {
        previous_result =
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
        if constexpr (SYNC) {
            previous_result = cudaDeviceSynchronize();
        } else {
            if (synchronize) {
                previous_result = cudaDeviceSynchronize();
            }
        }
    }
};

template <bool SYNC> struct CudaMemset {
    cudaError_t previous_result = {};
    cudaStream_t stream = {};

    CudaMemset(cudaStream_t stream) : stream(stream) {}

    void operator()(void *dst, int pattern, size_t bytes,
                    bool synchronize = false) noexcept {
        previous_result = cudaMemsetAsync(dst, pattern, bytes, stream);
        if constexpr (SYNC) {
            previous_result = cudaDeviceSynchronize();
        } else {
            if (synchronize) {
                previous_result = cudaDeviceSynchronize();
            }
        }
    }
};

template <bool SYNC> struct CudaMemoryOperations {
    static constexpr bool host_access_requires_copy = true;
    CudaAllocator allocate;
    CudaDeallocator deallocate;
    CudaMemcpy<SYNC> memcpy;
    CudaMemset<SYNC> memset;

    CudaMemoryOperations(cudaStream_t stream)
        : allocate(), deallocate(), memcpy(stream), memset(stream) {}
};

using CudaDeviceMemoryOperations = CudaMemoryOperations<true>;
using CudaDeviceMemoryOperationsAsync = CudaMemoryOperations<false>;
} // namespace aosoa
