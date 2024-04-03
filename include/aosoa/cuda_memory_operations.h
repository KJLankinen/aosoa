/*
    aosoa
    Copyright (C) 2024  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
