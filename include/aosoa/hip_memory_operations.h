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
#include <hip/hip_runtime.h>

namespace aosoa {
struct HipAllocator {
    hipError_t previous_result = {};
    void *operator()(size_t bytes) noexcept {
        void *ptr = nullptr;
        previous_result = hipMalloc(&ptr, bytes);
        return ptr;
    }
};

struct HipDeallocator {
    hipError_t previous_result = {};
    void operator()(void *ptr) noexcept { previous_result = hipFree(ptr); }
};

template <bool SYNC> struct HipMemcpy {
    hipError_t previous_result = {};
    hipStream_t stream = {};

    HipMemcpy(hipStream_t stream) : stream(stream) {}

    void operator()(void *dst, const void *src, size_t bytes,
                    bool synchronize = false) noexcept {
        previous_result =
            hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream);
        if constexpr (SYNC) {
            previous_result = hipDeviceSynchronize();
        } else {
            if (synchronize) {
                previous_result = hipDeviceSynchronize();
            }
        }
    }
};

template <bool SYNC> struct HipMemset {
    hipError_t previous_result = {};
    hipStream_t stream = {};

    HipMemset(hipStream_t stream) : stream(stream) {}

    void operator()(void *dst, int pattern, size_t bytes,
                    bool synchronize = false) noexcept {
        previous_result = hipMemsetAsync(dst, pattern, bytes, stream);
        if constexpr (SYNC) {
            previous_result = hipDeviceSynchronize();
        } else {
            if (synchronize) {
                previous_result = hipDeviceSynchronize();
            }
        }
    }
};

template <bool SYNC> struct HipMemoryOperations {
    static constexpr bool host_access_requires_copy = true;
    HipAllocator allocate;
    HipDeallocator deallocate;
    HipMemcpy<SYNC> memcpy;
    HipMemset<SYNC> memset;

    HipMemoryOperations(hipStream_t stream)
        : allocate(), deallocate(), memcpy(stream), memset(stream) {}
};

using HipDeviceMemoryOperations = HipMemoryOperations<true>;
using HipDeviceMemoryOperationsAsync = HipMemoryOperations<false>;
} // namespace aosoa
