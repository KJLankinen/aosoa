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
#include <sycl/sycl.hpp>

namespace aosoa {
template <sycl::usm::alloc kind> struct SyclAllocator {
    const sycl::queue &queue;
    const sycl::property_list prop_list;

    SyclAllocator(const sycl::queue &queue,
                  const sycl::property_list prop_list = {})
        : queue(queue), prop_list(prop_list) {}

    void *operator()(size_t bytes) {
        return sycl::malloc(bytes, queue, kind, prop_list);
    }
};

struct SyclDeallocator {
    const sycl::queue &queue;

    SyclDeallocator(const sycl::queue &queue) : queue(queue) {}

    void operator()(void *ptr) noexcept { sycl::free(ptr, queue); }
};

template <bool SYNC> struct SyclMemcpy {
    sycl::queue &queue;
    sycl::event event;

    SyclMemcpy(sycl::queue &queue) : queue(queue) {}

    void operator()(void *dst, const void *src, size_t bytes,
                    bool synchronize = false) noexcept {
        event = queue.memcpy(dst, src, bytes);
        if constexpr (SYNC) {
            event.wait();
        } else {
            if (synchronize) {
                event.wait();
            }
        }
    }
};

template <bool SYNC> struct SyclMemset {
    sycl::queue &queue;
    sycl::event event;

    SyclMemset(sycl::queue &queue) : queue(queue) {}

    void operator()(void *ptr, int pattern, size_t bytes,
                    bool synchronize = false) noexcept {
        event = queue.memset(ptr, pattern, bytes);
        if constexpr (SYNC) {
            event.wait();
        } else {
            if (synchronize) {
                event.wait();
            }
        }
    }
};

template <sycl::usm::alloc kind, bool SYNC> struct SyclMemoryOperations {
    static constexpr bool host_access_requires_copy =
        kind == sycl::usm::alloc::device;
    SyclAllocator<kind> allocate;
    SyclDeallocator deallocate;
    SyclMemcpy<SYNC> memcpy;
    SyclMemset<SYNC> memset;

    SyclMemoryOperations(sycl::queue &queue)
        : allocate(queue), deallocate(queue), memcpy(queue), memset(queue) {}
};

using SyclDeviceMemoryOperations =
    SyclMemoryOperations<sycl::usm::alloc::device, true>;
using SyclDeviceMemoryOperationsAsync =
    SyclMemoryOperations<sycl::usm::alloc::device, false>;
using SyclHostMemoryOperations =
    SyclMemoryOperations<sycl::usm::alloc::host, true>;
using SyclHostMemoryOperationsAsync =
    SyclMemoryOperations<sycl::usm::alloc::host, false>;
using SyclSharedMemoryOperations =
    SyclMemoryOperations<sycl::usm::alloc::shared, true>;
using SyclSharedMemoryOperationsAsync =
    SyclMemoryOperations<sycl::usm::alloc::shared, false>;
} // namespace aosoa
