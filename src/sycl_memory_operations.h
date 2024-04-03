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
#include <sycl/sycl.hpp>

#include "memory_operations.h"

namespace aosoa {
template <sycl::usm::alloc kind> struct SyclAllocator {
    const sycl::queue &queue;
    const sycl::property_list prop_list;

    SyclAllocator(const sycl::queue &queue,
                  const sycl::property_list &prop_list = {})
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

using SyclDeviceAllocator = SyclAllocator<sycl::usm::alloc::device>;
using SyclHostAllocator = SyclAllocator<sycl::usm::alloc::host>;
using SyclSharedAllocator = SyclAllocator<sycl::usm::alloc::shared>;
using SyclMemcpyAsync = SyclMemcpy<false>;
using SyclMemsetAsync = SyclMemset<false>;

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
