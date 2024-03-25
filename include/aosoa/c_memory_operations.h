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
#include <cstring>

#include "memory_operations.h"

namespace aosoa {
struct CAllocator {
    void *operator()(size_t bytes) const noexcept { return std::malloc(bytes); }
};

struct CDeallocator {
    void operator()(void *ptr) const noexcept { std::free(ptr); }
};

struct CMemcpy {
    void operator()(void *dst, const void *src, size_t bytes) const noexcept {
        std::memcpy(dst, src, bytes);
    }
};

struct CMemset {
    void operator()(void *dst, int pattern, size_t bytes) const noexcept {
        std::memset(dst, pattern, bytes);
    }
};

typedef MemoryOperations<false, CAllocator, CDeallocator, CMemcpy, CMemset>
    CMemoryOperations;
} // namespace aosoa
