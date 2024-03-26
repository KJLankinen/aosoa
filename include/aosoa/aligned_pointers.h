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

#include <memory>

#include "definitions.h"
#include "variable.h"

namespace aosoa {
// - An array of void *
// - Each pointer is aligned by the maximum alignment of all the types in
//   Variables... and MIN_ALIGN
template <size_t MIN_ALIGN, typename... Variables> struct AlignedPointers {
    static constexpr size_t size = sizeof...(Variables);

  private:
    void *pointers[size] = {};

  public:
    // Construct a target for copy
    HOST DEVICE AlignedPointers() {}

    AlignedPointers(size_t n, void *ptr) {
        alignPointers<size, 0, typename VariableTraits<Variables>::Type...>(
            ptr, pointers, ~0ul, n);
    }

    HOST DEVICE constexpr auto &operator[](size_t i) { return pointers[i]; }

    HOST DEVICE constexpr const auto &operator[](size_t i) const {
        return pointers[i];
    }

    // ==== Construction related functionality ====
    [[nodiscard]] static size_t getMemReq(size_t n) {
        // Get proper begin alignment: the strictest (largest) alignment
        // requirement between all the types and MIN_ALIGN
        alignas(getAlignment()) uint8_t dummy = 0;
        constexpr size_t max_size = ~size_t(0);
        size_t space = max_size;
        void *pointers[size] = {};
        alignPointers<size, 0, typename VariableTraits<Variables>::Type...>(
            static_cast<void *>(&dummy), pointers, std::move(space), n);

        const size_t num_bytes = max_size - space;
        // Require a block of (M + 1) * alignment bytes, where M is an integer.
        // The +1 is for manual alignment, if the memory allocation doesn't have
        // a strict enough alignment requirement.
        return num_bytes + bytesMissingFromAlignment(num_bytes) +
               getAlignment();
    }

    [[nodiscard]] consteval static size_t getAlignment() {
        static_assert((MIN_ALIGN & (MIN_ALIGN - 1)) == 0,
                      "MIN_ALIGN isn't a power of two");
        // Aligned by the strictest (largest) alignment requirement between all
        // the types and the MIN_ALIGN template argument
        // N.B. A statement with multiple alignas declarations is supposed to
        // pick the strictest one, but GCC for some reason picks the last one
        // that is applied...
        // If it weren't for that bug, could use:
        // struct alignas(MIN_ALIGN) alignas(typename
        // VariableTraits<Variables>::Type...) Aligned {}; return
        // alignof(Aligned);
        struct alignas(MIN_ALIGN) MinAligned {};
        return maxAlign<MinAligned,
                        typename VariableTraits<Variables>::Type...>();
    }

  private:
    [[nodiscard]] constexpr static size_t bytesMissingFromAlignment(size_t n) {
        return (getAlignment() - bytesOverAlignment(n)) & (getAlignment() - 1);
    }

    [[nodiscard]] constexpr static size_t bytesOverAlignment(size_t n) {
        return n & (getAlignment() - 1);
    }

    template <typename T, typename... Ts> consteval static size_t maxAlign() {
        if constexpr (sizeof...(Ts) == 0) {
            return alignof(T);
        } else {
            return std::max(alignof(T), maxAlign<Ts...>());
        }
    }

    template <size_t N, size_t>
    static void alignPointers(void *ptr, void *(&)[N], size_t &&space, size_t) {
        // Align the end of last pointer to the getAlignment() byte boundary so
        // the memory requirement is a multiple of getAlignment()
        if (ptr) {
            ptr = std::align(getAlignment(), 1, ptr, space);
        }
    }

    template <size_t N, size_t I, typename Head, typename... Tail>
    static void alignPointers(void *ptr, void *(&pointers)[N], size_t &&space,
                              size_t n) {
        constexpr size_t size_of_type = sizeof(Head);
        if (ptr) {
            ptr = std::align(getAlignment(), size_of_type, ptr, space);
            if (ptr) {
                pointers[I] = ptr;
                ptr = static_cast<void *>(static_cast<Head *>(ptr) + n);
                space -= size_of_type * n;

                alignPointers<N, I + 1, Tail...>(ptr, pointers,
                                                 std::move(space), n);
            }
        }
    }
};
} // namespace aosoa
