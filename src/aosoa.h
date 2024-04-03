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

#include <algorithm>
#include <vector>

#include "accessor.h"
#include "c_memory_operations.h"
#include "compile_time_string.h"
#include "type_operations.h"

namespace aosoa {
// - Used on the host to manage the accessor and the memory
template <size_t MIN_ALIGN, typename MemOps, typename... Variables>
struct StructureOfArrays {
    using ThisAccessor = Accessor<MIN_ALIGN, Variables...>;
    using FullRow = ThisAccessor::FullRow;

  private:
    static_assert(FullRow::unique_names,
                  "StructureOfArrays has clashing names");

    using CSoa = StructureOfArrays<MIN_ALIGN, CMemoryOperations, Variables...>;

    template <size_t MA, typename M, typename... V>
    friend struct StructureOfArrays;

    MemOps &memory_ops;
    const size_t max_num_elements;
    std::unique_ptr<uint8_t, decltype(MemOps::deallocate)> memory;
    ThisAccessor local_accessor;

  public:
    [[nodiscard]] static size_t getMemReq(size_t n) {
        return ThisAccessor::getMemReq(n);
    }

    StructureOfArrays(MemOps &mem_ops, size_t n,
                      ThisAccessor *accessor = nullptr)
        : memory_ops(mem_ops), max_num_elements(n),
          memory(static_cast<uint8_t *>(
                     memory_ops.allocate(getMemReq(max_num_elements))),
                 memory_ops.deallocate),
          local_accessor(max_num_elements, static_cast<void *>(memory.get())) {
        updateAccessor(accessor);
        memory_ops.memset(static_cast<void *>(memory.get()), 0,
                          getMemReq(max_num_elements));
    }

    // If the number of elements is very large, use the above constructor and
    // initialize the values in place to avoid running out of memory
    StructureOfArrays(MemOps &mem_ops, const std::vector<FullRow> &rows,
                      ThisAccessor *accessor = nullptr)
        : StructureOfArrays(mem_ops, rows.size(), accessor) {
        if (memory_ops.host_access_requires_copy) {
            CMemoryOperations c_mem_ops;
            CSoa host_soa(c_mem_ops, rows);

            memory_ops.memcpy(local_accessor.template get<0>(),
                              host_soa.local_accessor.template get<0>(),
                              getAlignedBlockSize());
        } else {
            size_t i = 0;
            for (const auto &row : rows) {
                local_accessor.set(i++, row);
            }
        }
    }

    void decreaseBy(size_t n, ThisAccessor *accessor = nullptr) {
        local_accessor.size() -= std::min(n, local_accessor.size());
        updateAccessor(accessor);
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2>
    void swap(ThisAccessor *accessor = nullptr) {
        using G1 = GetType<Cts1, Variables...>;
        using G2 = GetType<Cts2, Variables...>;

        static_assert(IsSame<typename G1::Type, typename G2::Type>::value,
                      "Mismatched types for swap");

        std::swap(local_accessor.template getRef<Cts1>(),
                  local_accessor.template getRef<Cts2>());

        updateAccessor(accessor);
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2,
              CompileTimeString Cts3, CompileTimeString... Tail>
    void swap(ThisAccessor *accessor = nullptr) {
        swap<Cts1, Cts2>();
        swap<Cts3, Tail...>();
        updateAccessor(accessor);
    }

    void updateAccessor(ThisAccessor *accessor) const {
        updateAccessor(memory_ops.memcpy, accessor);
    }

    template <typename F>
    void updateAccessor(F f, ThisAccessor *accessor) const {
        if (accessor != nullptr) {
            f(static_cast<void *>(accessor),
              static_cast<const void *>(&local_accessor), sizeof(ThisAccessor));
        }
    }

    std::vector<FullRow> getRows() const {
        // This is an expensive function: it copies all the memory twice, if the
        // memory recides on device. The first copy is the raw data from device
        // to host, the second is from soa (= current) layout to aos (= vector
        // of FullRow) layout
        if constexpr (MemOps::host_access_requires_copy) {
            // Create this structure backed by host memory, then call it's
            // version of this function

            static_assert(
                !CMemoryOperations::host_access_requires_copy,
                "C memory operations should not require copy for host access");

            CMemoryOperations c_mem_ops;
            CSoa host_soa(c_mem_ops, max_num_elements);
            memory_ops.memcpy(host_soa.local_accessor.template get<0>(),
                              local_accessor.template get<0>(),
                              getAlignedBlockSize(), true);
            host_soa.local_accessor.size() = local_accessor.size();

            return host_soa.getRows();
        } else {
            // Just convert to a vector of rows
            std::vector<FullRow> rows(local_accessor.size());
            std::generate(rows.begin(), rows.end(), [i = 0ul, this]() mutable {
                return local_accessor.get(i++);
            });
            return rows;
        }
    }

    const ThisAccessor &getAccess() const { return local_accessor; }
    ThisAccessor &getAccess() { return local_accessor; }

    // Internal to internal
    template <CompileTimeString DstName, CompileTimeString SrcName>
    void memcpy() const {
        memcpy<DstName, SrcName>(memory_ops.memcpy);
    }

    // Internal to internal
    template <CompileTimeString DstName, CompileTimeString SrcName, typename F>
    void memcpy(F f) const {
        using Dst = GetType<DstName, Variables...>;
        using Src = GetType<SrcName, Variables...>;
        static_assert(IsSame<typename Dst::Type, typename Src::Type>::value,
                      "Mismatched types for memcpy");
        static_assert(DstName != SrcName, "DstName and SrcName are the same");

        f(local_accessor.template get<DstName>(),
          local_accessor.template get<SrcName>(), getMemReq<DstName>());
    }

    // External to internal
    template <CompileTimeString DstName, typename SrcType>
    void memcpy(const SrcType *src) const {
        memcpy<DstName>(memory_ops.memcpy, src);
    }

    // External to internal
    template <CompileTimeString DstName, typename SrcType, typename F>
    void memcpy(F f, const SrcType *src) const {
        using Dst = GetType<DstName, Variables...>;
        static_assert(IsSame<typename Dst::Type, SrcType>::value,
                      "Mismatched types for memcpy");
        f(local_accessor.template get<DstName>(),
          static_cast<const void *>(src), getMemReq<DstName>());
    }

    // Internal to external
    template <CompileTimeString SrcName, typename DstType>
    void memcpy(DstType *dst) const {
        memcpy<SrcName>(memory_ops.memcpy, dst);
    }

    // Internal to external
    template <CompileTimeString SrcName, typename DstType, typename F>
    void memcpy(F f, DstType *dst) const {
        using Src = GetType<SrcName, Variables...>;
        static_assert(IsSame<typename Src::Type, DstType>::value,
                      "Mismatched types for memcpy");

        f(static_cast<void *>(dst), local_accessor.template get<SrcName>(),
          getMemReq<SrcName>());
    }

    // Memset column
    template <CompileTimeString DstName> void memset(int pattern) const {
        memset<DstName>(memory_ops.memset, pattern);
    }

    template <CompileTimeString DstName, typename F>
    void memset(F f, int pattern) const {
        f(local_accessor.template get<DstName>(), pattern,
          getMemReq<DstName>());
    }

  private:
    template <CompileTimeString Cts> [[nodiscard]] size_t getMemReq() const {
        return local_accessor.size() *
               sizeof(typename GetType<Cts, Variables...>::Type);
    }

    [[nodiscard]] uintptr_t getAlignedBlockSize() const {
        return getMemReq(max_num_elements) - ThisAccessor::getAlignment();
    }
};
} // namespace aosoa
