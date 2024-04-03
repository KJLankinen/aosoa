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

#include "aligned_pointers.h"
#include "compile_time_string.h"
#include "definitions.h"
#include "row.h"
#include "type_operations.h"

namespace aosoa {
// - Used to access data stored in the AlignedPointers array
template <size_t MIN_ALIGN, typename... Variables> struct Accessor {
    using FullRow = Row<Variables...>;

  private:
    using Pointers = AlignedPointers<MIN_ALIGN, Variables...>;
    size_t num_elements = 0;
    Pointers pointers;

  public:
    [[nodiscard]] static size_t getMemReq(size_t n) {
        return Pointers::getMemReq(n);
    }

    [[nodiscard]] consteval static size_t getAlignment() {
        return Pointers::getAlignment();
    }

    HOST DEVICE Accessor() {}

    Accessor(size_t n, void *ptr) : num_elements(n), pointers(n, ptr) {}

    template <CompileTimeString Cts>
    [[nodiscard]] HOST DEVICE auto get() const {
        using G = GetType<Cts, Variables...>;
        return static_cast<typename G::Type *>(pointers[G::i]);
    }

    template <size_t I> [[nodiscard]] HOST DEVICE auto get() const {
        using Type = typename NthType<I, typename Variables::Type...>::Type;
        return static_cast<Type *>(pointers[I]);
    }

    template <CompileTimeString Cts, size_t I>
    [[nodiscard]] HOST DEVICE auto get() const {
        return get<Cts>()[I];
    }

    template <CompileTimeString Cts>
    [[nodiscard]] HOST DEVICE auto get(size_t i) const {
        return get<Cts>()[i];
    }

    [[nodiscard]] HOST DEVICE FullRow get(size_t i) const {
        return toRow<Variables...>(i);
    }

    template <CompileTimeString Cts> [[nodiscard]] HOST DEVICE void *&getRef() {
        using G = GetType<Cts, Variables...>;
        return pointers[G::i];
    }

    template <CompileTimeString Cts, typename T>
    HOST DEVICE void set(size_t i, T value) const {
        get<Cts>()[i] = value;
    }

    HOST DEVICE void set(size_t i, const FullRow &t) const {
        fromRow<Variables...>(i, t);
    }

    HOST DEVICE void set(size_t i, FullRow &&t) const {
        fromRow<Variables...>(i, std::move(t));
    }

    HOST DEVICE size_t &size() { return num_elements; }
    HOST DEVICE const size_t &size() const { return num_elements; }

  private:
    template <typename Head, typename... Tail>
    [[nodiscard]] HOST DEVICE auto toRow(size_t i) const {
        if constexpr (sizeof...(Tail) > 0) {
            return Row<Head, Tail...>(get<Head::name>(i), toRow<Tail...>(i));
        } else {
            return Row<Head>(get<Head::name>(i));
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, const FullRow &row) const {
        set<Head::name, typename Head::Type>(i, row.template get<Head::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, row);
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, FullRow &&row) const {
        set<Head::name, typename Head::Type>(i, row.template get<Head::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, std::move(row));
        }
    }
};
} // namespace aosoa
