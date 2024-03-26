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

#include "aligned_pointers.h"
#include "compile_time_string.h"
#include "definitions.h"
#include "row.h"
#include "type_operations.h"
#include "variable.h"

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
    HOST DEVICE [[nodiscard]] auto get() const {
        using G = GetType<Cts, Variables...>;
        return static_cast<G::Type *>(pointers[G::i]);
    }

    template <size_t I> HOST DEVICE [[nodiscard]] auto get() const {
        using Type =
            typename NthType<I,
                             typename VariableTraits<Variables>::Type...>::Type;
        return static_cast<Type *>(pointers[I]);
    }

    template <CompileTimeString Cts, size_t I>
    HOST DEVICE [[nodiscard]] auto get() const {
        return get<Cts>()[I];
    }

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] auto get(size_t i) const {
        return get<Cts>()[i];
    }

    HOST DEVICE [[nodiscard]] FullRow get(size_t i) const {
        return toRow<Variables...>(i);
    }

    template <CompileTimeString Cts> HOST DEVICE [[nodiscard]] void *&getRef() {
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
        using Traits = VariableTraits<Head>;
        if constexpr (sizeof...(Tail) > 0) {
            return Row<Head, Tail...>(get<Traits::name>(i), toRow<Tail...>(i));
        } else {
            return Row<Head>(get<Traits::name>(i));
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, const FullRow &row) const {
        using Traits = VariableTraits<Head>;
        set<Traits::name, typename Traits::Type>(
            i, row.template get<Traits::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, row);
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, FullRow &&row) const {
        using Traits = VariableTraits<Head>;
        set<Traits::name, typename Traits::Type>(
            i, row.template get<Traits::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, std::move(row));
        }
    }
};
} // namespace aosoa
