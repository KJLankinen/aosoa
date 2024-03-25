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

#include <cstddef>
#include <type_traits>

namespace detail {
// Check if types are equal
template <typename T, typename U> struct IsSame {
    constexpr static bool value = false;
};

template <typename T> struct IsSame<T, T> {
    constexpr static bool value = true;
};

// - Get the Nth type from a parameter pack of types
template <size_t N, typename... Types> struct NthType {
  private:
    template <size_t I, typename Head, typename... Tail>
    consteval static auto ofType() {
        if constexpr (I == N) {
            return Head{};
        } else {
            return ofType<I + 1, Tail...>();
        }
    }

  public:
    using Type = std::invoke_result_t<decltype(ofType<0, Types...>)>;
};
} // namespace detail
