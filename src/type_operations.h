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

#include "compile_time_string.h"
#include "variable.h"
#include <cstddef>
#include <type_traits>

namespace aosoa {
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

// Find the index of a string
template <CompileTimeString MatchStr> struct Find {
    // ... from a pack of strings
    template <CompileTimeString... Candidates> struct FromStrings {
      private:
        template <size_t N> consteval static size_t find() { return ~0ul; }

        template <size_t N, CompileTimeString Head, CompileTimeString... Tail>
        consteval static size_t find() {
            if constexpr (MatchStr == Head) {
                return N;
            } else {
                return find<N + 1, Tail...>();
            }
        }

      public:
        constexpr static size_t index = find<0, Candidates...>();
    };

    // ... from a pack of Variables
    template <typename... Candidates> struct FromVariables {
        constexpr static size_t index =
            FromStrings<VariableTraits<Candidates>::name...>::index;
    };
};

// - Check if the string is in the parameter pack
template <CompileTimeString MatchStr> struct Is {
    template <CompileTimeString... Strings> struct ContainedIn {
        constexpr static bool value =
            Find<MatchStr>::template FromStrings<Strings...>::index !=
            Find<MatchStr>::template FromStrings<>::index;
    };
};

// - Get index and type corresponding to Cts
template <CompileTimeString Cts, typename... Variables> struct GetType {
    static constexpr size_t i =
        Find<Cts>::template FromVariables<Variables...>::index;
    using Type =
        typename NthType<i, typename VariableTraits<Variables>::Type...>::Type;
};

} // namespace aosoa
