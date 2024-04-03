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

#include "compile_time_string.h"
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
        constexpr static size_t index = FromStrings<Candidates::name...>::index;
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
    using Type = typename NthType<i, typename Variables::Type...>::Type;
};

} // namespace aosoa
