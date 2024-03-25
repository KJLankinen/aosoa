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

#include <algorithm>

namespace detail {
// ==== Compile time string ====
template <size_t N> struct CompileTimeString {
    char str[N + 1] = {};

    consteval CompileTimeString(const char (&s)[N + 1]) {
        std::copy_n(s, N + 1, str);
    }

    consteval bool operator==(const CompileTimeString<N> rhs) const {
        return std::equal(rhs.str, rhs.str + N, str);
    }

    template <size_t M>
    consteval bool operator==(const CompileTimeString<M>) const {
        return false;
    }

    template <size_t M>
    consteval CompileTimeString<N + M>
    operator+(const CompileTimeString<M> rhs) const {
        char out_str[N + 1 + M] = {};
        std::copy_n(str, N, out_str);
        std::copy_n(rhs.str, M + 1, out_str + N);
        return CompileTimeString<N + M>(out_str);
    }

    consteval char operator[](size_t i) const { return str[i]; }
    consteval char *data() const { return str; }
    consteval size_t size() const { return N - 1; }
};

template <size_t N, size_t M>
consteval bool operator==(const char (&lhs)[N], CompileTimeString<M> rhs) {
    return CompileTimeString<N - 1>(lhs) == rhs;
}

template <size_t N, size_t M>
consteval bool operator==(CompileTimeString<N> lhs, const char (&rhs)[M]) {
    return lhs == CompileTimeString<M - 1>(rhs);
}

template <size_t N, size_t M>
consteval auto operator+(const char (&lhs)[N], CompileTimeString<M> rhs) {
    return CompileTimeString<N - 1>(lhs) + rhs;
}

template <size_t N, size_t M>
consteval auto operator+(CompileTimeString<N> lhs, const char (&rhs)[M]) {
    return lhs + CompileTimeString<M - 1>(rhs);
}

// Deduction guide
template <size_t N>
CompileTimeString(const char (&)[N]) -> CompileTimeString<N - 1>;

template <CompileTimeString Cts> constexpr auto operator""_cts() { return Cts; }
} // namespace detail
