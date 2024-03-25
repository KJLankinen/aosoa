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

namespace detail {
// - Bind a type and a CompileTimeString together
template <typename, CompileTimeString> struct Variable {};

// - Extract the name and the type from a Variable<Type, Name>
template <typename> struct VariableTraits;
template <typename T, CompileTimeString Cts>
struct VariableTraits<Variable<T, Cts>> {
    using Type = T;
    static constexpr CompileTimeString name = Cts;
};
} // namespace detail
