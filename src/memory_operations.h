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

namespace detail {
// - These are used by StructureOfArrays to perform
//   - memory allocation at construction
//   - deallocation at unique_ptr destruction
//   - memcpy and memset between pointers
//   - update of the remote accessor
template <bool HostAccessRequiresCopy, typename Allocate, typename Free,
          typename Copy, typename Set>
struct MemoryOperations {
    using Deallocate = Free;
    static constexpr bool host_access_requires_copy = HostAccessRequiresCopy;
    Allocate allocate = {};
    Copy memcpy = {};
    Set memset = {};
};
} // namespace detail
