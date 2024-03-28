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

#include "c_memory_operations.h"
#include "common.h"

using PixelSoa = Soa<CMemoryOperations>;
using Pixels = Acc<CMemoryOperations>;

void init(Pixels *pixels) {
    for (size_t i = 0; i < pixels->size(); i++) {
        computeColor(i, pixels);
    }
}

int main(int , char **) {
    aosoa::CMemoryOperations memory_ops{
        CAllocator{}, CMemcpy{}, CMemset{}};

    Pixels pixels = {};
    PixelSoa pixel_soa(memory_ops, num_pixels, &pixels);
    init(&pixels);
    writePixelsToFile(pixel_soa, "pixels.png");

    return 0;
}
