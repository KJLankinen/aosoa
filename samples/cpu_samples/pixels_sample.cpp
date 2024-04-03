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
    aosoa::CMemoryOperations memory_ops{CAllocator{}, CDeallocator{}, CMemcpy{},
                                        CMemset{}};

    Pixels pixels = {};
    PixelSoa pixel_soa(memory_ops, num_pixels, &pixels);
    init(&pixels);
    writePixelsToFile(pixel_soa, "pixels_cpu.png");

    return 0;
}
