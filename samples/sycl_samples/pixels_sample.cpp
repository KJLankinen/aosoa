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

#include "common.h"
#include "sycl_memory_operations.h"

using namespace aosoa;

using MemOp = SyclDeviceMemoryOperationsAsync;
using PixelSoa = Soa<MemOp>;
using Pixels = Acc<MemOp>;

int main(int , char **) {
    sycl::device d(sycl::default_selector_v);
    sycl::property_list q_prop{sycl::property::queue::in_order()};
    sycl::queue queue{d, q_prop};

    MemOp memory_ops(queue);

    Pixels *d_pixels = sycl::malloc_device<Pixels>(1, queue);
    PixelSoa pixel_soa(memory_ops, num_pixels, d_pixels);

    [[maybe_unused]] auto event = queue.submit([&](sycl::handler &h) {
        h.parallel_for(num_pixels,
                       [=](auto idx) { computeColor(idx, d_pixels); });
    });
    writePixelsToFile(pixel_soa, "pixels_sycl.png");
    sycl::free(d_pixels, queue);

    return 0;
}
