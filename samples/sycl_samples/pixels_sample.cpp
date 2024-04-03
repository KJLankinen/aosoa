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

#include "common.h"
#include "sycl_memory_operations.h"

using namespace aosoa;

using MemOp = SyclHostMemoryOperationsAsync;
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
    queue.wait();
    writePixelsToFile(pixel_soa, "pixels_sycl.png");
    sycl::free(d_pixels, queue);

    return 0;
}
