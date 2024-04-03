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
#include "cuda_memory_operations.h"

using MemOp = CudaDeviceMemoryOperationsAsync;
using PixelSoa = Soa<MemOp>;
using Pixels = Acc<MemOp>;

__global__ void init(Pixels *pixels) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < pixels->size();
         i += blockDim.x * gridDim.x) {
        computeColor(i, pixels);
    }
}

int main(int , char **) {
    cudaStream_t stream = {};
    [[maybe_unused]] auto result = cudaStreamCreate(&stream);

    MemOp memory_ops(stream);

    Pixels *d_accessor = nullptr;
    result = cudaMalloc(&d_accessor, sizeof(Pixels));

    PixelSoa pixel_soa(memory_ops, num_pixels, d_accessor);

    init<<<128, 128>>>(d_accessor);
    result = cudaDeviceSynchronize();

    writePixelsToFile(pixel_soa, "pixels_cu.png");

    result = cudaStreamDestroy(stream);
    result = cudaFree(static_cast<void *>(d_accessor));

    return 0;
}
