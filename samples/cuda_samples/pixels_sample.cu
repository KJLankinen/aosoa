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
