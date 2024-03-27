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

#include "aosoa.h"
#include "cuda_memory_operations.h"
#include "definitions.h"
#include "variable.h"
#include <iostream>

using namespace aosoa;
using ParticleSoa =
    StructureOfArrays<256, CudaMemoryOperationsAsync, Variable<float, "x">,
                      Variable<float, "y">, Variable<float, "z">,
                      Variable<float, "r">>;
using Particles = ParticleSoa::ThisAccessor;

HOST DEVICE void setAllTo(size_t i, Particles *particles) {
    particles->set<"x">(i, static_cast<float>(i));
    particles->set<"y">(i, static_cast<float>(i));
    particles->set<"z">(i, static_cast<float>(i));
    particles->set<"r">(i, static_cast<float>(i));
}

__global__ void init(Particles *particles) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < particles->size(); i += blockDim.x * gridDim.x) {
        setAllTo(i, particles);
    }
}

int main(int , char **) {
    cudaStream_t stream = {};
    [[maybe_unused]] auto result = cudaStreamCreate(&stream);

    CudaMemoryOperationsAsync memory_ops{
        CudaAllocator{}, CudaMemcpyAsync(stream), CudaMemsetAsync(stream)};

    Particles *d_accessor = nullptr;
    result = cudaMalloc(&d_accessor, sizeof(Particles));

    ParticleSoa particle_soa(memory_ops, 1000, d_accessor);

    init<<<128, 128>>>(d_accessor);
    result = cudaDeviceSynchronize();

    auto rows = particle_soa.getRows();
    for (const auto &row : rows) {
        std::cout << row << std::endl;
    }

    result = cudaStreamDestroy(stream);
    result = cudaFree(static_cast<void *>(d_accessor));

    return 0;
}
