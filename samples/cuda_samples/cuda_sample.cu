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

#include "aosoa/aosoa.h"
#include "aosoa/cuda_memory_operations.h"

int main(int , char **) {
    using namespace aosoa;
    cudaStream_t stream = {};
    auto result = cudaStreamCreate(&stream);

    CudaMemoryOperations<false> memory_ops{
        CudaAllocator{}, CudaMemcpy<false>(stream), CudaMemset<false>(stream)};

    using Soa = StructureOfArrays<256, CudaMemoryOperations<false>,
                                  Variable<float, "foo">, Variable<int, "bar">,
                                  Variable<double, "baz">>;
    Soa soa(memory_ops, 1000);

    result = cudaStreamDestroy(stream);

    return 0;
}
