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

#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "aosoa.h"
#include "definitions.h"
#include "variable.h"

constexpr size_t tile_width = 32ul;
constexpr size_t num_tiles_i = 2ul;
constexpr size_t num_tiles_j = 2ul;
constexpr size_t num_pixels =
    tile_width * tile_width * num_tiles_i * num_tiles_j;

using namespace aosoa;

// clang-format off
template <typename M>
using Soa = StructureOfArrays<
    256,
    M,
    Variable<float, "r">,
    Variable<float, "g">,
    Variable<float, "b">
    >;
// clang-format on

template <typename M> using Acc = Soa<M>::ThisAccessor;

template <typename A>
HOST DEVICE void computeColor(size_t linear_index, A *accessor) {
    const size_t i = linear_index % (tile_width * num_tiles_i);
    const size_t j = linear_index / (tile_width * num_tiles_i);
    const size_t tile_i = i % tile_width;
    const size_t tile_j = j % tile_width;
    const float mul = static_cast<float>(tile_i + tile_j * tile_width) /
                      (tile_width * tile_width);
    // Modulo 4
    const size_t color_index =
        (i / tile_width + j / tile_width * num_tiles_i) & 3;
    size_t color_bits = 1ul << color_index;
    // First is non-zero for color_index 0, 1, 2
    // Second is non-zero for color_index 3
    color_bits = (color_bits & 7ul) + (color_bits >> 3ul) * (color_bits - 1ul);

    const float r = mul * static_cast<float>(color_bits & 1ul);
    const float g = mul * static_cast<float>((color_bits & 2ul) >> 1);
    const float b = mul * static_cast<float>((color_bits & 4ul) >> 2);

    accessor->template set<"r">(linear_index, r);
    accessor->template set<"g">(linear_index, g);
    accessor->template set<"b">(linear_index, b);
}

template <typename Soa>
int writePixelsToFile(const Soa &soa, const char *filename) {
    const auto rows = soa.getRows();
    std::vector<uint8_t> bytes(3 * rows.size());
    size_t i = 0;
    for (const auto &pixel : rows) {
        bytes[0 + i * 3] =
            static_cast<uint8_t>(pixel.template get<"r">() * 255.0f);
        bytes[1 + i * 3] =
            static_cast<uint8_t>(pixel.template get<"g">() * 255.0f);
        bytes[2 + i * 3] =
            static_cast<uint8_t>(pixel.template get<"b">() * 255.0f);
        i++;
    }

    return stbi_write_png(filename, tile_width * num_tiles_i,
                          tile_width * num_tiles_j, 3, bytes.data(),
                          3 * tile_width * num_tiles_i);
}
