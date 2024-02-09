#include "../aosoa.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace aosoa;

void soa() {
    typedef aosoa::StructureOfArrays<128, aosoa::Variable<bool, "is_visible">,
                                     aosoa::Variable<float, "radius">,
                                     aosoa::Variable<double, "radius2">,
                                     aosoa::Variable<int, "num_hits">>
        Thingie;

    const size_t n = 5;
    const size_t mem_req = Thingie::getMemReq(n);
    std::cout << "mem req: " << mem_req << std::endl;

    std::vector<uint8_t> memory(mem_req);
    Thingie thingie(n, memory.data());

    bool *is_visible = thingie.get<"is_visible">();
    float *radii = thingie.get<"radius">();
    double *radii2 = thingie.get<"radius2">();
    int *num_hits = thingie.get<"num_hits">();

    for (size_t i = 0; i < n; i++) {
        is_visible[i] = i < n / 2;
        radii[i] = static_cast<float>(i);
        radii2[i] = static_cast<double>(i);
        num_hits[i] = -static_cast<int>(i);
    }

    std::cout << thingie.get<"is_visible">(n / 2 - 1) << " "
              << thingie.get<"is_visible">(n / 2) << " "
              << thingie.get<"radius">(n / 2 - 1) << " "
              << thingie.get<"radius">(n / 2) << std::endl;

    thingie.set<"radius">(4, 1338.0f);
    std::cout << thingie.get<"radius2">(4) << " " << thingie.get<"radius">(4)
              << std::endl;

    Thingie soa2;
    std::memcpy(static_cast<void *>(&soa2), static_cast<void *>(&thingie),
                sizeof(Thingie));
    std::cout << thingie << "\n" << soa2 << std::endl;

    for (size_t i = 0; i < n; i++) {
        std::cout << soa2.get(i) << std::endl;
    }

    thingie.set(2, Thingie::FullRow(true, 1337.0f, 1337.0, -12));
    std::cout << soa2.get(2) << std::endl;
}

namespace {
// Definitions for a simple testing harness
struct Result {
    bool success;
    std::string msg;
};

typedef void (*Fn)(Result &);
struct Test {
    const char *test_name;
    Fn fn;
};

#define OK()                                                                   \
    Result { true, "" }

#define ERR(output)                                                            \
    Result { false, output }

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            result = ERR(msg);                                                 \
            return;                                                            \
        }                                                                      \
    } while (0)
} // namespace

// Row
// construction
// unique names
// StructureOfArrays
// Test all constructors: pay attention to space and pointers
// Test getMemReq() with multiple template arguments
// Test swap
// Test all gets
// Test all sets
constexpr static Test tests[]{
    {"Aos_construct1",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             aos(1.0, 1.0f, 1);

         ASSERT(aos.get<"foo">() == 1.0, "foo incorrect");
         ASSERT(aos.get<"bar">() == 1.0f, "bar incorrect");
         ASSERT(aos.get<"baz">() == 1, "baz incorrect");
     }},
    {"AoSoa_getMemReq1",
     [](Result &result) {
         constexpr size_t alignment = 1;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq2",
     [](Result &result) {
         constexpr size_t alignment = 2;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq3",
     [](Result &result) {
         constexpr size_t alignment = 4;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq4",
     [](Result &result) {
         constexpr size_t alignment = 8;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq5",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq6",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq7",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1024;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == n * (sizeof(double) + sizeof(float)) + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq8",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 8064 + 4096 + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq9",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<float, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 8064 + 4096 + 4096 + 1024 + 4096 + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq10",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 3216547;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<char, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_getMemReq11",
     [](Result &result) {
         constexpr size_t alignment = 32;
         constexpr size_t n = 3216547;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<char, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_default_constructor",
     [](Result &result) {
         constexpr size_t alignment = 128;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Aosoa;
         constexpr Aosoa a;
         ASSERT(a.get<"first">() == nullptr, "First pointer should be nullpt");
         ASSERT(a.get<"second">() == nullptr,
                "Second pointer shold be nullptr");
     }},
    {"AoSoa_construction1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<float, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;

         const size_t mem_req = Aosoa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         const Aosoa a(n, static_cast<void *>(bytes.data()));

         const std::array<uintptr_t, 5> pointers = {
             reinterpret_cast<uintptr_t>(a.get<"first">()),
             reinterpret_cast<uintptr_t>(a.get<"second">()),
             reinterpret_cast<uintptr_t>(a.get<"third">()),
             reinterpret_cast<uintptr_t>(a.get<"fourth">()),
             reinterpret_cast<uintptr_t>(a.get<"fifth">()),
         };

         for (auto pointer : pointers) {
             ASSERT((pointer & (alignment - 1)) == 0,
                    "Pointer is not aligned correctly");
         }

         constexpr std::array<uintptr_t, 4> sizes = {
             8064,
             4096,
             4096,
             1024,
         };

         for (size_t i = 0; i < pointers.size() - 1; i++) {
             ASSERT(pointers[i + 1] - pointers[i] == sizes[i], "Size wrong");
         }

         const uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
         const uintptr_t bytes_at_end = mem_req + begin - pointers[4] - 4096;
         const uintptr_t bytes_at_begin = pointers[0] - begin;
         ASSERT(
             bytes_at_end == alignment - bytes_at_begin,
             "Bytes at the end should be alignment - bytes at the beginning");
     }},
    {"AoSoa_construction2",
     [](Result &result) {
         constexpr size_t alignment = 1;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<char, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;

         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 18008, "Memory requirement incorrect");
         std::vector<uint8_t> bytes(mem_req);
         const Aosoa a(n, static_cast<void *>(bytes.data()));

         const std::array<uintptr_t, 5> pointers = {
             reinterpret_cast<uintptr_t>(a.get<"first">()),
             reinterpret_cast<uintptr_t>(a.get<"second">()),
             reinterpret_cast<uintptr_t>(a.get<"third">()),
             reinterpret_cast<uintptr_t>(a.get<"fourth">()),
             reinterpret_cast<uintptr_t>(a.get<"fifth">()),
         };

         for (auto pointer : pointers) {
             ASSERT((pointer & (alignment - 1)) == 0,
                    "Pointer is not aligned correctly");
         }

         constexpr std::array<uintptr_t, 4> sizes = {
             8000,
             1000,
             4000,
             1000,
         };

         for (size_t i = 0; i < pointers.size() - 1; i++) {
             ASSERT(pointers[i + 1] - pointers[i] == sizes[i], "Size wrong");
         }

         const uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
         const uintptr_t bytes_at_end = mem_req + begin - pointers[4] - 4000;
         const uintptr_t bytes_at_begin = pointers[0] - begin;
         ASSERT(
             bytes_at_end == alignof(double) - bytes_at_begin,
             "Bytes at the end should be alignment - bytes at the beginning");
     }},
    {"AoSoa_swap",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Aosoa;

         const size_t mem_req = Aosoa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Aosoa a(n, static_cast<void *>(bytes.data()));

         const std::array<uintptr_t, 2> original_pointers = {
             reinterpret_cast<uintptr_t>(a.get<"first">()),
             reinterpret_cast<uintptr_t>(a.get<"second">()),
         };

         a.swap<"first", "second">();

         const std::array<uintptr_t, 2> pointers = {
             reinterpret_cast<uintptr_t>(a.get<"first">()),
             reinterpret_cast<uintptr_t>(a.get<"second">()),
         };

         ASSERT(original_pointers[0] == pointers[1],
                "Pointers not swapped correctly 1");
         ASSERT(original_pointers[1] == pointers[0],
                "Pointers not swapped correctly 2");
     }},
};

int main(int, char **) {
    Result result;
    for (auto [test_name, test] : tests) {
        test(result);
        printf("%s %s%s\n", result.success ? "OK  " : "FAIL", test_name,
               result.success ? "" : (" \"" + result.msg + "\"").c_str());
    }
}
