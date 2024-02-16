#include "../aosoa.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace aosoa;

void soa() {
    typedef aosoa::StructureOfArrays<128, aosoa::Variable<bool, "is_visible">,
                                     aosoa::Variable<float, "radius">,
                                     aosoa::Variable<double, "radius2">,
                                     aosoa::Variable<int, "num_hits">>
        Soa;

    const size_t n = 5;
    const size_t mem_req = Soa::getMemReq(n);
    std::cout << "mem req: " << mem_req << std::endl;

    std::vector<uint8_t> memory(mem_req);
    Soa::Accessor accessor;
    Soa thingie(n, &accessor);
    thingie.allocate(memory.data());
    thingie.update();

    bool *is_visible = accessor.get<"is_visible">();
    float *radii = accessor.get<"radius">();
    double *radii2 = accessor.get<"radius2">();
    int *num_hits = accessor.get<"num_hits">();

    for (size_t i = 0; i < n; i++) {
        is_visible[i] = i < n / 2;
        radii[i] = static_cast<float>(i);
        radii2[i] = static_cast<double>(i);
        num_hits[i] = -static_cast<int>(i);
    }

    thingie.update();

    std::cout << accessor.get<"is_visible">(n / 2 - 1) << " "
              << accessor.get<"is_visible">(n / 2) << " "
              << accessor.get<"radius">(n / 2 - 1) << " "
              << accessor.get<"radius">(n / 2) << std::endl;

    accessor.set<"radius">(4, 1338.0f);
    thingie.update();

    std::cout << accessor.get<"radius2">(4) << " " << accessor.get<"radius">(4)
              << std::endl;

    Soa::Accessor accessor2;
    std::memcpy(static_cast<void *>(&accessor2), static_cast<void *>(&accessor),
                sizeof(Soa::Accessor));

    for (size_t i = 0; i < n; i++) {
        std::cout << accessor2.get(i) << std::endl;
    }

    accessor.set(2, Soa::FullRow(true, 1337.0f, 1337.0, -12));
    thingie.update();
    std::cout << accessor2.get(2) << std::endl;
}

namespace {
// Definitions for a simple testing harness
struct Result {
    bool success = true;
    std::string msg = "";
};

typedef void (*Fn)(Result &);
struct Test {
    const char *test_name;
    Fn fn;
};

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            result = Result{false, msg};                                       \
            return;                                                            \
        }                                                                      \
    } while (0)
} // namespace

constexpr static Test tests[]{
    {"Row_construct1",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             row(1.0, 1.0f, 1);

         ASSERT(row.get<"foo">() == 1.0, "foo incorrect");
         ASSERT(row.get<"bar">() == 1.0f, "bar incorrect");
         ASSERT(row.get<"baz">() == 1, "baz incorrect");
     }},
    {"Row_construct2",
     [](Result &result) {
         const Row<Variable<double, "foo">> row(1.0);
         ASSERT(row.get<"foo">() == 1.0, "foo incorrect");
     }},
    {"Row_construct3",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             row(Row<Variable<double, "foo">, Variable<float, "bar">,
                     Variable<int, "baz">>{5.0, 6.0f, 7});

         ASSERT(row.get<"foo">() == 5.0, "foo incorrect");
         ASSERT(row.get<"bar">() == 6.0f, "bar incorrect");
         ASSERT(row.get<"baz">() == 7, "baz incorrect");
     }},
    {"Row_construct4",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             row(5.0,
                 Row<Variable<float, "bar">, Variable<int, "baz">>{6.0f, 7});

         ASSERT(row.get<"foo">() == 5.0, "foo incorrect");
         ASSERT(row.get<"bar">() == 6.0f, "bar incorrect");
         ASSERT(row.get<"baz">() == 7, "baz incorrect");
     }},
    {"Row_construct5",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             row(5.0, 6.0f, 7);

         ASSERT(row.get<"foo">() == 5.0, "foo incorrect");
         ASSERT(row.get<"bar">() == 6.0f, "bar incorrect");
         ASSERT(row.get<"baz">() == 7, "baz incorrect");
     }},
    {"Row_default_construct",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">>
             row;

         ASSERT(row.get<"foo">() == 0.0, "foo incorrect");
         ASSERT(row.get<"bar">() == 0.0f, "bar incorrect");
         ASSERT(row.get<"baz">() == 0, "baz incorrect");
     }},
    {"Row_set",
     [](Result &result) {
         Row<Variable<double, "foo">> row(1.0);
         row.set<"foo">(2.0);
         ASSERT(row.get<"foo">() == 2.0, "foo incorrect");
     }},
    {"Row_set2",
     [](Result &result) {
         Row<Variable<double, "foo">> row(1.0);
         row.get<"foo">() = 2.0;
         ASSERT(row.get<"foo">() == 2.0, "foo incorrect");
     }},
    {"Row_equality1",
     [](Result &result) {
         Row<Variable<double, "foo">> row(1.0);
         row.get<"foo">() = 2.0;
         ASSERT((row == Row<Variable<double, "foo">>(2.0)), "Values inequal");
     }},
    {"Row_equality2",
     [](Result &result) {
         typedef Row<Variable<double, "foo">, Variable<int, "bar">,
                     Variable<char, "c">>
             R;
         const R row(1.0, 22222, 'c');
         ASSERT((row == R(1.0, 22222, 'c')), "Values inequal");
     }},
    {"sizeof(row)",
     [](Result &result) {
         const Row<Variable<double, "foo">, Variable<float, "bar">,
                   Variable<int, "baz">, Variable<char, "foo2">>
             row(1.0, 1.0f, 1, 'b');

         // double + (float + int) + (char + padding)
         ASSERT(sizeof(row) == 3 * sizeof(double), "Size incorrect");
     }},
    {"AoSoa_getMemReq1",
     [](Result &result) {
         constexpr size_t alignment = 1;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq2",
     [](Result &result) {
         constexpr size_t alignment = 2;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq3",
     [](Result &result) {
         constexpr size_t alignment = 4;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq4",
     [](Result &result) {
         constexpr size_t alignment = 8;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq5",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq6",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq7",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1024;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == n * (sizeof(double) + sizeof(float)) + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq8",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<alignment, Variable<double, "first">,
                                   Variable<float, "second">>
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
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
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
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
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
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
             Soa;
         const size_t mem_req = Soa::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_getMemReq_BigType",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 100;

         typedef StructureOfArrays<
             alignment, Variable<double, "1">, Variable<double, "2">,
             Variable<double, "3">, Variable<double, "4">,
             Variable<double, "5">, Variable<double, "6">,
             Variable<double, "7">, Variable<double, "8">,
             Variable<double, "9">, Variable<double, "10">,
             Variable<double, "11">, Variable<float, "12">,
             Variable<float, "13">, Variable<float, "14">,
             Variable<float, "15">, Variable<float, "16">,
             Variable<float, "17">, Variable<float, "18">,
             Variable<float, "19">, Variable<float, "20">,
             Variable<float, "21">, Variable<float, "22">,
             Variable<float, "23">, Variable<float, "24">, Variable<int, "25">,
             Variable<int, "26">, Variable<int, "27">, Variable<int, "28">,
             Variable<int, "29">, Variable<int, "30">, Variable<int, "31">,
             Variable<int, "32">, Variable<int, "33">, Variable<int, "34">,
             Variable<int, "35">, Variable<int, "36">, Variable<int, "37">,
             Variable<int, "38">, Variable<int, "39">, Variable<int, "40">,
             Variable<int, "41">, Variable<int, "42">, Variable<int, "43">,
             Variable<int, "44">, Variable<int, "45">, Variable<int, "46">,
             Variable<int, "47">, Variable<int, "48">, Variable<int, "49">,
             Variable<int, "50">, Variable<bool, "51">, Variable<bool, "52">,
             Variable<bool, "53">, Variable<bool, "54">, Variable<bool, "55">,
             Variable<bool, "56">, Variable<bool, "57">, Variable<bool, "58">,
             Variable<bool, "59">, Variable<bool, "60">, Variable<bool, "61">,
             Variable<bool, "62">, Variable<bool, "63">, Variable<char, "64">,
             Variable<char, "65">, Variable<char, "66">, Variable<char, "67">,
             Variable<char, "68">, Variable<char, "69">, Variable<char, "70">,
             Variable<char, "71">, Variable<char, "72">, Variable<char, "73">>
             BigType;

         const size_t mem_req = BigType::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_construction1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<float, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         const std::array<void *, 5> pointers = {
             static_cast<void *>(accessor.get<"first">()),
             static_cast<void *>(accessor.get<"second">()),
             static_cast<void *>(accessor.get<"third">()),
             static_cast<void *>(accessor.get<"fourth">()),
             static_cast<void *>(accessor.get<"fifth">()),
         };

         constexpr size_t max = ~0ul;
         size_t space = max;
         for (auto pointer : pointers) {
             auto ptr = std::align(alignment, 1, pointer, space);
             ASSERT(ptr == pointer, "Pointer is not aligned correctly");
             ASSERT(space == max, "Space should not change");
         }

         constexpr std::array<uintptr_t, 4> sizes = {
             8064,
             4096,
             4096,
             1024,
         };

         for (size_t i = 0; i < pointers.size() - 1; i++) {
             ASSERT(reinterpret_cast<uintptr_t>(pointers[i + 1]) -
                            reinterpret_cast<uintptr_t>(pointers[i]) ==
                        sizes[i],
                    "Size wrong");
         }

         const uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
         const uintptr_t bytes_at_end =
             mem_req + begin - reinterpret_cast<uintptr_t>(pointers[4]) - 4096;
         const uintptr_t bytes_at_begin =
             reinterpret_cast<uintptr_t>(pointers[0]) - begin;
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
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         ASSERT(mem_req == 18008, "Memory requirement incorrect");
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         const std::array<uintptr_t, 5> pointers = {
             reinterpret_cast<uintptr_t>(accessor.get<"first">()),
             reinterpret_cast<uintptr_t>(accessor.get<"second">()),
             reinterpret_cast<uintptr_t>(accessor.get<"third">()),
             reinterpret_cast<uintptr_t>(accessor.get<"fourth">()),
             reinterpret_cast<uintptr_t>(accessor.get<"fifth">()),
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
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         const std::array<uintptr_t, 2> original_pointers = {
             reinterpret_cast<uintptr_t>(accessor.get<"first">()),
             reinterpret_cast<uintptr_t>(accessor.get<"second">()),
         };

         soa.swap<"first", "second">();
         soa.update();

         const std::array<uintptr_t, 2> pointers = {
             reinterpret_cast<uintptr_t>(accessor.get<"first">()),
             reinterpret_cast<uintptr_t>(accessor.get<"second">()),
         };

         ASSERT(original_pointers[0] == pointers[1],
                "Pointers not swapped correctly 1");
         ASSERT(original_pointers[1] == pointers[0],
                "Pointers not swapped correctly 2");
     }},
    {"AoSoa_get_set1",
     [](Result &result) {
         // Assuming that values are by default 0... Might not be the case
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         constexpr size_t a = 0;
         constexpr size_t b = 32;
         constexpr size_t c = 555;

         accessor.set(a, Soa::FullRow(1.0, 2.0, 3, true, 5.0f));
         accessor.set(b, Soa::FullRow(1.0, 2.0, 3, true, 5.0f));
         accessor.set(c, Soa::FullRow(1.0, 2.0, 3, true, 5.0f));
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             const bool is_default = i != a && i != b && i != c;
             if (is_default) {
                 ASSERT(accessor.get(i) ==
                            Soa::FullRow(0.0, 0.0, 0, false, 0.0f),
                        "Incorrect default value");
             } else {
                 ASSERT(accessor.get(i) ==
                            Soa::FullRow(1.0, 2.0, 3, true, 5.0f),
                        "Incorrect default value");
             }
         }
     }},
    {"AoSoa_get_set2",
     [](Result &result) {
         // Assuming that values are by default 0... Might not be the case
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         constexpr size_t a = 0;
         constexpr size_t b = 2;
         constexpr size_t c = 55;

         accessor.set<"first">(a, 5.0);
         accessor.set<"first">(b, 666.666);
         accessor.set<"first">(c, 321);
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             const bool is_default = i != a && i != b && i != c;
             if (is_default) {
                 ASSERT(accessor.get<"first">(i) == 0.0,
                        "Incorrect default value");
             } else if (i == a) {
                 ASSERT(accessor.get<"first">(i) == 5.0, "Incorrect value");
             } else if (i == b) {
                 ASSERT(accessor.get<"first">(i) == 666.666, "Incorrect value");
             } else if (i == c) {
                 ASSERT(accessor.get<"first">(i) == 321, "Incorrect value");
             }
         }

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get<"second">(i) == 0.0,
                    "Incorrect default value");
             ASSERT(accessor.get<"third">(i) == 0, "Incorrect default value");
             ASSERT(accessor.get<"fourth">(i) == false,
                    "Incorrect default value");
             ASSERT(accessor.get<"fifth">(i) == 0.0f,
                    "Incorrect default value");
         }
     }},
    {"AoSoa_get_set3",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         constexpr size_t a = 0;
         constexpr size_t b = 2;
         constexpr size_t c = 55;

         auto value = accessor.get<"first">();
         value[a] = 1.0;
         value[b] = 2.0;
         value[c] = 3.0;
         soa.update();

         ASSERT(accessor.get<"first">(a) == 1.0, "Value incorrect");
         ASSERT(accessor.get<"first">(b) == 2.0, "Value incorrect");
         ASSERT(accessor.get<"first">(c) == 3.0, "Value incorrect");
     }},
    {"AoSoa_get_set4",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         constexpr size_t a = 0;
         constexpr size_t b = 2;
         constexpr size_t c = 55;

         auto value = accessor.get<"first">();
         value[a] = 1.0;
         value[b] = 2.0;
         value[c] = 3.0;
         soa.update();

         ASSERT((accessor.get<"first", a>() == 1.0), "Value incorrect");
         ASSERT((accessor.get<"first", b>() == 2.0), "Value incorrect");
         ASSERT((accessor.get<"first", c>() == 3.0), "Value incorrect");
     }},
    {"AoSoa_memset",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         soa.memset<"third">(std::memset, 0xFF);
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(static_cast<uint32_t>(accessor.get<"third">(i)) ==
                        0xFFFFFFFF,
                    "Value incorrect first");
         }

         soa.memset<"third">(std::memset, 0);
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get<"third">(i) == 0, "Value incorrect second");
         }
     }},
    {"AoSoa_memcpy1",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));
         soa.update();

         const std::vector<double> rands1([]() {
             std::vector<double> vec(n);
             std::random_device rd{};
             std::mt19937 generator{rd()};
             std::normal_distribution distribution{0.0, 1.0};
             std::generate(vec.begin(), vec.end(),
                           [&distribution, &generator]() {
                               return distribution(generator);
                           });
             return vec;
         }());

         soa.memcpy<"first">(rands1.data(), std::memcpy);
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get<"first">(i) == rands1[i],
                    "Incorret value first");
         }

         soa.memcpy<"second", "first">(std::memcpy);
         soa.update();

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get<"second">(i) == rands1[i],
                    "Incorret value second");
         }

         std::vector<double> rands2(n);
         soa.memcpy<"second">(rands2.data(), std::memcpy);

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(rands2[i] == rands1[i], "Incorret value rands2");
         }
     }},
    {"AoSoa_alignment_bytes",
     [](Result &result) {
         constexpr size_t alignment = 64;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));

         const uintptr_t address = reinterpret_cast<uintptr_t>(bytes.data());
         const uintptr_t over_alignment = address & (alignment - 1);
         const uintptr_t missing =
             (alignment - over_alignment) & (alignment - 1);

         ASSERT(soa.getAlignmentBytes() == missing,
                "Incorrectly computed alignment bytes");
     }},
    {"AoSoa_aligned_block_size",
     [](Result &result) {
         constexpr size_t alignment = 64;
         constexpr size_t n = 1000;
         typedef StructureOfArrays<
             alignment, Variable<double, "first">, Variable<double, "second">,
             Variable<int, "third">, Variable<bool, "fourth">,
             Variable<float, "fifth">>
             Soa;

         const size_t mem_req = Soa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         Soa::Accessor accessor;
         Soa soa(n, &accessor);
         soa.allocate(static_cast<void *>(bytes.data()));

         ASSERT(alignment + soa.getAlignedBlockSize() == Soa::getMemReq(n),
                "Incorrectly computed alignment bytes");
     }},
};

int main(int, char **) {
    for (auto [test_name, test] : tests) {
        Result result{};
        test(result);
        printf("%s %s%s\n", result.success ? "OK  " : "FAIL", test_name,
               result.success ? "" : (" \"" + result.msg + "\"").c_str());
    }
}
