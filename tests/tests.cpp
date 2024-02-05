#include "../aosoa.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace aosoa;

void soa() {
    using aosoa::operator""_idx;

    typedef aosoa::AoSoa<128, aosoa::IndexTypePair<"is_visible"_idx, bool>,
                         aosoa::IndexTypePair<"radius"_idx, float>,
                         aosoa::IndexTypePair<"radius2"_idx, double>,
                         aosoa::IndexTypePair<"num_hits"_idx, int>>
        Thingie;

    const size_t n = 5;
    const size_t mem_req = Thingie::getMemReq(n);
    std::cout << "mem req: " << mem_req << std::endl;

    std::vector<uint8_t> memory(mem_req);
    Thingie thingie(n, memory.data());

    auto is_visible = thingie.get<"is_visible"_idx>();
    auto radii = thingie.get<"radius"_idx>();
    auto radii2 = thingie.get<"radius2"_idx>();
    auto num_hits = thingie.get<"num_hits"_idx>();

    for (size_t i = 0; i < n; i++) {
        is_visible[i] = i < n / 2;
        radii[i] = static_cast<float>(i);
        radii2[i] = static_cast<double>(i);
        num_hits[i] = -static_cast<int>(i);
    }

    std::cout << thingie.get<"is_visible"_idx>(n / 2 - 1) << " "
              << thingie.get<"is_visible"_idx>(n / 2) << " "
              << thingie.get<"radius"_idx>(n / 2 - 1) << " "
              << thingie.get<"radius"_idx>(n / 2) << std::endl;

    thingie.set<"radius"_idx>(4, 1338.0f);
    std::cout << thingie.get<"radius2"_idx>(4) << " "
              << thingie.get<"radius"_idx>(4) << std::endl;

    Thingie soa2;
    std::memcpy(static_cast<void *>(&soa2), static_cast<void *>(&thingie),
                sizeof(Thingie));
    std::cout << thingie << soa2 << std::endl;

    for (size_t i = 0; i < n; i++) {
        std::cout << soa2.get<Thingie::Aos>(i) << std::endl;
    }

    thingie.set(2, Thingie::Aos(true, 1337.0f, 1337.0, -12));
    std::cout << soa2.get<Thingie::Aos>(2) << std::endl;

    auto soa = thingie.get<Thingie::Soa>(0);
    bool *bptr = soa.get<0>();
    for (size_t i = 0; i < n; i++) {
        std::cout << *(bptr++) << std::endl;
    }
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

// AoSoa
// Test all constructors: pay attention to space and pointers
// Test getMemReq() with multiple template arguments
// Test swap
// Test all gets
// Test all sets
constexpr static Test tests[]{
    {"operator\"\"_idx_equal",
     [](Result &result) {
         using aosoa::operator""_idx;
         const uint32_t a = "test"_idx;
         const uint32_t b = "test"_idx;
         ASSERT(a == b, "a should be equal to b");
     }},
    {"operator\"\"_idx_different",
     [](Result &result) {
         using aosoa::operator""_idx;
         const uint32_t a = "aaa"_idx;
         const uint32_t b = "bbb"_idx;
         ASSERT(a != b, "a shound not be equal to b");
     }},
    {"Tuple_construction1",
     [](Result &result) {
         const Tuple<float> t;
         ASSERT(t.head == 0.0f, "Head not zero");
         ASSERT(t.tail == Tuple<>(), "Tail not empty");
     }},
    {"Tuple_construction2",
     [](Result &result) {
         const Tuple<float, double, int> t(
             0.0f, Tuple<double, int>(0.0, Tuple<int>()));
         ASSERT(t.head == 0.0f, "Head not zero");
         const bool condition = t.tail == Tuple<double, int>();
         ASSERT(condition, "Tail not default");
     }},
    {"Tuple_get1",
     [](Result &result) {
         const Tuple<float, double, int, bool> t(0.0f, 0.0, 0, true);
         ASSERT(t.get<0>() == 0.0f, "Value mismatch");
     }},
    {"Tuple_get2",
     [](Result &result) {
         const Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         ASSERT(t.get<1>() == 12.0, "Value mismatch");
     }},
    {"Tuple_get3",
     [](Result &result) {
         const Tuple<float, double, int, bool> t(0.0f, 12.0, -50, true);
         ASSERT(t.get<2>() == -50, "Value mismatch");
     }},
    {"Tuple_get4",
     [](Result &result) {
         const Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         ASSERT(t.get<3>(), "Value mismatch");
     }},
    {"Tuple_set1",
     [](Result &result) {
         Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         constexpr auto value = 666.666f;
         t.set<0>(value);
         ASSERT(t.get<0>() == value, "Value mismatch");
     }},
    {"Tuple_set2",
     [](Result &result) {
         Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         constexpr auto value = 666.666;
         t.set<1>(value);
         ASSERT(t.get<1>() == value, "Value mismatch");
     }},
    {"Tuple_set3",
     [](Result &result) {
         Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         constexpr auto value = 666;
         t.set<2>(value);
         ASSERT(t.get<2>() == value, "Value mismatch");
     }},
    {"Tuple_set4",
     [](Result &result) {
         Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         constexpr auto value = false;
         t.set<3>(value);
         ASSERT(t.get<3>() == value, "Value mismatch");
     }},
    {"Tuple_set5",
     [](Result &result) {
         Tuple<float, double, int, bool> t(0.0f, 12.0, 0, true);
         constexpr auto value = false;
         t.set<3>(value);
         ASSERT(t.get<0>() == 0.0f, "First value should not be changed");
         ASSERT(t.get<1>() == 12.0, "Second value should not be changed");
         ASSERT(t.get<2>() == 0, "Third value should not be changed");
         ASSERT(t.get<3>() == value, "Fourth value should be changed");
     }},
    {"AoSoa_getMemReq1",
     [](Result &result) {
         constexpr size_t alignment = 1;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq2",
     [](Result &result) {
         constexpr size_t alignment = 2;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq3",
     [](Result &result) {
         constexpr size_t alignment = 4;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq4",
     [](Result &result) {
         constexpr size_t alignment = 8;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq5",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq6",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq7",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1024;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == n * (sizeof(double) + sizeof(float)) + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq8",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 8064 + 4096 + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq9",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>,
                       IndexTypePair<"third"_idx, int>,
                       IndexTypePair<"fourth"_idx, bool>,
                       IndexTypePair<"fifth"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT(mem_req == 8064 + 4096 + 4096 + 1024 + 4096 + alignment,
                "Memory requirement mismatch");
     }},
    {"AoSoa_getMemReq10",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 3216547;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, char>,
                       IndexTypePair<"third"_idx, int>,
                       IndexTypePair<"fourth"_idx, bool>,
                       IndexTypePair<"fifth"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_getMemReq11",
     [](Result &result) {
         constexpr size_t alignment = 32;
         constexpr size_t n = 3216547;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, char>,
                       IndexTypePair<"third"_idx, int>,
                       IndexTypePair<"fourth"_idx, bool>,
                       IndexTypePair<"fifth"_idx, float>>
             Aosoa;
         const size_t mem_req = Aosoa::getMemReq(n);
         ASSERT((mem_req & (alignment - 1)) == 0,
                "Total memory requirement must be a multiple of alignment");
     }},
    {"AoSoa_default_constructor",
     [](Result &result) {
         constexpr size_t alignment = 128;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>>
             Aosoa;
         constexpr Aosoa a;
         ASSERT(a.get<"first"_idx>() == nullptr,
                "First pointer should be nullpt");
         ASSERT(a.get<"second"_idx>() == nullptr,
                "Second pointer shold be nullptr");
     }},
    {"AoSoa_construction1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         typedef AoSoa<alignment, IndexTypePair<"first"_idx, double>,
                       IndexTypePair<"second"_idx, float>,
                       IndexTypePair<"third"_idx, int>,
                       IndexTypePair<"fourth"_idx, bool>,
                       IndexTypePair<"fifth"_idx, float>>
             Aosoa;

         const size_t mem_req = Aosoa::getMemReq(n);
         std::vector<uint8_t> bytes(mem_req);
         const Aosoa a(n, static_cast<void *>(bytes.data()));

         const std::array<uintptr_t, 5> pointers = {
             reinterpret_cast<uintptr_t>(a.get<"first"_idx>()),
             reinterpret_cast<uintptr_t>(a.get<"second"_idx>()),
             reinterpret_cast<uintptr_t>(a.get<"third"_idx>()),
             reinterpret_cast<uintptr_t>(a.get<"fourth"_idx>()),
             reinterpret_cast<uintptr_t>(a.get<"fifth"_idx>()),
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
};

int main(int, char **) {
    Result result;
    for (auto [test_name, test] : tests) {
        test(result);
        printf("%s %s%s\n", result.success ? "OK  " : "FAIL", test_name,
               result.success ? "" : (" \"" + result.msg + "\"").c_str());
    }
}
