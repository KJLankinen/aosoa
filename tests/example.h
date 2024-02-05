#include "../aosoa.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

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
struct TestResult {
    bool success;
    std::string output;
};

typedef TestResult (*Fn)();
struct Test {
    const char *test_name;
    Fn fn;
};

#define TEST(test)                                                             \
    Test { #test, test }

#define OK()                                                                   \
    TestResult { true, "" }
#define ERR(output)                                                            \
    TestResult { false, output }
} // namespace

// Hash:
// hashing
// clashing
// crc
// Tuple:
// Test all constructors
// Test getting with constructor
// Test setting, then getting
// AoSoa
// Test all constructors
// Test getMemReq() with multiple template arguments
// Test swap
// Test all gets
// Test all sets
const static Test tests[]{
    {"First", []() -> TestResult { return OK(); }},
    {"Second", []() -> TestResult { return ERR("I fail a lot"); }},
};

void test() {
    for (auto [test_name, test] : tests) {
        const auto [success, output] = test();
        printf("%s %s\n", test_name,
               success ? "OK  " : ("FAIL \"" + output + "\"").c_str());
    }
}
