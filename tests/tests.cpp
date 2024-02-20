#include "../aosoa.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace aosoa;

// TODO:
// StructureOfArrays tests with Balls
//
// Buffer test:
// - constructing and freeing in a loop to see memory usage
// - a throwing constructor for a type that uses buffer: see if deallocation
// works correctly sycl/cuda/hip testing

void soa() {
    typedef aosoa::StructureOfArrays<128, aosoa::Variable<bool, "is_visible">,
                                     aosoa::Variable<float, "radius">,
                                     aosoa::Variable<double, "radius2">,
                                     aosoa::Variable<int, "num_hits">>
        Soa;

    const size_t n = 5;
    const size_t mem_req = Soa::getMemReq(n);
    std::cout << "mem req: " << mem_req << std::endl;

    const aosoa::CMemoryOps memory_ops;
    Soa::Accessor accessor;
    Soa soa(memory_ops, n, &accessor);

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

    std::cout << accessor.get<"is_visible">(n / 2 - 1) << " "
              << accessor.get<"is_visible">(n / 2) << " "
              << accessor.get<"radius">(n / 2 - 1) << " "
              << accessor.get<"radius">(n / 2) << std::endl;

    accessor.set<"radius">(4, 1338.0f);

    std::cout << accessor.get<"radius2">(4) << " " << accessor.get<"radius">(4)
              << std::endl;

    Soa::Accessor accessor2;
    std::memcpy(static_cast<void *>(&accessor2), static_cast<void *>(&accessor),
                sizeof(Soa::Accessor));

    for (size_t i = 0; i < n; i++) {
        std::cout << accessor2.get(i) << std::endl;
    }

    accessor.set(2, Soa::FullRow(true, 1337.0f, 1337.0, -12));
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

typedef Row<Variable<float, "head">> RowSingle;
typedef Row<Variable<float, "head">, Variable<int32_t, "tail">> RowDouble;

template <size_t Alignment>
using MemReqSoa = StructureOfArrays<Alignment, Variable<double, "first">,
                                    Variable<float, "second">>;

// clang-format off
template <size_t Alignment>
using Balls = StructureOfArrays<Alignment,
            Variable<double, "position_x">,
            Variable<double, "position_y">,
            Variable<double, "position_z">,
            Variable<double, "radius">,
            Variable<float, "color_r">,
            Variable<float, "color_g">,
            Variable<float, "color_b">,
            Variable<uint32_t, "index">,
            Variable<int32_t, "index_distance">,
            Variable<bool, "is_visible">>;
// clang-format on

template <size_t Alignment> using Ball = Balls<Alignment>::FullRow;
const aosoa::CMemoryOps memory_ops = {};

constexpr static Test tests[]{
    {"sizeof(RowSingle)",
     [](Result &) { static_assert(sizeof(RowSingle) == sizeof(float)); }},
    {"sizeof(RowDouble)",
     [](Result &) {
         static_assert(sizeof(RowDouble) == sizeof(float) + sizeof(int32_t));
     }},
    {"RowSingle_construct1",
     [](Result &result) {
         ASSERT(RowSingle().get<"head">() == 0.0f,
                "Default value should be 0.0f");
     }},
    {"RowSingle_construct2",
     [](Result &result) {
         ASSERT(RowSingle(1.0f).get<"head">() == 1.0f, "Value should be 1.0f");
     }},
    {"RowSingle_construct3",
     [](Result &result) {
         ASSERT(RowSingle(RowSingle(2.0f)).get<"head">() == 2.0f,
                "Value should be 2.0f");
     }},
    {"RowSingle_get_set1",
     [](Result &result) {
         RowSingle row;
         row.get<"head">() = 10.0f;
         ASSERT(row.get<"head">() == 10.0f, "Value should be 10.0f");
     }},
    {"RowSingle_get_set2",
     [](Result &result) {
         RowSingle row;
         row.set<"head">(10.0f);
         ASSERT(row.get<"head">() == 10.0f, "Value should be 10.0f");
     }},
    {"RowSingle_getconst",
     [](Result &result) {
         const RowSingle row(666.666f);
         const float val = row.get<"head">();
         ASSERT(val == 666.666f, "Value should be 666.666f");
     }},
    {"RowSingle_equality1",
     [](Result &result) {
         const RowSingle row(666.666f);
         ASSERT(row == RowSingle(666.666f), "Values should be equal");
     }},
    {"RowSingle_equality2",
     [](Result &result) {
         const RowSingle row{};
         ASSERT(row == RowSingle(0.0f), "Values should be equal");
     }},
    {"RowSingle_equality3",
     [](Result &result) {
         const RowSingle row{};
         ASSERT(row == row, "Row should be equal to itself");
     }},
    {"RowDouble_construct1",
     [](Result &result) {
         const RowDouble row{};
         ASSERT(row.get<"head">() == 0.0f, "Default value should be 0.0f");
         ASSERT(row.get<"tail">() == 0, "Default value should be 0");
     }},
    {"RowDouble_construct2",
     [](Result &result) {
         const RowDouble row(1.0f, 2);
         ASSERT(row.get<"head">() == 1.0f, "Value should be 1.0f");
         ASSERT(row.get<"tail">() == 2, "Value should be 2");
     }},
    {"RowDouble_construct3",
     [](Result &result) {
         const RowDouble row(RowDouble(3.0f, 666));
         ASSERT(row.get<"head">() == 3.0f, "Value should be 3.0f");
         ASSERT(row.get<"tail">() == 666, "Value should be 666");
     }},
    {"RowDouble_construct4",
     [](Result &result) {
         const RowDouble row(4.0f, Row<Variable<int32_t, "tail">>(666));
         ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
         ASSERT(row.get<"tail">() == 666, "Value should be 666");
     }},
    {"RowDouble_get_set1",
     [](Result &result) {
         RowDouble row{};
         row.get<"head">() = 4.0f;
         row.get<"tail">() = 666;
         ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
         ASSERT(row.get<"tail">() == 666, "Value should be 666");
     }},
    {"RowDouble_get_set2",
     [](Result &result) {
         RowDouble row{};
         row.set<"head">(4.0f);
         row.set<"tail">(666);
         ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
         ASSERT(row.get<"tail">() == 666, "Value should be 666");
     }},
    {"RowDouble_equality1",
     [](Result &result) {
         const RowDouble row{};
         ASSERT(row == RowDouble(), "Rows should be equal");
     }},
    {"RowDouble_equality2",
     [](Result &result) {
         const RowDouble row(1.0f, 16);
         ASSERT(row == RowDouble(1.0f, 16), "Rows should be equal");
     }},
    {"RowDouble_equality3",
     [](Result &result) {
         const RowDouble row(1.0f, 16);
         ASSERT(row == row, "Row should be equal to itself");
     }},
    {"ShouldFailCompilationIfEnabled",
     [](Result &) {
         // Row<Variable<float, "head">, Variable<int32_t, "head">>
         // fail_static_assert;
     }},
    {"StructureOfArrays_getMemReq1",
     [](Result &result) {
         constexpr size_t alignment = 1;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq2",
     [](Result &result) {
         constexpr size_t alignment = 2;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq3",
     [](Result &result) {
         constexpr size_t alignment = 4;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq4",
     [](Result &result) {
         constexpr size_t alignment = 8;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq5",
     [](Result &result) {
         constexpr size_t alignment = 16;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq6",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq7",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1024;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == n * (sizeof(double) + sizeof(float)) + alignment,
                "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq8",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 1000;
         const size_t mem_req = MemReqSoa<alignment>::getMemReq(n);
         ASSERT(mem_req == 8064 + 4096 + alignment,
                "Memory requirement mismatch");
     }},
    {"StructureOfArrays_getMemReq9",
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
    {"StructureOfArrays_getMemReq10",
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
    {"StructureOfArrays_getMemReq11",
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
    {"StructureOfArrays_getMemReq_BigType",
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
    {"StructureOfArrays_construction1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         const std::vector<Ball> init(n);
         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get(i) == Ball{},
                    "Balls should contain default initialized balls");
         }
     }},
    {"StructureOfArrays_construction2",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);

         for (size_t i = 0; i < accessor.size(); i++) {
             ASSERT(accessor.get(i) == Ball{},
                    "Balls should contain default initialized balls");
         }
     }},
    {"StructureOfArrays_construction3",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init(n);
         {
             size_t i = 0;
             for (auto &ball : init) {
                 const auto di = static_cast<double>(i);
                 const auto fi = static_cast<float>(i);
                 const auto ui = static_cast<uint32_t>(i);
                 const auto ii = static_cast<int32_t>(i);
                 const auto bi = static_cast<bool>(i);

                 ball.get<"position_x">() = di;
                 ball.get<"position_y">() = di;
                 ball.get<"position_z">() = di;
                 ball.get<"radius">() = di;
                 ball.get<"color_r">() = fi;
                 ball.get<"color_g">() = fi;
                 ball.get<"color_b">() = fi;
                 ball.get<"index">() = ui;
                 ball.get<"index_distance">() = ii;
                 ball.get<"is_visible">() = bi;

                 i++;
             }
         }
         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);

         for (size_t i = 0; i < accessor.size(); i++) {
             const auto di = static_cast<double>(i);
             const auto fi = static_cast<float>(i);
             const auto ui = static_cast<uint32_t>(i);
             const auto ii = static_cast<int32_t>(i);
             const auto bi = static_cast<bool>(i);
             ASSERT(accessor.get(i) ==
                        Ball(di, di, di, di, fi, fi, fi, ui, ii, bi),
                    "Balls should contain default initialized balls");
         }
     }},
    {"StructureOfArrays_construction3",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init(n);
         {
             size_t i = 0;
             for (auto &ball : init) {
                 const auto di = static_cast<double>(i);
                 const auto fi = static_cast<float>(i);
                 const auto ui = static_cast<uint32_t>(i);
                 const auto ii = static_cast<int32_t>(i);
                 const auto bi = static_cast<bool>(i);

                 ball.get<"position_x">() = di;
                 ball.get<"position_y">() = di;
                 ball.get<"position_z">() = di;
                 ball.get<"radius">() = di;
                 ball.get<"color_r">() = fi;
                 ball.get<"color_g">() = fi;
                 ball.get<"color_b">() = fi;
                 ball.get<"index">() = ui;
                 ball.get<"index_distance">() = ii;
                 ball.get<"is_visible">() = bi;

                 i++;
             }
         }
         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);

         for (size_t i = 0; i < accessor.size(); i++) {
             const auto di = static_cast<double>(i);
             const auto fi = static_cast<float>(i);
             const auto ui = static_cast<uint32_t>(i);
             const auto ii = static_cast<int32_t>(i);
             const auto bi = static_cast<bool>(i);
             ASSERT(accessor.get(i) ==
                        Ball(di, di, di, di, fi, fi, fi, ui, ii, bi),
                    "Balls should contain default initialized balls");
         }
     }},
    {"StructureOfArrays_getMemReqRow1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init(n);
         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);

         ASSERT(balls.getMemReq<"radius">() == sizeof(double) * n,
                "Radius memory requirement should be n * sizeof(double)");
     }},
    {"StructureOfArrays_getMemReqRow2",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init(n);
         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);
         balls.decreaseBy(28, true);

         ASSERT(
             balls.getMemReq<"radius">() == sizeof(double) * (n - 28),
             "Radius memory requirement should be (n - 28) * sizeof(double)");
     }},
    {"StructureOfArrays_getAlignedBlockSize",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 128;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);

         ASSERT(balls.getAlignedBlockSize() == Balls::getMemReq(n) - alignment,
                "Aligned block size should be equal to total memory "
                "requirement minus alignment");
     }},
    {"StructureOfArrays_alignmentBytes",
     [](Result &result) {
         constexpr size_t alignment = 2048;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init{
             Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
             Ball(0.0, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);
         uint8_t *ptr = static_cast<uint8_t *>(balls.data());
         ptr += balls.getAlignmentBytes();
         const double first = *static_cast<double *>(static_cast<void *>(ptr));

         // printf("%lu\n", balls.getAlignmentBytes());
         ASSERT(first == 666.666, "Incorrect value read from first array");
     }},
    {"StructureOfArrays_decreaseBy1",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 666;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);
         balls.decreaseBy(6);

         ASSERT(accessor.size() == 666,
                "Unupdated accessor should have original size");
     }},
    {"StructureOfArrays_decreaseBy2",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 666;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);
         balls.decreaseBy(6, true);

         ASSERT(accessor.size() == 660,
                "Updated accessor should have updated size");
     }},
    {"StructureOfArrays_decreaseBy3",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 666;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);
         balls.decreaseBy(6);
         balls.updateAccessor();

         ASSERT(accessor.size() == 660,
                "Updated accessor should have updated size");
     }},
    {"StructureOfArrays_decreaseBy4",
     [](Result &result) {
         constexpr size_t alignment = 128;
         constexpr size_t n = 666;
         typedef Balls<alignment> Balls;

         Balls::Accessor accessor;
         Balls balls(memory_ops, n, &accessor);
         balls.decreaseBy(3);
         balls.decreaseBy(3, true);

         ASSERT(accessor.size() == 660,
                "Updated accessor should have updated size");
     }},
    {"StructureOfArrays_swap1",
     [](Result &result) {
         constexpr size_t alignment = 2048;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init{
             Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
             Ball(0.0, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);
         balls.swap<"position_x", "position_y">();
         ASSERT(accessor.get<"position_x">(0) == 666.666,
                "Unupdated swap should not be visible at accessor");
     }},
    {"StructureOfArrays_swap2",
     [](Result &result) {
         constexpr size_t alignment = 2048;
         typedef Ball<alignment> Ball;
         typedef Balls<alignment> Balls;

         std::vector<Ball> init{
             Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
             Ball(0.0, 13.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

         Balls::Accessor accessor;
         Balls balls(memory_ops, init, &accessor);
         balls.swap<"position_x", "position_y">(true);
         ASSERT(accessor.get<"position_y">(0) == 666.666,
                "Updated swap should be visible at accessor");
         ASSERT(accessor.get<"position_x">(1) == 13.0,
                "Updated swap should be visible at accessor");
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
