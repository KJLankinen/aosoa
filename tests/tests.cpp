#include "aosoa.h"
#include "json.hpp"
#include "tabulate/table.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace aosoa;

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
using Pointers = AlignedPointers<Alignment, Variable<float, "head">,
                                 Variable<double, "tail">>;

// clang-format off
template <size_t Alignment, typename MemOps>
using Balls = StructureOfArrays<
            Alignment,
            MemOps,
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

template <size_t Alignment> using CBalls = Balls<Alignment, CMemoryOperations>;
template <size_t Alignment> using Ball = CBalls<Alignment>::FullRow;
template <size_t Alignment>
using BallAccessor = CBalls<Alignment>::ThisAccessor;

const CMemoryOperations memory_ops;

typedef MemoryOperations<true, CAllocator, CDeallocator, CMemcpy, CMemset>
    DummyDeviceMemoryOps;

template <size_t Alignment, CompileTimeString Cts>
void assertUntouchedCorrect(const std::vector<Ball<Alignment>> &init,
                            typename CBalls<Alignment>::ThisAccessor &balls,
                            Result &result) {
    for (size_t i = 0; i < init.size(); i++) {
        ASSERT(balls.template get<Cts>(i) == init[i].template get<Cts>(),
               (Cts + " incorrect").str);
    }
}

template <size_t Alignment, CompileTimeString Cts, CompileTimeString Head,
          CompileTimeString... Tail>
void assertUntouchedCorrect(const std::vector<Ball<Alignment>> &init,
                            typename CBalls<Alignment>::ThisAccessor &balls,
                            Result &result) {
    assertUntouchedCorrect<Alignment, Cts>(init, balls, result);
    if (result.success) {
        assertUntouchedCorrect<Alignment, Head>(init, balls, result);
        if constexpr (sizeof...(Tail) > 0) {
            if (result.success) {
                assertUntouchedCorrect<Alignment, Tail...>(init, balls, result);
            }
        }
    }
}

template <size_t A> void assertAligned(Pointers<A> &pointers, Result &result) {
    constexpr size_t max_size_t = ~0ul;
    size_t space = max_size_t;
    for (size_t i = 0; i < pointers.size; i++) {
        void *ptr = pointers[i];
        std::align(Pointers<A>::getAlignment(), 1, ptr, space);
        ASSERT(space == max_size_t,
               std::string("Incorrect alignment for ") + std::to_string(i));
    }
}

constexpr static std::array test_arr = {
    Test("sizeof(RowSingle)",
         [](Result &) { static_assert(sizeof(RowSingle) == sizeof(float)); }),
    Test("sizeof(RowDouble)",
         [](Result &) {
             static_assert(sizeof(RowDouble) ==
                           sizeof(float) + sizeof(int32_t));
         }),
    Test("RowSingle_construct1",
         [](Result &result) {
             ASSERT(RowSingle().get<"head">() == 0.0f,
                    "Default value should be 0.0f");
         }),
    Test("RowSingle_construct2",
         [](Result &result) {
             ASSERT(RowSingle(1.0f).get<"head">() == 1.0f,
                    "Value should be 1.0f");
         }),
    Test("RowSingle_construct3",
         [](Result &result) {
             ASSERT(RowSingle(RowSingle(2.0f)).get<"head">() == 2.0f,
                    "Value should be 2.0f");
         }),
    Test("RowSingle_get_set1",
         [](Result &result) {
             RowSingle row;
             row.get<"head">() = 10.0f;
             ASSERT(row.get<"head">() == 10.0f, "Value should be 10.0f");
         }),
    Test("RowSingle_get_set2",
         [](Result &result) {
             RowSingle row;
             row.set<"head">(10.0f);
             ASSERT(row.get<"head">() == 10.0f, "Value should be 10.0f");
         }),
    Test("RowSingle_getconst",
         [](Result &result) {
             const RowSingle row(666.666f);
             const float val = row.get<"head">();
             ASSERT(val == 666.666f, "Value should be 666.666f");
         }),
    Test("RowSingle_equality1",
         [](Result &result) {
             const RowSingle row(666.666f);
             ASSERT(row == RowSingle(666.666f), "Values should be equal");
         }),
    Test("RowSingle_equality2",
         [](Result &result) {
             const RowSingle row{};
             ASSERT(row == RowSingle(0.0f), "Values should be equal");
         }),
    Test("RowSingle_equality3",
         [](Result &result) {
             const RowSingle row{};
             ASSERT(row == row, "Row should be equal to itself");
         }),
    Test("RowDouble_construct1",
         [](Result &result) {
             const RowDouble row{};
             ASSERT(row.get<"head">() == 0.0f, "Default value should be 0.0f");
             ASSERT(row.get<"tail">() == 0, "Default value should be 0");
         }),
    Test("RowDouble_construct2",
         [](Result &result) {
             const RowDouble row(1.0f, 2);
             ASSERT(row.get<"head">() == 1.0f, "Value should be 1.0f");
             ASSERT(row.get<"tail">() == 2, "Value should be 2");
         }),
    Test("RowDouble_construct3",
         [](Result &result) {
             const RowDouble row(RowDouble(3.0f, 666));
             ASSERT(row.get<"head">() == 3.0f, "Value should be 3.0f");
             ASSERT(row.get<"tail">() == 666, "Value should be 666");
         }),
    Test("RowDouble_construct4",
         [](Result &result) {
             const RowDouble row(4.0f, Row<Variable<int32_t, "tail">>(666));
             ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
             ASSERT(row.get<"tail">() == 666, "Value should be 666");
         }),
    Test("RowDouble_get_set1",
         [](Result &result) {
             RowDouble row{};
             row.get<"head">() = 4.0f;
             row.get<"tail">() = 666;
             ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
             ASSERT(row.get<"tail">() == 666, "Value should be 666");
         }),
    Test("RowDouble_get_set2",
         [](Result &result) {
             RowDouble row{};
             row.set<"head">(4.0f);
             row.set<"tail">(666);
             ASSERT(row.get<"head">() == 4.0f, "Value should be 4.0f");
             ASSERT(row.get<"tail">() == 666, "Value should be 666");
         }),
    Test("RowDouble_equality1",
         [](Result &result) {
             const RowDouble row{};
             ASSERT(row == RowDouble(), "Rows should be equal");
         }),
    Test("RowDouble_equality2",
         [](Result &result) {
             const RowDouble row(1.0f, 16);
             ASSERT(row == RowDouble(1.0f, 16), "Rows should be equal");
         }),
    Test("RowDouble_equality3",
         [](Result &result) {
             const RowDouble row(1.0f, 16);
             ASSERT(row == row, "Row should be equal to itself");
         }),
    Test("Row_to_from_json",
         [](Result &result) {
             using Row =
                 const Row<Variable<float, "foo">, Variable<double, "bar">,
                           Variable<int, "baz">, Variable<bool, "foobar">>;

             const Row row(1.0f, 10.0, -666, true);
             const nlohmann::json j = row;
             const Row row2 = j;

             ASSERT(row == row2, "Row should be equal to itself round tripped "
                                 "through json conversion");
         }),
    Test("ShouldFailCompilationIfEnabled",
         [](Result &) {
             // Row<Variable<float, "head">, Variable<int32_t, "head">>
             // fail_static_assert;
         }),
    Test("StructureOfArrays_construction1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             const std::vector<Ball> init(n);
             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             for (size_t i = 0; i < accessor.size(); i++) {
                 ASSERT(accessor.get(i) == Ball{},
                        "Ball at index " + std::to_string(i) +
                            " should be equal to Ball{} but is not");
             }
         }),
    Test("StructureOfArrays_construction2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n, &accessor);

             for (size_t i = 0; i < accessor.size(); i++) {
                 ASSERT(accessor.get(i) == Ball{},
                        "Ball at index " + std::to_string(i) +
                            " should be equal to Ball{} but is not");
             }
         }),
    Test("StructureOfArrays_construction3",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

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
             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             for (size_t i = 0; i < accessor.size(); i++) {
                 const auto di = static_cast<double>(i);
                 const auto fi = static_cast<float>(i);
                 const auto ui = static_cast<uint32_t>(i);
                 const auto ii = static_cast<int32_t>(i);
                 const auto bi = static_cast<bool>(i);
                 ASSERT(accessor.get(i) ==
                            Ball(di, di, di, di, fi, fi, fi, ui, ii, bi),
                        "Ball at index " + std::to_string(i) +
                            " contains incorrect data");
             }
         }),
    Test("StructureOfArrays_construction4",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef Balls<alignment, DummyDeviceMemoryOps> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
             };

             // Using dummy memory ops that uses another copy
             const DummyDeviceMemoryOps dummy_memory_ops;
             Balls::ThisAccessor accessor;
             Balls balls(dummy_memory_ops, init, &accessor);

             for (size_t i = 0; i < init.size(); i++) {
                 ASSERT(accessor.get(i) == init[i],
                        "accessor.get(i) != init[i] at index " +
                            std::to_string(i));
             }
         }),
    Test("StructureOfArrays_construction5",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             const std::vector<Ball> init(n);
             Balls balls(memory_ops, init);

             for (size_t i = 0; i < balls.getAccess().size(); i++) {
                 ASSERT(balls.getAccess().get(i) == Ball{},
                        "Ball at index " + std::to_string(i) +
                            " should be equal to Ball{} but is not");
             }
         }),
    Test("StructureOfArrays_construction6",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             const std::vector<Ball> init(n);
             const Balls balls(memory_ops, init);

             for (size_t i = 0; i < balls.getAccess().size(); i++) {
                 ASSERT(balls.getAccess().get(i) == Ball{},
                        "Ball at index " + std::to_string(i) +
                            " should be equal to Ball{} but is not");
             }
         }),
    Test("StructureOfArrays_construction7",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

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
             Balls balls(memory_ops, init);

             for (size_t i = 0; i < balls.getAccess().size(); i++) {
                 const auto di = static_cast<double>(i);
                 const auto fi = static_cast<float>(i);
                 const auto ui = static_cast<uint32_t>(i);
                 const auto ii = static_cast<int32_t>(i);
                 const auto bi = static_cast<bool>(i);
                 ASSERT(balls.getAccess().get(i) ==
                            Ball(di, di, di, di, fi, fi, fi, ui, ii, bi),
                        "Ball at index " + std::to_string(i) +
                            " contains incorrect data");
             }
         }),
    Test("StructureOfArrays_decreaseBy1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n, &accessor);
             balls.decreaseBy(6);

             ASSERT(accessor.size() == 666,
                    "Unupdated accessor should have original size");
         }),
    Test("StructureOfArrays_decreaseBy2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n);
             balls.decreaseBy(6, &accessor);

             ASSERT(accessor.size() == 660,
                    "Updated accessor should have updated size");
         }),
    Test("StructureOfArrays_decreaseBy3",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n);
             balls.decreaseBy(6);
             balls.updateAccessor(&accessor);

             ASSERT(accessor.size() == 660,
                    "Updated accessor should have updated size");
         }),
    Test("StructureOfArrays_decreaseBy4",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n);
             balls.decreaseBy(3);
             balls.decreaseBy(3, &accessor);

             ASSERT(accessor.size() == 660,
                    "Updated accessor should have updated size");
         }),
    Test("StructureOfArrays_decreaseBy5",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls balls(memory_ops, n);
             balls.decreaseBy(6);

             ASSERT(balls.getAccess().size() == 660,
                    "Updated accessor should have reduced size");
         }),
    Test("StructureOfArrays_swap1",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
                 Ball(0.0, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.swap<"position_x", "position_y">();
             ASSERT(accessor.get<"position_x">(0) == 666.666,
                    "Unupdated swap should not be visible at accessor");
         }),
    Test("StructureOfArrays_swap2",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
                 Ball(0.0, 13.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.swap<"position_x", "position_y">(&accessor);
             ASSERT(accessor.get<"position_y">(0) == 666.666,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_x">(1) == 13.0,
                    "Updated swap should be visible at accessor");
         }),
    Test("StructureOfArrays_swap3",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
                 Ball(0.0, 13.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.swap<"position_x", "position_y">();
             balls.updateAccessor(&accessor);

             ASSERT(accessor.get<"position_y">(0) == 666.666,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_x">(1) == 13.0,
                    "Updated swap should be visible at accessor");
         }),
    Test("StructureOfArrays_swap4",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             balls.swap<"position_x", "position_y", "position_z", "radius">(
                 &accessor);

             ASSERT(accessor.get<"position_x">(0) == 1.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(0) == 0.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(0) == 3.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"radius">(0) == 2.0,
                    "Updated swap should be visible at accessor");

             ASSERT(accessor.get<"position_x">(1) == 10.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(1) == 9.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(1) == 12.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"radius">(1) == 11.0,
                    "Updated swap should be visible at accessor");
         }),
    Test("StructureOfArrays_swap5",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             balls.swap<"position_x", "position_y", "position_y", "position_z">(
                 &accessor);

             ASSERT(accessor.get<"position_x">(0) == 1.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(0) == 2.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(0) == 0.0,
                    "Updated swap should be visible at accessor");

             ASSERT(accessor.get<"position_x">(1) == 10.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(1) == 11.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(1) == 9.0,
                    "Updated swap should be visible at accessor");
         }),
    Test("StructureOfArrays_swap6",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             balls.swap<"position_x", "position_y", "position_y",
                        "position_z">();

             ASSERT(accessor.get<"position_x">(0) == 0.0,
                    "Un updated swap should not be visible at accessor");
             ASSERT(accessor.get<"position_y">(0) == 1.0,
                    "Un updated swap should not be visible at accessor");
             ASSERT(accessor.get<"position_z">(0) == 2.0,
                    "Un updated swap should not be visible at accessor");

             ASSERT(accessor.get<"position_x">(1) == 9.0,
                    "Un updated swap should not be visible at accessor");
             ASSERT(accessor.get<"position_y">(1) == 10.0,
                    "Un updated swap should not be visible at accessor");
             ASSERT(accessor.get<"position_z">(1) == 11.0,
                    "Un updated swap should not be visible at accessor");
         }),
    Test("StructureOfArrays_swap7",
         [](Result &result) {
             constexpr size_t alignment = 2048;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);

             balls.swap<"position_x", "position_y", "position_y",
                        "position_z">();
             balls.updateAccessor(&accessor);

             ASSERT(accessor.get<"position_x">(0) == 1.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(0) == 2.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(0) == 0.0,
                    "Updated swap should be visible at accessor");

             ASSERT(accessor.get<"position_x">(1) == 10.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_y">(1) == 11.0,
                    "Updated swap should be visible at accessor");
             ASSERT(accessor.get<"position_z">(1) == 9.0,
                    "Updated swap should be visible at accessor");
         }),
    Test("StructureOfArrays_updateAccessor1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 666;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, n, &accessor);
             balls.decreaseBy(6);
             balls.updateAccessor(std::memcpy, &accessor);

             ASSERT(accessor.size() == 660,
                    "Updated accessor should have updated size");
         }),
    Test("StructureOfArrays_getRows1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             const std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
             };

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             const auto rows = balls.getRows();

             ASSERT(rows.size() == init.size(),
                    "Initial data and copied data sizes are not equal");

             for (size_t i = 0; i < rows.size(); i++) {
                 ASSERT(rows[i] == init[i],
                        "rows[i] != init[i] at index " + std::to_string(i));
             }
         }),
    Test("StructureOfArrays_getRows2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef Balls<alignment, DummyDeviceMemoryOps> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
             };

             // Using dummy memory ops that uses another copy
             const DummyDeviceMemoryOps dummy_memory_ops;
             Balls::ThisAccessor accessor;
             Balls balls(dummy_memory_ops, init, &accessor);
             const auto rows = balls.getRows();

             ASSERT(rows.size() == init.size(),
                    "Initial data and copied data sizes are not equal");

             std::string msg = "";
             for (size_t i = 0; i < rows.size(); i++) {
                 ASSERT(rows[i] == init[i],
                        "rows[i] != init[i] at index " + std::to_string(i));
             }
         }),
    Test("StructureOfArrays_memcpy_internal_to_internal1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.memcpy<"position_y", "position_x">();

             ASSERT(accessor.get<"position_x">(0) == 0.0,
                    "position_x[0] incorrect");
             ASSERT(accessor.get<"position_x">(1) == 9.0,
                    "position_x[1] incorrect");
             ASSERT(accessor.get<"position_y">(0) == 0.0,
                    "position_y[0] incorrect");
             ASSERT(accessor.get<"position_y">(1) == 9.0,
                    "position_y[1] incorrect");

             assertUntouchedCorrect<alignment, "position_z", "radius",
                                    "color_r", "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memcpy_internal_to_internal2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.memcpy<"position_y", "position_x">(std::memcpy);

             ASSERT(accessor.get<"position_x">(0) == 0.0,
                    "position_x[0] incorrect");
             ASSERT(accessor.get<"position_x">(1) == 9.0,
                    "position_x[1] incorrect");
             ASSERT(accessor.get<"position_y">(0) == 0.0,
                    "position_y[0] incorrect");
             ASSERT(accessor.get<"position_y">(1) == 9.0,
                    "position_y[1] incorrect");

             assertUntouchedCorrect<alignment, "position_z", "radius",
                                    "color_r", "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memcpy_internal_compile_fail_if_enabled",
         [](Result &) {
             constexpr size_t alignment = 128;
             typedef CBalls<alignment> Balls;

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, 1, &accessor);
             // Uncommenting the line below should give a compiler error
             // balls.memcpy<"position_x", "position_x">(std::memcpy);
         }),
    Test("StructureOfArrays_memcpy_external_to_internal1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             const std::vector<double> src{0.0, 9.0};
             balls.memcpy<"position_y">(src.data());

             ASSERT(accessor.get<"position_x">(0) == 0.0,
                    "position_x[0] incorrect");
             ASSERT(accessor.get<"position_x">(1) == 9.0,
                    "position_x[1] incorrect");
             ASSERT(accessor.get<"position_y">(0) == 0.0,
                    "position_y[0] incorrect");
             ASSERT(accessor.get<"position_y">(1) == 9.0,
                    "position_y[1] incorrect");

             assertUntouchedCorrect<alignment, "position_z", "radius",
                                    "color_r", "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memcpy_external_to_internal2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             const std::vector<double> src{0.0, 9.0};
             balls.memcpy<"position_y">(src.data());

             ASSERT(accessor.get<"position_x">(0) == 0.0,
                    "position_x[0] incorrect");
             ASSERT(accessor.get<"position_x">(1) == 9.0,
                    "position_x[1] incorrect");
             ASSERT(accessor.get<"position_y">(0) == 0.0,
                    "position_y[0] incorrect");
             ASSERT(accessor.get<"position_y">(1) == 9.0,
                    "position_y[1] incorrect");

             assertUntouchedCorrect<alignment, "position_z", "radius",
                                    "color_r", "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memcpy_internal_to_external1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             std::vector<double> dst{666.0, 666.0};
             balls.memcpy<"position_y">(dst.data());

             ASSERT(dst[0] == init[0].get<"position_y">(), "dst[0] incorrect");
             ASSERT(dst[1] == init[1].get<"position_y">(), "dst[1] incorrect");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memcpy_internal_to_external2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             std::vector<double> dst{666.0, 666.0};
             balls.memcpy<"position_y">(std::memcpy, dst.data());

             ASSERT(dst[0] == init[0].get<"position_y">(), "dst[0] incorrect");
             ASSERT(dst[1] == init[1].get<"position_y">(), "dst[1] incorrect");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index",
                                    "index_distance", "is_visible">(
                 init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memset1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.memset<"index">(0);
             ASSERT(accessor.get<"index">(0) == 0, "index[0] set incorrectly");
             ASSERT(accessor.get<"index">(1) == 0, "index[1] set incorrectly");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index_distance",
                                    "is_visible">(init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memset2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17,
                      true)};

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.memset<"index">(std::memset, 0);
             ASSERT(accessor.get<"index">(0) == 0, "index[0] set incorrectly");
             ASSERT(accessor.get<"index">(1) == 0, "index[1] set incorrectly");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index_distance",
                                    "is_visible">(init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memset3",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
             };

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             balls.decreaseBy(10, &accessor);
             balls.memset<"index">(0);

             // Only the first should be set, since size was decreased
             ASSERT(accessor.get<"index">(0) == 0, "index[0] set incorrectly");
             ASSERT(accessor.get<"index">(1) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(2) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(3) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(4) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(5) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(6) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(7) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(8) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(9) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(10) == 7u,
                    "index[1] set incorrectly");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index_distance",
                                    "is_visible">(init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("StructureOfArrays_memset4",
         [](Result &result) {
             constexpr size_t alignment = 128;
             typedef Ball<alignment> Ball;
             typedef CBalls<alignment> Balls;

             std::vector<Ball> init{
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
                 Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
             };

             Balls::ThisAccessor accessor;
             Balls balls(memory_ops, init, &accessor);
             // This is false, so accessor is not updated, but still the memset
             // should've gone through
             balls.decreaseBy(10);
             balls.memset<"index">(0);

             // Only the first should be set, since size was decreased
             ASSERT(accessor.get<"index">(0) == 0, "index[0] set incorrectly");
             ASSERT(accessor.get<"index">(1) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(2) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(3) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(4) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(5) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(6) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(7) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(8) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(9) == 7u, "index[1] set incorrectly");
             ASSERT(accessor.get<"index">(10) == 7u,
                    "index[1] set incorrectly");

             assertUntouchedCorrect<alignment, "position_x", "position_y",
                                    "position_z", "radius", "color_r",
                                    "color_g", "color_b", "index_distance",
                                    "is_visible">(init, accessor, result);
             ASSERT(result.success, result.msg);
         }),
    Test("NthType",
         [](Result &) {
             static_assert(
                 std::is_same<typename NthType<0, double, float, int>::Type,
                              double>::value,
                 "0th type should be double");
             static_assert(
                 std::is_same<typename NthType<1, double, float, int>::Type,
                              float>::value,
                 "1st type should be float");
             static_assert(
                 std::is_same<typename NthType<2, double, float, int>::Type,
                              int>::value,
                 "2nd type should be int");
         }),
    Test("IndexOfString",
         [](Result &) {
             static_assert(0ul == IndexOfString<"foo"_cts, "foo"_cts, "bar"_cts,
                                                "baz"_cts>::i,
                           "Index of foo should be 0");
             static_assert(1ul == IndexOfString<"bar"_cts, "foo"_cts, "bar"_cts,
                                                "baz"_cts>::i,
                           "Index of bar should be 1");
             static_assert(2ul == IndexOfString<"baz"_cts, "foo"_cts, "bar"_cts,
                                                "baz"_cts>::i,
                           "Index of baz should be 2");
             static_assert(~0ul == IndexOfString<"nope"_cts, "foo"_cts,
                                                 "bar"_cts, "baz"_cts>::i,
                           "Index of nope should be ~0ul");
         }),
    Test("FindString",
         [](Result &) {
             static_assert(
                 FindString<"foo"_cts>::template From<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                 "foo should be found");
             static_assert(
                 FindString<"bar"_cts>::template From<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                 "bar should be found");
             static_assert(
                 FindString<"baz"_cts>::template From<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                 "baz should be found");
             static_assert(!FindString<"not_found"_cts>::template From<
                               "foo"_cts, "bar"_cts, "baz"_cts>::value,
                           "not_found should not be found");
         }),
    Test("FailAllocation_length_error",
         [](Result &result) {
             try {
                 constexpr size_t alignment = 128;
                 typedef Ball<alignment> Ball;
                 typedef CBalls<alignment> Balls;
                 std::vector<Ball> init;
                 const size_t n = init.max_size() + 1;

                 Balls::ThisAccessor accessor;
                 Balls balls(memory_ops, std::vector<Ball>(n), &accessor);
             } catch (std::length_error &e) {
                 constexpr CompileTimeString substr = "max_size"_cts;
                 const auto pos = std::string(e.what()).find(substr.str);
                 ASSERT(
                     pos != std::string::npos,
                     ("Exception should contain the substr \"" + substr + "\"")
                         .str);
             } catch (const std::exception &e) {
                 ASSERT(false, std::string("Unhandled exception: ") + e.what());
             }
         }),
    Test("FailAllocation_bad_alloc",
         [](Result &result) {
             try {
                 constexpr size_t alignment = 128;
                 typedef Ball<alignment> Ball;
                 typedef CBalls<alignment> Balls;
                 std::vector<Ball> init;
                 const size_t n = init.max_size();

                 Balls::ThisAccessor accessor;
                 Balls balls(memory_ops, std::vector<Ball>(n), &accessor);
             } catch (std::bad_alloc &e) {
                 constexpr CompileTimeString substr = "bad_alloc"_cts;
                 const auto pos = std::string(e.what()).find(substr.str);
                 ASSERT(
                     pos != std::string::npos,
                     ("Exception should contain the substr \"" + substr + "\"")
                         .str);
             } catch (const std::exception &e) {
                 ASSERT(false, std::string("Unhandled exception: ") + e.what());
             }
         }),
    Test("AlignedPointers_construct1",
         [](Result &result) {
             constexpr size_t alignment = 256;
             using Pointers = Pointers<alignment>;
             static_assert(Pointers::size == 2,
                           "Expecting Pointers type to have two variables");

             const Pointers pointers = {};
             ASSERT(pointers[0] == nullptr,
                    "Default constructed pointers array "
                    "should contain only nullptrs");
             ASSERT(pointers[1] == nullptr,
                    "Default constructed pointers array "
                    "should contain only nullptrs");
         }),
    Test("AlignedPointers_construct2",
         [](Result &result) {
             constexpr size_t alignment = 256;
             using Pointers = Pointers<alignment>;
             static_assert(Pointers::getAlignment() == alignment,
                           "Alignment should be 256");

             constexpr size_t n = 100;
             const size_t bytes = Pointers::getMemReq(n);
             std::vector<uint8_t> data(bytes);

             Pointers pointers(n, data.data());
             assertAligned(pointers, result);
             ASSERT(result.success, result.msg);
         }),
    Test("AlignedPointers_alignment",
         [](Result &) {
             static_assert(Pointers<256>::getAlignment() == 256,
                           "Incorrect alignment");
             static_assert(Pointers<128>::getAlignment() == 128,
                           "Incorrect alignment");
             static_assert(Pointers<64>::getAlignment() == 64,
                           "Incorrect alignment");
             static_assert(Pointers<32>::getAlignment() == 32,
                           "Incorrect alignment");
             static_assert(Pointers<16>::getAlignment() == 16,
                           "Incorrect alignment");
             static_assert(Pointers<8>::getAlignment() == 8,
                           "Incorrect alignment");
             static_assert(Pointers<4>::getAlignment() == 8,
                           "Incorrect alignment");
             static_assert(Pointers<2>::getAlignment() == 8,
                           "Incorrect alignment");
             static_assert(Pointers<1>::getAlignment() == 8,
                           "Incorrect alignment");
         }),
    Test("AlignedPointers_getMemReq1",
         [](Result &result) {
             constexpr size_t alignment = 1;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq2",
         [](Result &result) {
             constexpr size_t alignment = 2;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq3",
         [](Result &result) {
             constexpr size_t alignment = 4;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * 8, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq4",
         [](Result &result) {
             constexpr size_t alignment = 8;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq5",
         [](Result &result) {
             constexpr size_t alignment = 16;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq6",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 3 * alignment, "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq7",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1024;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == n * (sizeof(double) + sizeof(float)) + alignment,
                    "Memory requirement mismatch");
         }),
    Test("AlignedPointers_getMemReq8",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1000;
             const size_t mem_req = Pointers<alignment>::getMemReq(n);
             ASSERT(mem_req == 8064 + 4096 + alignment,
                    "Memory requirement mismatch");
         }),
    Test(
        "AlignedPointers_getMemReq9",
        [](Result &result) {
            constexpr size_t alignment = 128;
            constexpr size_t n = 1000;
            typedef AlignedPointers<
                alignment, Variable<double, "first">, Variable<float, "second">,
                Variable<int, "third">, Variable<bool, "fourth">,
                Variable<float, "fifth">>
                Pointers;
            const size_t mem_req = Pointers::getMemReq(n);
            ASSERT(mem_req == 8064 + 4096 + 4096 + 1024 + 4096 + alignment,
                   "Memory requirement mismatch");
        }),
    Test(
        "AlignedPointers_getMemReq10",
        [](Result &result) {
            constexpr size_t alignment = 128;
            constexpr size_t n = 3216547;
            typedef AlignedPointers<
                alignment, Variable<double, "first">, Variable<char, "second">,
                Variable<int, "third">, Variable<bool, "fourth">,
                Variable<float, "fifth">>
                Pointers;
            const size_t mem_req = Pointers::getMemReq(n);
            ASSERT((mem_req & (alignment - 1)) == 0,
                   "Total memory requirement must be a multiple of alignment");
        }),
    Test(
        "AlignedPointers_getMemReq11",
        [](Result &result) {
            constexpr size_t alignment = 32;
            constexpr size_t n = 3216547;
            typedef AlignedPointers<
                alignment, Variable<double, "first">, Variable<char, "second">,
                Variable<int, "third">, Variable<bool, "fourth">,
                Variable<float, "fifth">>
                Pointers;
            const size_t mem_req = Pointers::getMemReq(n);
            ASSERT((mem_req & (alignment - 1)) == 0,
                   "Total memory requirement must be a multiple of alignment");
        }),
    Test("AlignedPointers_getMemReq_BigType",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 100;

             typedef AlignedPointers<
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
                 Variable<float, "23">, Variable<float, "24">,
                 Variable<int, "25">, Variable<int, "26">, Variable<int, "27">,
                 Variable<int, "28">, Variable<int, "29">, Variable<int, "30">,
                 Variable<int, "31">, Variable<int, "32">, Variable<int, "33">,
                 Variable<int, "34">, Variable<int, "35">, Variable<int, "36">,
                 Variable<int, "37">, Variable<int, "38">, Variable<int, "39">,
                 Variable<int, "40">, Variable<int, "41">, Variable<int, "42">,
                 Variable<int, "43">, Variable<int, "44">, Variable<int, "45">,
                 Variable<int, "46">, Variable<int, "47">, Variable<int, "48">,
                 Variable<int, "49">, Variable<int, "50">, Variable<bool, "51">,
                 Variable<bool, "52">, Variable<bool, "53">,
                 Variable<bool, "54">, Variable<bool, "55">,
                 Variable<bool, "56">, Variable<bool, "57">,
                 Variable<bool, "58">, Variable<bool, "59">,
                 Variable<bool, "60">, Variable<bool, "61">,
                 Variable<bool, "62">, Variable<bool, "63">,
                 Variable<char, "64">, Variable<char, "65">,
                 Variable<char, "66">, Variable<char, "67">,
                 Variable<char, "68">, Variable<char, "69">,
                 Variable<char, "70">, Variable<char, "71">,
                 Variable<char, "72">, Variable<char, "73">>
                 Pointers;

             const size_t mem_req = Pointers::getMemReq(n);
             ASSERT((mem_req & (alignment - 1)) == 0,
                    "Total memory requirement must be a multiple of alignment");
         }),
    Test("AlignedPointers_Accessor_construction1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             using Accessor = BallAccessor<alignment>;
             const Accessor accessor = {};

             ASSERT(accessor.get<"position_x">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"position_y">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"position_z">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"radius">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"color_r">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"color_g">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"color_b">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"index">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"index_distance">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
             ASSERT(accessor.get<"is_visible">() == nullptr,
                    "Default constructed accessor should have only nullptrs");
         }),
    Test("AlignedPointers_Accessor_get1",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             Accessor accessor(n, data.data());
             accessor.set<"position_x">(0, 666.666);

             ASSERT(accessor.get<"position_x">()[0] == 666.666,
                    "Values from get should be equal 1");
             ASSERT(accessor.get<"position_x">()[0] ==
                        accessor.get<"position_x">(0),
                    "Values from get should be equal 2");
             ASSERT((accessor.get<"position_x">()[0] ==
                     accessor.get<"position_x", 0>()),
                    "Values from get should be equal 3");
         }),
    Test("AlignedPointers_Accessor_get2",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             const Accessor accessor(n, data.data());

             ASSERT(accessor.get<"position_x">() == accessor.get<0>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"position_y">() == accessor.get<1>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"position_z">() == accessor.get<2>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"radius">() == accessor.get<3>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"color_r">() == accessor.get<4>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"color_g">() == accessor.get<5>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"color_b">() == accessor.get<6>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"index">() == accessor.get<7>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"index_distance">() == accessor.get<8>(),
                    "Pointers from get should be equal");
             ASSERT(accessor.get<"is_visible">() == accessor.get<9>(),
                    "Pointers from get should be equal");
         }),
    Test("AlignedPointers_Accessor_get3",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             Accessor accessor(n, data.data());

             const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f,
                                         8u, 9, true);
             accessor.set(666, row);
             ASSERT(accessor.get(666) == row, "Row should be equal to set row");
         }),
    Test("AlignedPointers_Accessor_get4",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             Accessor accessor(n, data.data());

             const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f,
                                         8u, 9, true);
             accessor.set(666, row);
             ASSERT(accessor.get<0>()[666] == row.get<"position_x">(),
                    "Values should be equal");
             ASSERT(accessor.get<"position_x">()[666] ==
                        row.get<"position_x">(),
                    "Values should be equal");
             ASSERT(accessor.get<"position_x">(666) == row.get<"position_x">(),
                    "Values should be equal");
             ASSERT(
                 (accessor.get<"position_x", 666>() == row.get<"position_x">()),
                 "Values should be equal");
         }),
    Test("AlignedPointers_Accessor_get5",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             Accessor accessor(n, data.data());

             const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f,
                                         8u, 9, true);
             accessor.set(666, row);
             accessor.set<"radius">(666, 666.0);
             ASSERT(accessor.get<"radius">(666) == 666.0,
                    "Radius should be changed");
         }),
    Test("AlignedPointers_Accessor_get6",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(Accessor::getMemReq(n));
             Accessor accessor(n, data.data());

             const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f,
                                         8u, 9, true);
             accessor.set(666, row);
             accessor.get<"radius">()[666] = 666.0;
             ASSERT(accessor.get<"radius">(666) == 666.0,
                    "Radius should be changed");
         }),
    Test("AlignedPointers_Accessor_size",
         [](Result &result) {
             constexpr size_t alignment = 128;
             constexpr size_t n = 1189;
             using Accessor = BallAccessor<alignment>;
             std::vector<uint8_t> data(CBalls<alignment>::getMemReq(n));
             const Accessor accessor(n, data.data());

             ASSERT(accessor.size() == n, "Accessor size should be equal to n");
         }),
};

int tests(int, char **) {
    tabulate::Table successful_tests;
    tabulate::Table failed_tests;

    for (auto [test_name, test] : test_arr) {
        Result result{};
        test(result);
        if (result.success) {
            successful_tests.add_row({"OK", test_name});
        } else {
            failed_tests.add_row({"FAIL", test_name, "\"" + result.msg + "\""});
        }
    }

    successful_tests.format()
        .border_top("")
        .border_bottom("")
        .border_left("")
        .border_right("")
        .corner("");

    failed_tests.format()
        .border_top("")
        .border_bottom("")
        .border_left("")
        .border_right("")
        .corner("");

    successful_tests.column(0).format().font_color(tabulate::Color::green);
    failed_tests.column(0).format().font_color(tabulate::Color::red);

    std::cout << "Successful test_arr: " << successful_tests.size() << "/"
              << test_arr.size() << "\n"
              << successful_tests << "\n"
              << std::endl;
    std::cout << "Failed tests: " << failed_tests.size() << "/"
              << test_arr.size() << "\n"
              << failed_tests << std::endl;

    return 0;
}
