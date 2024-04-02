#include "aosoa.h"
#include "balls.h"

#include <gtest/gtest.h>

CMemoryOperations memory_ops;

typedef MemoryOperations<true, CAllocator, CDeallocator, CMemcpy, CMemset>
    DummyDeviceMemoryOps;

template <size_t Alignment, CompileTimeString Cts>
std::tuple<std::string, bool>
assertUntouchedCorrect(const std::vector<Ball<Alignment>> &init,
                       typename CBalls<Alignment>::ThisAccessor &balls) {
    bool result = true;
    size_t i = 0;
    for (; i < init.size() && result; i++) {
        result = balls.template get<Cts>(i) == init[i].template get<Cts>();
    }

    return std::make_tuple(std::string((Cts + " incorrect at index ").str) +
                               std::to_string(i),
                           result);
}

template <size_t Alignment, CompileTimeString Cts, CompileTimeString Head,
          CompileTimeString... Tail>
std::tuple<std::string, bool>
assertUntouchedCorrect(const std::vector<Ball<Alignment>> &init,
                       typename CBalls<Alignment>::ThisAccessor &balls) {
    auto [str, result] = assertUntouchedCorrect<Alignment, Cts>(init, balls);
    if (result) {
        auto [str, result] =
            assertUntouchedCorrect<Alignment, Head>(init, balls);
        if constexpr (sizeof...(Tail) > 0) {
            if (result) {
                auto [str, result] =
                    assertUntouchedCorrect<Alignment, Tail...>(init, balls);
            }
        }
    }

    return std::make_tuple(str, result);
}

TEST(aosoa_test, StructureOfArrays_construction1) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    const std::vector<Ball> init(n);
    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);

    for (size_t i = 0; i < accessor.size(); i++) {
        ASSERT_EQ(accessor.get(i), Ball{})
            << "Ball at index " + std::to_string(i) +
                   " should be equal to Ball{} but is not";
    }
}

TEST(aosoa_test, StructureOfArrays_construction2) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n, &accessor);

    for (size_t i = 0; i < accessor.size(); i++) {
        ASSERT_EQ(accessor.get(i), Ball{})
            << "Ball at index " + std::to_string(i) +
                   " should be equal to Ball{} but is not";
    }
}

TEST(aosoa_test, StructureOfArrays_construction3) {
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
        ASSERT_EQ(accessor.get(i), Ball(di, di, di, di, fi, fi, fi, ui, ii, bi))
            << "Ball at index " + std::to_string(i) +
                   " contains incorrect data";
    }
}

TEST(aosoa_test, StructureOfArrays_construction4) {
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
    DummyDeviceMemoryOps dummy_memory_ops;
    Balls::ThisAccessor accessor;
    Balls balls(dummy_memory_ops, init, &accessor);

    for (size_t i = 0; i < init.size(); i++) {
        ASSERT_EQ(accessor.get(i), init[i])
            << "accessor.get(i) != init[i] at index " + std::to_string(i);
    }
}

TEST(aosoa_test, StructureOfArrays_construction5) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    const std::vector<Ball> init(n);
    Balls balls(memory_ops, init);

    for (size_t i = 0; i < balls.getAccess().size(); i++) {
        ASSERT_EQ(balls.getAccess().get(i), Ball{})
            << "Ball at index " + std::to_string(i) +
                   " should be equal to Ball{} but is not";
    }
}

TEST(aosoa_test, StructureOfArrays_construction6) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    const std::vector<Ball> init(n);
    const Balls balls(memory_ops, init);

    for (size_t i = 0; i < balls.getAccess().size(); i++) {
        ASSERT_EQ(balls.getAccess().get(i), Ball{})
            << "Ball at index " + std::to_string(i) +
                   " should be equal to Ball{} but is not";
    }
}

TEST(aosoa_test, StructureOfArrays_construction7) {
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
        ASSERT_EQ(balls.getAccess().get(i),
                  Ball(di, di, di, di, fi, fi, fi, ui, ii, bi))
            << "Ball at index " + std::to_string(i) +
                   " contains incorrect data";
    }
}

TEST(aosoa_test, StructureOfArrays_decreaseBy1) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n, &accessor);
    balls.decreaseBy(6);

    ASSERT_EQ(accessor.size(), 666)
        << "Unupdated accessor should have original size";
}

TEST(aosoa_test, StructureOfArrays_decreaseBy2) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n);
    balls.decreaseBy(6, &accessor);

    ASSERT_EQ(accessor.size(), 660)
        << "Updated accessor should have updated size";
}

TEST(aosoa_test, StructureOfArrays_decreaseBy3) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n);
    balls.decreaseBy(6);
    balls.updateAccessor(&accessor);

    ASSERT_EQ(accessor.size(), 660)
        << "Updated accessor should have updated size";
}

TEST(aosoa_test, StructureOfArrays_decreaseBy4) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n);
    balls.decreaseBy(3);
    balls.decreaseBy(3, &accessor);

    ASSERT_EQ(accessor.size(), 660)
        << "Updated accessor should have updated size";
}

TEST(aosoa_test, StructureOfArrays_decreaseBy5) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls balls(memory_ops, n);
    balls.decreaseBy(6);

    ASSERT_EQ(balls.getAccess().size(), 660)
        << "Updated accessor should have reduced size";
}

TEST(aosoa_test, StructureOfArrays_swap1) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
        Ball(0.0, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.swap<"position_x", "position_y">();
    ASSERT_EQ(accessor.get<"position_x">(0), 666.666)
        << "Unupdated swap should not be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap2) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(666.666, 0.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false),
        Ball(0.0, 13.0, 0.0, 0.0, 1.0f, 0.5f, 0.7f, 12u, -5, false)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.swap<"position_x", "position_y">(&accessor);
    ASSERT_EQ(accessor.get<"position_y">(0), 666.666)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_x">(1), 13.0)
        << "Updated swap should be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap3) {
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

    ASSERT_EQ(accessor.get<"position_y">(0), 666.666)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_x">(1), 13.0)
        << "Updated swap should be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap4) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);

    balls.swap<"position_x", "position_y", "position_z", "radius">(&accessor);

    ASSERT_EQ(accessor.get<"position_x">(0), 1.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(0), 0.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(0), 3.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"radius">(0), 2.0)
        << "Updated swap should be visible at accessor";

    ASSERT_EQ(accessor.get<"position_x">(1), 10.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(1), 9.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(1), 12.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"radius">(1), 11.0)
        << "Updated swap should be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap5) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);

    balls.swap<"position_x", "position_y", "position_y", "position_z">(
        &accessor);

    ASSERT_EQ(accessor.get<"position_x">(0), 1.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(0), 2.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(0), 0.0)
        << "Updated swap should be visible at accessor";

    ASSERT_EQ(accessor.get<"position_x">(1), 10.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(1), 11.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(1), 9.0)
        << "Updated swap should be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap6) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);

    balls.swap<"position_x", "position_y", "position_y", "position_z">();

    ASSERT_EQ(accessor.get<"position_x">(0), 0.0)
        << "Un updated swap should not be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(0), 1.0)
        << "Un updated swap should not be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(0), 2.0)
        << "Un updated swap should not be visible at accessor";

    ASSERT_EQ(accessor.get<"position_x">(1), 9.0)
        << "Un updated swap should not be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(1), 10.0)
        << "Un updated swap should not be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(1), 11.0)
        << "Un updated swap should not be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_swap7) {
    constexpr size_t alignment = 2048;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);

    balls.swap<"position_x", "position_y", "position_y", "position_z">();
    balls.updateAccessor(&accessor);

    ASSERT_EQ(accessor.get<"position_x">(0), 1.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(0), 2.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(0), 0.0)
        << "Updated swap should be visible at accessor";

    ASSERT_EQ(accessor.get<"position_x">(1), 10.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_y">(1), 11.0)
        << "Updated swap should be visible at accessor";
    ASSERT_EQ(accessor.get<"position_z">(1), 9.0)
        << "Updated swap should be visible at accessor";
}

TEST(aosoa_test, StructureOfArrays_updateAccessor1) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 666;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, n, &accessor);
    balls.decreaseBy(6);
    balls.updateAccessor(std::memcpy, &accessor);

    ASSERT_EQ(accessor.size(), 660)
        << "Updated accessor should have updated size";
}

TEST(aosoa_test, StructureOfArrays_getRows1) {
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

    ASSERT_EQ(rows.size(), init.size())
        << "Initial data and copied data sizes are not equal";

    for (size_t i = 0; i < rows.size(); i++) {
        ASSERT_EQ(rows[i], init[i])
            << "rows[i] != init[i] at index " + std::to_string(i);
    }
}

TEST(aosoa_test, StructureOfArrays_getRows2) {
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
    DummyDeviceMemoryOps dummy_memory_ops;
    Balls::ThisAccessor accessor;
    Balls balls(dummy_memory_ops, init, &accessor);
    const auto rows = balls.getRows();

    ASSERT_EQ(rows.size(), init.size())
        << "Initial data and copied data sizes are not equal";

    std::string msg = "";
    for (size_t i = 0; i < rows.size(); i++) {
        ASSERT_EQ(rows[i], init[i])
            << "rows[i] != init[i] at index " + std::to_string(i);
    }
}

TEST(aosoa_test, StructureOfArrays_memcpy_internal_to_internal1) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.memcpy<"position_y", "position_x">();

    ASSERT_EQ(accessor.get<"position_x">(0), 0.0) << "position_x[0] incorrect";
    ASSERT_EQ(accessor.get<"position_x">(1), 9.0) << "position_x[1] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(0), 0.0) << "position_y[0] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(1), 9.0) << "position_y[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_z", "radius", "color_r",
                               "color_g", "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memcpy_internal_to_internal2) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.memcpy<"position_y", "position_x">(std::memcpy);

    ASSERT_EQ(accessor.get<"position_x">(0), 0.0) << "position_x[0] incorrect";
    ASSERT_EQ(accessor.get<"position_x">(1), 9.0) << "position_x[1] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(0), 0.0) << "position_y[0] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(1), 9.0) << "position_y[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_z", "radius", "color_r",
                               "color_g", "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memcpy_internal_compile_fail_if_enabled) {
    constexpr size_t alignment = 128;
    typedef CBalls<alignment> Balls;

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, 1, &accessor);
    // Uncommenting the line below should give a compiler error
    // balls.memcpy<"position_x", "position_x">(std::memcpy);
}

TEST(aosoa_test, StructureOfArrays_memcpy_external_to_internal1) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    const std::vector<double> src{0.0, 9.0};
    balls.memcpy<"position_y">(src.data());

    ASSERT_EQ(accessor.get<"position_x">(0), 0.0) << "position_x[0] incorrect";
    ASSERT_EQ(accessor.get<"position_x">(1), 9.0) << "position_x[1] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(0), 0.0) << "position_y[0] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(1), 9.0) << "position_y[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_z", "radius", "color_r",
                               "color_g", "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memcpy_external_to_internal2) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    const std::vector<double> src{0.0, 9.0};
    balls.memcpy<"position_y">(src.data());

    ASSERT_EQ(accessor.get<"position_x">(0), 0.0) << "position_x[0] incorrect";
    ASSERT_EQ(accessor.get<"position_x">(1), 9.0) << "position_x[1] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(0), 0.0) << "position_y[0] incorrect";
    ASSERT_EQ(accessor.get<"position_y">(1), 9.0) << "position_y[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_z", "radius", "color_r",
                               "color_g", "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memcpy_internal_to_external1) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    std::vector<double> dst{666.0, 666.0};
    balls.memcpy<"position_y">(dst.data());

    ASSERT_EQ(dst[0], init[0].get<"position_y">()) << "dst[0] incorrect";
    ASSERT_EQ(dst[1], init[1].get<"position_y">()) << "dst[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memcpy_internal_to_external2) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    std::vector<double> dst{666.0, 666.0};
    balls.memcpy<"position_y">(std::memcpy, dst.data());

    ASSERT_EQ(dst[0], init[0].get<"position_y">()) << "dst[0] incorrect";
    ASSERT_EQ(dst[1], init[1].get<"position_y">()) << "dst[1] incorrect";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index", "index_distance",
                               "is_visible">(init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memset1) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.memset<"index">(0);
    ASSERT_EQ(accessor.get<"index">(0), 0) << "index[0] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(1), 0) << "index[1] set incorrectly";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index_distance", "is_visible">(
            init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memset2) {
    constexpr size_t alignment = 128;
    typedef Ball<alignment> Ball;
    typedef CBalls<alignment> Balls;

    std::vector<Ball> init{
        Ball(0.0, 1.0, 2.0, 3.0, 4.0f, 5.0f, 6.0f, 7u, -8, false),
        Ball(9.0, 10.0, 11.0, 12.0, 13.0f, 14.0f, 15.0f, 16u, -17, true)};

    Balls::ThisAccessor accessor;
    Balls balls(memory_ops, init, &accessor);
    balls.memset<"index">(std::memset, 0);
    ASSERT_EQ(accessor.get<"index">(0), 0) << "index[0] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(1), 0) << "index[1] set incorrectly";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index_distance", "is_visible">(
            init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memset3) {
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
    ASSERT_EQ(accessor.get<"index">(0), 0) << "index[0] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(1), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(2), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(3), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(4), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(5), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(6), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(7), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(8), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(9), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(10), 7u) << "index[1] set incorrectly";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index_distance", "is_visible">(
            init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, StructureOfArrays_memset4) {
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
    ASSERT_EQ(accessor.get<"index">(0), 0) << "index[0] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(1), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(2), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(3), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(4), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(5), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(6), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(7), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(8), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(9), 7u) << "index[1] set incorrectly";
    ASSERT_EQ(accessor.get<"index">(10), 7u) << "index[1] set incorrectly";

    auto [str, result] =
        assertUntouchedCorrect<alignment, "position_x", "position_y",
                               "position_z", "radius", "color_r", "color_g",
                               "color_b", "index_distance", "is_visible">(
            init, accessor);
    ASSERT_TRUE(result) << str;
}

TEST(aosoa_test, FailAllocation_length_error) {
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
        ASSERT_NE(pos, std::string::npos)
            << ("Exception should contain the substr \"" + substr + "\"").str;
    } catch (const std::exception &e) {
        ASSERT_FALSE(false) << std::string("Unhandled exception: ") + e.what();
    }
}

TEST(aosoa_test, FailAllocation_bad_alloc) {
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
        ASSERT_NE(pos, std::string::npos)
            << ("Exception should contain the substr \"" + substr + "\"").str;
    } catch (const std::exception &e) {
        ASSERT_FALSE(false) << std::string("Unhandled exception: ") + e.what();
    }
}
