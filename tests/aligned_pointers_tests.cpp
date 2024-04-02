#include "balls.h"
#include <gtest/gtest.h>

template <size_t Alignment>
using Pointers = AlignedPointers<Alignment, Variable<float, "head">,
                                 Variable<double, "tail">>;

template <size_t A>
std::tuple<std::string, bool> assertAligned(Pointers<A> &pointers) {
    bool result = true;
    size_t i = 0;
    constexpr size_t max_size_t = ~0ul;
    size_t space = max_size_t;
    for (; i < pointers.size && result; i++) {
        void *ptr = pointers[i];
        std::align(Pointers<A>::getAlignment(), 1, ptr, space);
        result = space == max_size_t && ptr != nullptr;
    }
    return std::make_tuple(
        std::string("Incorrect alignment for ") + std::to_string(i), result);
}

TEST(aligned_pointers_test, AlignedPointers_construct1) {
    constexpr size_t alignment = 256;
    using Pointers = Pointers<alignment>;
    static_assert(Pointers::size == 2,
                  "Expecting Pointers type to have two variables");

    const Pointers pointers = {};
    ASSERT_EQ(pointers[0], nullptr) << "Default constructed pointers array "
                                       "should contain only nullptrs";
    ASSERT_EQ(pointers[1], nullptr) << "Default constructed pointers array "
                                       "should contain only nullptrs";
}

TEST(aligned_pointers_test, AlignedPointers_construct2) {
    constexpr size_t alignment = 256;
    using Pointers = Pointers<alignment>;
    static_assert(Pointers::getAlignment() == alignment,
                  "Alignment should be 256");

    constexpr size_t n = 100;
    const size_t bytes = Pointers::getMemReq(n);
    std::vector<uint8_t> data(bytes);

    Pointers pointers(n, data.data());

    auto [str, result] = assertAligned(pointers);
    ASSERT_TRUE(result) << str;
}

TEST(aligned_pointers_test, AlignedPointers_construct3) {
    constexpr size_t alignment = 256;
    using Pointers = Pointers<alignment>;
    static_assert(Pointers::getAlignment() == alignment,
                  "Alignment should be 256");

    constexpr size_t n = 100;
    const size_t bytes = Pointers::getMemReq(n);
    std::vector<uint8_t> data(bytes);

    Pointers pointers(n, data.data());
    for (size_t i = 0; i < pointers.size; i++) {
        ASSERT_TRUE(pointers[i] != nullptr)
            << "Pointer " + std::to_string(i) + " is nullptr";
    }
}

TEST(aligned_pointers_test, AlignedPointers_alignment) {
    static_assert(Pointers<256>::getAlignment() == 256, "Incorrect alignment");
    static_assert(Pointers<128>::getAlignment() == 128, "Incorrect alignment");
    static_assert(Pointers<64>::getAlignment() == 64, "Incorrect alignment");
    static_assert(Pointers<32>::getAlignment() == 32, "Incorrect alignment");
    static_assert(Pointers<16>::getAlignment() == 16, "Incorrect alignment");
    static_assert(Pointers<8>::getAlignment() == 8, "Incorrect alignment");
    static_assert(Pointers<4>::getAlignment() == 8, "Incorrect alignment");
    static_assert(Pointers<2>::getAlignment() == 8, "Incorrect alignment");
    static_assert(Pointers<1>::getAlignment() == 8, "Incorrect alignment");
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq1) {
    constexpr size_t alignment = 1;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * 8) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq2) {
    constexpr size_t alignment = 2;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * 8) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq3) {
    constexpr size_t alignment = 4;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * 8) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq4) {
    constexpr size_t alignment = 8;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * alignment) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq5) {
    constexpr size_t alignment = 16;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * alignment) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq6) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 3 * alignment) << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq7) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1024;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, n * (sizeof(double) + sizeof(float)) + alignment)
        << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq8) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1000;
    const size_t mem_req = Pointers<alignment>::getMemReq(n);
    ASSERT_EQ(mem_req, 8064 + 4096 + alignment)
        << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq9) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1000;
    typedef AlignedPointers<alignment, Variable<double, "first">,
                            Variable<float, "second">, Variable<int, "third">,
                            Variable<bool, "fourth">, Variable<float, "fifth">>
        Pointers;
    const size_t mem_req = Pointers::getMemReq(n);
    ASSERT_EQ(mem_req, 8064 + 4096 + 4096 + 1024 + 4096 + alignment)
        << "Memory requirement mismatch";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq10) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 3216547;
    typedef AlignedPointers<alignment, Variable<double, "first">,
                            Variable<char, "second">, Variable<int, "third">,
                            Variable<bool, "fourth">, Variable<float, "fifth">>
        Pointers;
    const size_t mem_req = Pointers::getMemReq(n);
    ASSERT_EQ((mem_req & (alignment - 1)), 0)
        << "Total memory requirement must be a multiple of alignment";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq11) {
    constexpr size_t alignment = 32;
    constexpr size_t n = 3216547;
    typedef AlignedPointers<alignment, Variable<double, "first">,
                            Variable<char, "second">, Variable<int, "third">,
                            Variable<bool, "fourth">, Variable<float, "fifth">>
        Pointers;
    const size_t mem_req = Pointers::getMemReq(n);
    ASSERT_EQ((mem_req & (alignment - 1)), 0)
        << "Total memory requirement must be a multiple of alignment";
}

TEST(aligned_pointers_test, AlignedPointers_getMemReq_BigType) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 100;

    typedef AlignedPointers<
        alignment, Variable<double, "1">, Variable<double, "2">,
        Variable<double, "3">, Variable<double, "4">, Variable<double, "5">,
        Variable<double, "6">, Variable<double, "7">, Variable<double, "8">,
        Variable<double, "9">, Variable<double, "10">, Variable<double, "11">,
        Variable<float, "12">, Variable<float, "13">, Variable<float, "14">,
        Variable<float, "15">, Variable<float, "16">, Variable<float, "17">,
        Variable<float, "18">, Variable<float, "19">, Variable<float, "20">,
        Variable<float, "21">, Variable<float, "22">, Variable<float, "23">,
        Variable<float, "24">, Variable<int, "25">, Variable<int, "26">,
        Variable<int, "27">, Variable<int, "28">, Variable<int, "29">,
        Variable<int, "30">, Variable<int, "31">, Variable<int, "32">,
        Variable<int, "33">, Variable<int, "34">, Variable<int, "35">,
        Variable<int, "36">, Variable<int, "37">, Variable<int, "38">,
        Variable<int, "39">, Variable<int, "40">, Variable<int, "41">,
        Variable<int, "42">, Variable<int, "43">, Variable<int, "44">,
        Variable<int, "45">, Variable<int, "46">, Variable<int, "47">,
        Variable<int, "48">, Variable<int, "49">, Variable<int, "50">,
        Variable<bool, "51">, Variable<bool, "52">, Variable<bool, "53">,
        Variable<bool, "54">, Variable<bool, "55">, Variable<bool, "56">,
        Variable<bool, "57">, Variable<bool, "58">, Variable<bool, "59">,
        Variable<bool, "60">, Variable<bool, "61">, Variable<bool, "62">,
        Variable<bool, "63">, Variable<char, "64">, Variable<char, "65">,
        Variable<char, "66">, Variable<char, "67">, Variable<char, "68">,
        Variable<char, "69">, Variable<char, "70">, Variable<char, "71">,
        Variable<char, "72">, Variable<char, "73">>
        Pointers;

    const size_t mem_req = Pointers::getMemReq(n);
    ASSERT_EQ((mem_req & (alignment - 1)), 0)
        << "Total memory requirement must be a multiple of alignment";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_construction1) {
    constexpr size_t alignment = 128;
    using Accessor = BallAccessor<alignment>;
    const Accessor accessor = {};

    ASSERT_EQ(accessor.get<"position_x">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"position_y">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"position_z">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"radius">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"color_r">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"color_g">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"color_b">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"index">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"index_distance">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
    ASSERT_EQ(accessor.get<"is_visible">(), nullptr)
        << "Default constructed accessor should have only nullptrs";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get1) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    Accessor accessor(n, data.data());
    accessor.set<"position_x">(0, 666.666);

    ASSERT_EQ(accessor.get<"position_x">()[0], 666.666)
        << "Values from get should be equal 1";
    ASSERT_EQ(accessor.get<"position_x">()[0], accessor.get<"position_x">(0))
        << "Values from get should be equal 2";
    ASSERT_EQ(accessor.get<"position_x">()[0],
              (accessor.get<"position_x", 0>()))
        << "Values from get should be equal 3";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get2) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    const Accessor accessor(n, data.data());

    ASSERT_EQ(accessor.get<"position_x">(), accessor.get<0>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"position_y">(), accessor.get<1>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"position_z">(), accessor.get<2>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"radius">(), accessor.get<3>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"color_r">(), accessor.get<4>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"color_g">(), accessor.get<5>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"color_b">(), accessor.get<6>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"index">(), accessor.get<7>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"index_distance">(), accessor.get<8>())
        << "Pointers from get should be equal";
    ASSERT_EQ(accessor.get<"is_visible">(), accessor.get<9>())
        << "Pointers from get should be equal";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get3) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    Accessor accessor(n, data.data());

    const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f, 8u, 9,
                                true);
    accessor.set(666, row);
    ASSERT_EQ(accessor.get(666), row) << "Row should be equal to set row";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get4) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    Accessor accessor(n, data.data());

    const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f, 8u, 9,
                                true);
    accessor.set(666, row);
    ASSERT_EQ(accessor.get<0>()[666], row.get<"position_x">())
        << "Values should be equal";
    ASSERT_EQ(accessor.get<"position_x">()[666], row.get<"position_x">())
        << "Values should be equal";
    ASSERT_EQ(accessor.get<"position_x">(666), row.get<"position_x">())
        << "Values should be equal";
    ASSERT_EQ((accessor.get<"position_x", 666>()), row.get<"position_x">())
        << "Values should be equal";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get5) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    Accessor accessor(n, data.data());

    const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f, 8u, 9,
                                true);
    accessor.set(666, row);
    accessor.set<"radius">(666, 666.0);
    ASSERT_EQ(accessor.get<"radius">(666), 666.0) << "Radius should be changed";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_get6) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(Accessor::getMemReq(n));
    Accessor accessor(n, data.data());

    const Accessor::FullRow row(1.0, 2.0, 3.0, 4.0, 5.0f, 6.0f, 7.0f, 8u, 9,
                                true);
    accessor.set(666, row);
    accessor.get<"radius">()[666] = 666.0;
    ASSERT_EQ(accessor.get<"radius">(666), 666.0) << "Radius should be changed";
}

TEST(aligned_pointers_test, AlignedPointers_Accessor_size) {
    constexpr size_t alignment = 128;
    constexpr size_t n = 1189;
    using Accessor = BallAccessor<alignment>;
    std::vector<uint8_t> data(CBalls<alignment>::getMemReq(n));
    const Accessor accessor(n, data.data());

    ASSERT_EQ(accessor.size(), n) << "Accessor size should be equal to n";
}
