#include "row.h"
#include "variable.h"
#include "json.hpp"

#include <cstdint>
#include <gtest/gtest.h>

using namespace aosoa;

typedef Row<Variable<float, "head">> RowSingle;
typedef Row<Variable<float, "head">, Variable<int32_t, "tail">> RowDouble;

TEST(row_test, ShouldFailCompilationIfEnabled) {
    // Row<Variable<float, "head">, Variable<int32_t, "head">>
    // fail_static_assert;
}

TEST(row_test, sizeof_RowSingle) {
    static_assert(sizeof(RowSingle) == sizeof(float));
}

TEST(row_test, sizeof_RowDouble) {
    static_assert(sizeof(RowDouble) == sizeof(float) + sizeof(int32_t));
}

TEST(row_test, RowSingle_construct1) {
    ASSERT_EQ(RowSingle().get<"head">(), 0.0f)
        << "Default value should be 0.0f";
}

TEST(row_test, RowSingle_construct2) {
    ASSERT_EQ(RowSingle(1.0f).get<"head">(), 1.0f) << "Value should be 1.0f";
}

TEST(row_test, RowSingle_construct3) {
    ASSERT_EQ(RowSingle(RowSingle(2.0f)).get<"head">(), 2.0f)
        << "Value should be 2.0f";
}

TEST(row_test, RowSingle_get_set1) {
    RowSingle row;
    row.get<"head">() = 10.0f;
    ASSERT_EQ(row.get<"head">(), 10.0f) << "Value should be 10.0f";
}

TEST(row_test, RowSingle_get_set2) {
    RowSingle row;
    row.set<"head">(10.0f);
    ASSERT_EQ(row.get<"head">(), 10.0f) << "Value should be 10.0f";
}

TEST(row_test, RowSingle_getconst) {
    const RowSingle row(666.666f);
    const float val = row.get<"head">();
    ASSERT_EQ(val, 666.666f) << "Value should be 666.666f";
}

TEST(row_test, RowSingle_equality1) {
    const RowSingle row(666.666f);
    ASSERT_EQ(row, RowSingle(666.666f)) << "Values should be equal";
}

TEST(row_test, RowSingle_equality2) {
    const RowSingle row{};
    ASSERT_EQ(row, RowSingle(0.0f)) << "Values should be equal";
}

TEST(row_test, RowSingle_equality3) {
    const RowSingle row{};
    ASSERT_EQ(row, row) << "Row should be equal to itself";
}

TEST(row_test, RowDouble_construct1) {
    const RowDouble row{};
    ASSERT_EQ(row.get<"head">(), 0.0f) << "Default value should be 0.0f";
    ASSERT_EQ(row.get<"tail">(), 0) << "Default value should be 0";
}

TEST(row_test, RowDouble_construct2) {
    const RowDouble row(1.0f, 2);
    ASSERT_EQ(row.get<"head">(), 1.0f) << "Value should be 1.0f";
    ASSERT_EQ(row.get<"tail">(), 2) << "Value should be 2";
}

TEST(row_test, RowDouble_construct3) {
    const RowDouble row(RowDouble(3.0f, 666));
    ASSERT_EQ(row.get<"head">(), 3.0f) << "Value should be 3.0f";
    ASSERT_EQ(row.get<"tail">(), 666) << "Value should be 666";
}

TEST(row_test, RowDouble_construct4) {
    const RowDouble row(4.0f, Row<Variable<int32_t, "tail">>(666));
    ASSERT_EQ(row.get<"head">(), 4.0f) << "Value should be 4.0f";
    ASSERT_EQ(row.get<"tail">(), 666) << "Value should be 666";
}

TEST(row_test, RowDouble_get_set1) {
    RowDouble row{};
    row.get<"head">() = 4.0f;
    row.get<"tail">() = 666;
    ASSERT_EQ(row.get<"head">(), 4.0f) << "Value should be 4.0f";
    ASSERT_EQ(row.get<"tail">(), 666) << "Value should be 666";
}

TEST(row_test, RowDouble_get_set2) {
    RowDouble row{};
    row.set<"head">(4.0f);
    row.set<"tail">(666);
    ASSERT_EQ(row.get<"head">(), 4.0f) << "Value should be 4.0f";
    ASSERT_EQ(row.get<"tail">(), 666) << "Value should be 666";
}

TEST(row_test, RowDouble_equality1) {
    const RowDouble row{};
    ASSERT_EQ(row, RowDouble()) << "Rows should be equal";
}

TEST(row_test, RowDouble_equality2) {
    const RowDouble row(1.0f, 16);
    ASSERT_EQ(row, RowDouble(1.0f, 16)) << "Rows should be equal";
}

TEST(row_test, RowDouble_equality3) {
    const RowDouble row(1.0f, 16);
    ASSERT_EQ(row, row) << "Row should be equal to itself";
}

TEST(row_test, Row_to_from_json) {
    using Row = const Row<Variable<float, "foo">, Variable<double, "bar">,
                          Variable<int, "baz">, Variable<bool, "foobar">>;

    const Row row(1.0f, 10.0, -666, true);
    const nlohmann::json j = row;
    const Row row2 = j;

    ASSERT_EQ(row, row2) << "Row should be equal to itself round tripped "
                            "through json conversion";
}
