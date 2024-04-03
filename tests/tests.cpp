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

#include "type_operations.h"
#include "variable.h"
#include <gtest/gtest.h>

using namespace aosoa;

TEST(type_test, NthType) {
    static_assert(std::is_same<typename NthType<0, double, float, int>::Type,
                               double>::value,
                  "0th type should be double");
    static_assert(std::is_same<typename NthType<1, double, float, int>::Type,
                               float>::value,
                  "1st type should be float");
    static_assert(
        std::is_same<typename NthType<2, double, float, int>::Type, int>::value,
        "2nd type should be int");
}

TEST(type_test, FindFromStrings) {
    static_assert(
        0ul == Find<"foo"_cts>::template FromStrings<"foo"_cts, "bar"_cts,
                                                     "baz"_cts>::index,
        "Index of foo should be 0");
    static_assert(
        1ul == Find<"bar"_cts>::template FromStrings<"foo"_cts, "bar"_cts,
                                                     "baz"_cts>::index,
        "Index of bar should be 1");
    static_assert(
        2ul == Find<"baz"_cts>::template FromStrings<"foo"_cts, "bar"_cts,
                                                     "baz"_cts>::index,
        "Index of baz should be 2");
    static_assert(
        ~0ul == Find<"nope"_cts>::template FromStrings<"foo"_cts, "bar"_cts,
                                                       "baz"_cts>::index,
        "Index of nope should be ~0ul");
}

TEST(type_test, FindFromVariables) {
    static_assert(0ul ==
                      Find<"foo"_cts>::template FromVariables<
                          Variable<float, "foo"_cts>, Variable<int, "bar"_cts>,
                          Variable<bool, "baz"_cts>>::index,
                  "Index of foo should be 0");
    static_assert(1ul ==
                      Find<"bar"_cts>::template FromVariables<
                          Variable<float, "foo"_cts>, Variable<int, "bar"_cts>,
                          Variable<bool, "baz"_cts>>::index,
                  "Index of foo should be 0");
    static_assert(2ul ==
                      Find<"baz"_cts>::template FromVariables<
                          Variable<float, "foo"_cts>, Variable<int, "bar"_cts>,
                          Variable<bool, "baz"_cts>>::index,
                  "Index of foo should be 0");
    static_assert(~0ul ==
                      Find<"nope"_cts>::template FromVariables<
                          Variable<float, "foo"_cts>, Variable<int, "bar"_cts>,
                          Variable<bool, "baz"_cts>>::index,
                  "Index of foo should be 0");
}

TEST(type_test, IsStringContainedIn) {
    static_assert(Is<"foo"_cts>::template ContainedIn<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                  "foo should be found");
    static_assert(Is<"bar"_cts>::template ContainedIn<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                  "bar should be found");
    static_assert(Is<"baz"_cts>::template ContainedIn<"foo"_cts, "bar"_cts,
                                                      "baz"_cts>::value,
                  "baz should be found");
    static_assert(
        !Is<"not_found"_cts>::template ContainedIn<"foo"_cts, "bar"_cts,
                                                   "baz"_cts>::value,
        "not_found should not be found");
}
