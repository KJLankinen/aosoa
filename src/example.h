#include "struct_iterator.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>

struct Example {
    float a = 0.0f;
    int32_t b = 1;
    uint32_t c = 2;
};

namespace struct_iterator {
template <> struct MemberTypeGetter<Example, 0> {
    typedef float Type;
};
template <> struct MemberTypeGetter<Example, 1> {
    typedef int32_t Type;
};
template <> struct MemberTypeGetter<Example, 2> {
    typedef uint32_t Type;
};
template <>
PointerToMember<Example, 0>::Type pointerToMember<Example, 0>(void) {
    return &Example::a;
}
template <>
PointerToMember<Example, 1>::Type pointerToMember<Example, 1>(void) {
    return &Example::b;
}
template <>
PointerToMember<Example, 2>::Type pointerToMember<Example, 2>(void) {
    return &Example::c;
}
} // namespace struct_iterator

template <typename T, size_t N> struct Displayer {
    void operator()(const T *const t, std::string *display_str) {
        typedef
            typename struct_iterator::MemberTypeGetter<T, N>::Type MemberType;
        MemberType value = t->*struct_iterator::pointerToMember<T, N>();
        display_str->append(std::to_string(value));
        display_str->append(std::string(", "));
    }
};

template <typename T, size_t N> struct Counter {
    void operator()(T *count) { *count += N; }
};

void count() {
    size_t count = 0;
    struct_iterator::forEachFunctor<10, size_t, Counter>(&count);
    printf("%d\n", count);
    assert(count == 45);
}

void display_values() {
    const Example example{};
    std::string display_str;
    struct_iterator::forEachFunctor<3, Example, Displayer>(&example,
                                                           &display_str);
    std::cout << display_str << std::endl;
}
