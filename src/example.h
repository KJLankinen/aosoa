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

struct Displayer {
    std::string str;

    template <size_t N, typename T> void operator()(const T *const t) {
        typedef
            typename struct_iterator::MemberTypeGetter<T, N>::Type MemberType;
        MemberType value = t->*struct_iterator::pointerToMember<T, N>();
        str.append(std::to_string(value));
        str.append(std::string(", "));
    }
};

struct Counter {
    size_t count = 0;

    template <size_t N, typename T> void operator()() { count += 1; }
};

void count() {
    Counter counter;
    struct_iterator::forEachFunctor<0, 10, size_t, Counter>(counter);
    printf("%d\n", counter.count);
    assert(counter.count == 10);
}

void display_values() {
    const Example example{};
    Displayer displayer;
    struct_iterator::forEachFunctor<0, 3, Example, Displayer>(displayer,
                                                              &example);
    std::cout << displayer.str << std::endl;
}
