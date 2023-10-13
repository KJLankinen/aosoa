#include "struct_iterator.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>

struct Example {
    float a = 0.0f;
    int32_t b = 1;
    uint32_t c = 2;

    template <size_t I> constexpr auto get() const;

    constexpr static size_t index(const std::string_view name) {
        if (name == "a") {
            return 0;
        } else if (name == "b") {
            return 1;
        } else if (name == "c") {
            return 2;
        } else {
            // static_assert(false, "No such member on example");
        }

        return 0;
    }
};

namespace struct_iterator {
template <> struct MemberTypeGetter<0, Example> {
    typedef float Type;
};
template <> struct MemberTypeGetter<1, Example> {
    typedef int32_t Type;
};
template <> struct MemberTypeGetter<2, Example> {
    typedef uint32_t Type;
};
template <>
PointerToMember<0, Example>::Type pointerToMember<0, Example>(void) {
    return &Example::a;
}
template <>
PointerToMember<1, Example>::Type pointerToMember<1, Example>(void) {
    return &Example::b;
}
template <>
PointerToMember<2, Example>::Type pointerToMember<2, Example>(void) {
    return &Example::c;
}
} // namespace struct_iterator

template <size_t I> constexpr auto Example::get() const {
    return struct_iterator::get<I>(this);
}

struct Displayer {
    std::string str;

    template <size_t N, typename T> void operator()(const T *const t) {
        typedef
            typename struct_iterator::MemberTypeGetter<N, T>::Type MemberType;
        MemberType value = t->*struct_iterator::pointerToMember<N, T>();
        str.append(std::to_string(value));
        str.append(std::string(", "));
    }
};

struct Counter {
    size_t count = 0;

    template <size_t N> void operator()() { count += 1; }
};

void count() {
    Counter counter;
    struct_iterator::forEachFunctor<0, 10, Counter>(counter);
    printf("%d\n", counter.count);
    assert(counter.count == 10);
}

void display_values() {
    const Example example{};
    Displayer displayer;
    struct_iterator::forEachFunctor<0, 3, Displayer>(displayer, &example);
    std::cout << displayer.str << std::endl;
    float a = example.get<Example::index("a")>();
    std::cout << a << std::endl;
}
