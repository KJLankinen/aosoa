#include "struct_iterator.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>

constexpr size_t hash(const char *str, size_t size, size_t n = 0,
                      size_t h = 2166136261) {
    return n == size ? h : hash(str, size, n + 1, (h * 16777619) ^ (str[n]));
}

size_t constexpr operator""_m(const char *str, size_t size) {
    return hash(str, size);
}

struct Example {
    float length = 0.0f;
    int32_t b = 1;
    uint32_t c = 2;

    template <size_t I> constexpr auto get() const;

    constexpr static size_t index(const std::string_view name) {
        if (name == "length") {
            return 0;
        } else if (name == "b") {
            return 1;
        } else if (name == "c") {
            return 2;
        }

        return ~static_cast<size_t>(0);
    }

    constexpr static size_t index(size_t hash) {
        if (hash == "length"_m) {
            return 0;
        } else if (hash == "b"_m) {
            return 1;
        } else if (hash == "c"_m) {
            return 2;
        }

        return ~static_cast<size_t>(0);
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
    return struct_iterator::get<Example::index(I)>(this);
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
    float a = example.get<"length"_m>();
    uint32_t b = example.get<"b"_m>();
    int32_t c = example.get<"c"_m>();
    std::cout << a << b << c << std::endl;
}
