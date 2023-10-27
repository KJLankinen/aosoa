#include "aosoa.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>

using aosoa::operator""_idx;

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

    constexpr static size_t index(size_t idx) {
        if (idx == "length"_idx) {
            return 0;
        } else if (idx == "b"_idx) {
            return 1;
        } else if (idx == "c"_idx) {
            return 2;
        }

        return ~static_cast<size_t>(0);
    }
};

namespace aosoa {
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
    return &Example::length;
}
template <>
PointerToMember<1, Example>::Type pointerToMember<1, Example>(void) {
    return &Example::b;
}
template <>
PointerToMember<2, Example>::Type pointerToMember<2, Example>(void) {
    return &Example::c;
}
} // namespace aosoa

template <size_t I> constexpr auto Example::get() const {
    return aosoa::get<Example::index(I)>(this);
}

struct Displayer {
    std::string str;

    template <size_t N, typename T> void operator()(const T *const t) {
        typedef typename aosoa::MemberTypeGetter<N, T>::Type MemberType;
        MemberType value = t->*aosoa::pointerToMember<N, T>();
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
    aosoa::forEachFunctor<0, 10>(counter);
    printf("%ld\n", counter.count);
    assert(counter.count == 10);
}

void display_values() {
    const Example example{};
    Displayer displayer;
    aosoa::forEachFunctor<0, 3>(displayer, &example);
    std::cout << displayer.str << std::endl;
    float a = example.get<"length"_idx>();
    int32_t b = example.get<"b"_idx>();
    uint32_t c = example.get<"c"_idx>();
    std::cout << a << b << c << std::endl;
}

void aosoatest() {
    aosoa::Aosoa<aosoa::IndexTypePair<"is_visible"_idx, bool>,
                 aosoa::IndexTypePair<"radius"_idx, float>,
                 aosoa::IndexTypePair<"num_hits"_idx, int>>
        bfi(true, 2.5f, 1);

    std::cout << bfi.get<"is_visible"_idx>() << std::endl;
    std::cout << bfi.get<"radius"_idx>() << std::endl;
    std::cout << bfi.get<"num_hits"_idx>() << std::endl;

    aosoa::Aosoa<aosoa::IndexTypePair<"is_not_visible"_idx, bool>,
                 aosoa::IndexTypePair<"radius"_idx, float>,
                 aosoa::IndexTypePair<"num_hits"_idx, int>>
        bfi2(false, 2.5f, 1);

    bfi2.set<"radius"_idx>(100.0f);

    std::cout << bfi2.get<"is_not_visible"_idx>() << std::endl;
    std::cout << bfi2.get<"radius"_idx>() << std::endl;
    std::cout << bfi2.get<"num_hits"_idx>() << std::endl;
}

void soa() {
    typedef aosoa::StructureOfArrays<
        aosoa::IndexTypePair<"is_visible"_idx, bool>,
        aosoa::IndexTypePair<"radius"_idx, float>,
        aosoa::IndexTypePair<"radius2"_idx, double>,
        aosoa::IndexTypePair<"num_hits"_idx, int>>
        Soa;

    const size_t n = 1321357;
    const size_t mem_req = Soa::getMemReq(n);
    std::cout << "mem req: " << mem_req << std::endl;

    Soa soa(n);
    void *memory = std::malloc(mem_req);
    bool success = soa.init(memory);
    std::cout << "success : " << success << std::endl;

    Soa soa2;
    std::memcpy(static_cast<void *>(&soa2), static_cast<void *>(&soa),
                sizeof(Soa));
    std::cout << soa << soa2 << std::endl;

    float value = soa.template getValue<"radius"_idx>(10);
    std::cout << value << std::endl;

    std::free(memory);
}

void test() {
    display_values();
    count();
    aosoatest();
    soa();
}
