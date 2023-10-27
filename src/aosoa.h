#include <cstddef>
#include <cstdint>

namespace aosoa {
// This must be specialized for each type T
// Specifically, each T should define the 'Type' member for each N
template <size_t N, typename T> struct MemberTypeGetter;

template <size_t N, typename T> struct PointerToMember {
    // Compare to
    // typedef float A::* B;
    // Now B's type is a pointer to a float member of struct A
    typedef typename MemberTypeGetter<N, T>::Type MemberType;
    typedef MemberType T::*Type;
};

// This returns an actual pointer to N'th member of T
template <size_t N, typename T>
typename PointerToMember<N, T>::Type pointerToMember(void);

// BoolAsType<true> is a distinct type from BoolAsType<false> and can be
// used to specialize functions
template <bool> struct BoolAsType {};

// Call the function/functor of type F with arguments Args while I < N
template <size_t I, size_t N, typename F, typename... Args>
void forEach(BoolAsType<true>, F &f, Args... args) {
    // Call once...
    f.template operator()<I>(args...);

    // ... or more times, if I + 1 < N
    constexpr size_t J = I + 1;
    constexpr bool less = J < N;
    forEach<J, N>(BoolAsType<less>(), f, args...);
}

template <size_t I, size_t N, typename F, typename... Args>
void forEach(BoolAsType<false>, F &f, Args... args) {}

template <size_t I, size_t N, typename F, typename... Args>
void forEachFunctor(F &f, Args... args) {
    constexpr bool less = I < N;
    forEach<I, N>(BoolAsType<less>(), f, args...);
}

template <size_t N, typename T>
constexpr typename MemberTypeGetter<N, T>::Type get(const T *const t) {
    return t->*pointerToMember<N, T>();
}

// Implementation for generic tuple-like struct for holding stuff
template <typename... Ts> class Tuple;
template <> class Tuple<> {};
template <typename T, typename... Ts> class Tuple<T, Ts...> {
    T value;
    Tuple<Ts...> next;

  public:
    template <size_t N> auto get() {
        constexpr bool LESS = 0 < N;
        return getter<0, N>(BoolAsType<LESS>{});
    }

    template <size_t N> void set(auto u) {
        constexpr bool LESS = 0 < N;
        setter<0, N>(BoolAsType<LESS>{}, u);
    }

    template <size_t I, size_t N> auto getter(BoolAsType<true>) {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        return next.template getter<NEXT, N>(BoolAsType<LESS>{});
    }

    template <size_t I, size_t N> auto getter(BoolAsType<false>) {
        return value;
    }

    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<true>, U u) {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        next.template setter<NEXT, N>(BoolAsType<LESS>{}, u);
    }

    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<false>, U u) {
        value = u;
    }
};
} // namespace aosoa
