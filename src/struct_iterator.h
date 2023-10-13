#include <cstddef>
#include <cstdint>

namespace struct_iterator {
// This must be specialized for each type T
// Specifically, each T should define the 'Type' member for each N
template <typename T, size_t N> struct MemberTypeGetter;

template <typename T, size_t N> struct PointerToMember {
    // Compare to
    // typedef float A::* B;
    // Now B's type is a pointer to a float member of struct A
    typedef typename MemberTypeGetter<T, N>::Type MemberType;
    typedef MemberType T::*Type;
};

// This returns an actual pointer to N'th member of T
template <typename T, size_t N>
typename PointerToMember<T, N>::Type pointerToMember(void);

// BoolAsType<true> is a distinct type from BoolAsType<false> and can be
// used to specialize functions
template <bool> struct BoolAsType {};

// Call the function/functor of type F with arguments Args while I < N
template <size_t I, size_t N, typename T, typename F, typename... Args>
void forEach(BoolAsType<true>, F &f, Args... args) {
    // Call once...
    f.template operator()<I, T>(args...);

    // ... or more times, if I + 1 < N
    constexpr size_t J = I + 1;
    constexpr bool less = J < N;
    forEach<J, N, T>(BoolAsType<less>(), f, args...);
}

template <size_t I, size_t N, typename T, typename F, typename... Args>
void forEach(BoolAsType<false>, F &f, Args... args) {}

template <size_t I, size_t N, typename T, typename F, typename... Args>
void forEachFunctor(F &f, Args... args) {
    constexpr bool less = I < N;
    forEach<I, N, T>(BoolAsType<less>(), f, args...);
}
} // namespace struct_iterator
