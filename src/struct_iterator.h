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
template <size_t I, size_t N, typename T,
          template <typename U, size_t M> typename F, typename... Args>
void forEach(BoolAsType<true>, Args... args) {
    // Call once...
    F<T, I> f{};
    f(args...);

    // ... or more times, if I + 1 < N
    constexpr size_t J = I + 1;
    constexpr bool less = J < N;
    forEach<J, N, T, F>(BoolAsType<less>(), args...);
}

template <size_t I, size_t N, typename T,
          template <typename U, size_t M> typename F, typename... Args>
void forEach(BoolAsType<false>, Args... args) {}

template <size_t N, typename T, template <typename U, size_t M> typename F,
          typename... Args>
void forEachFunctor(Args... args) {
    constexpr bool less = 0 < N;
    forEach<0, N, T, F>(BoolAsType<less>(), args...);
}
} // namespace struct_iterator
