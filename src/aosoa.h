#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

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
void forEach(BoolAsType<false>, F &, Args...) {}

template <size_t I, size_t N, typename F, typename... Args>
void forEachFunctor(F &f, Args... args) {
    constexpr bool less = I < N;
    forEach<I, N>(BoolAsType<less>(), f, args...);
}

template <size_t N, typename T>
constexpr typename MemberTypeGetter<N, T>::Type get(const T *const t) {
    return t->*pointerToMember<N, T>();
}

constexpr size_t hash(const char *str, size_t size, size_t n = 0,
                      size_t h = 2166136261) {
    return n == size ? h
                     : hash(str, size, n + 1,
                            (h * 16777619) ^ static_cast<size_t>(str[n]));
}

size_t constexpr operator""_idx(const char *str, size_t size) {
    return hash(str, size);
}

// Implementation for generic tuple-like struct for holding stuff
template <size_t I, typename T> struct IndexTypePair {};
template <typename> struct PairTraits;
template <size_t I, typename T> struct PairTraits<IndexTypePair<I, T>> {
    static constexpr size_t i = I;
    using Type = T;
};

template <typename... Types> class Tuple;
template <> class Tuple<> {
    template <size_t I, size_t N> auto getter(BoolAsType<true>) = delete;
    template <size_t I, size_t N> auto getter(BoolAsType<false>) = delete;
    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<true>, U u) = delete;
    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<false>, U u) = delete;
};
template <typename T, typename... Types> class Tuple<T, Types...> {
    // All instantiations are friends with each other
    template <typename... Us> friend class Tuple;

    T value;
    Tuple<Types...> next;

  public:
    template <typename... Args>
    constexpr Tuple(T t, Args... args) : value(t), next(args...) {}

    template <size_t N> auto get() const {
        constexpr bool LESS = 0 < N;
        return getter<0, N>(BoolAsType<LESS>{});
    }

    template <size_t N> void set(auto u) {
        constexpr bool LESS = 0 < N;
        setter<0, N>(BoolAsType<LESS>{}, u);
    }

  private:
    // These getters/setters find the value from the correct depth of the tuple
    template <size_t I, size_t N> auto getter(BoolAsType<true>) const {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        return next.template getter<NEXT, N>(BoolAsType<LESS>{});
    }

    template <size_t I, size_t N> auto getter(BoolAsType<false>) const {
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

template <typename... Pairs> class Aosoa {
    static constexpr size_t indices[sizeof...(Pairs)] = {
        PairTraits<Pairs>::i...};
    Tuple<typename PairTraits<Pairs>::Type...> aos;
    Tuple<typename PairTraits<Pairs>::Type *...> soa;

  public:
    template <typename... Args>
    constexpr Aosoa(Args... args) : aos(args...), soa(&args...) {}

    template <size_t I> [[nodiscard]] auto get() const {
        constexpr bool EQ = I == indices[0];
        constexpr size_t N = Aosoa::linearIndex<I, 0>(BoolAsType<EQ>{});
        return aos.template get<N>();
    }

    template <size_t I, typename T> void set(T t) {
        constexpr bool EQ = I == indices[0];
        constexpr size_t N = Aosoa::linearIndex<I, 0>(BoolAsType<EQ>{});
        aos.template set<N>(t);
    }

  private:
    // Find the N for which indices[N] == I
    template <size_t I, size_t N>
    static constexpr size_t linearIndex(BoolAsType<false>) {
        constexpr size_t NEXT = N + 1;
        constexpr bool EQ = I == indices[NEXT];
        return Aosoa::linearIndex<I, NEXT>(BoolAsType<EQ>{});
    }

    template <size_t I, size_t N>
    static constexpr size_t linearIndex(BoolAsType<true>) {
        return N;
    }
};

// StructureOfArrays
template <size_t I>
[[nodiscard]] void *setPointers(void *ptr, void **, size_t &, size_t) {
    return ptr;
}

// Pointers array will contain memory addresses aligned properly for each type T
// such that num_elements consecutive values of type T fit between pointers[I]
// and pointers[I + 1], where typeof(I) == T
template <size_t I, typename T, typename... Types>
[[nodiscard]] void *setPointers(void *ptr, void **pointers, size_t &space,
                                size_t num_elements) {
    // Align input ptr for this type
    if (ptr) {
        ptr = std::align(alignof(T), sizeof(T), ptr, space);
        if (ptr) {
            pointers[I] = ptr;
            T *t = static_cast<T *>(ptr) + num_elements;
            ptr = static_cast<void *>(t);
            space -= sizeof(T) * num_elements;

            return setPointers<I + 1, Types...>(ptr, pointers, space,
                                                num_elements);
        }

        return nullptr;
    }

    return nullptr;
}

template <size_t I, size_t N, typename T, typename... Types>
auto getPointer(BoolAsType<true>, void *ptr) {
    return static_cast<T *>(ptr);
}

template <size_t I, size_t N, typename T, typename... Types>
auto getPointer(BoolAsType<false>, void *ptr) {
    constexpr size_t NEXT = I + 1;
    return getPointer<NEXT, N, Types...>(BoolAsType<NEXT == N>{}, ptr);
}

template <typename... Pairs> class StructureOfArrays {
    static constexpr size_t NUM_MEMBERS = sizeof...(Pairs);
    static constexpr size_t UIDS[NUM_MEMBERS] = {PairTraits<Pairs>::i...};

    void *data = nullptr;
    void *pointers[NUM_MEMBERS] = {nullptr};
    const size_t num_elements = 0;

  public:
    constexpr StructureOfArrays() {}
    constexpr StructureOfArrays(size_t n) : num_elements(n) {}

    [[nodiscard]] bool init(void *ptr) {
        data = ptr;
        size_t space = getMemReq(num_elements);
        return setPointers<0, typename PairTraits<Pairs>::Type...>(
                   data, pointers, space, num_elements) != nullptr &&
               space == 0;
    }

    [[nodiscard]] static size_t getMemReq(size_t num_elements) {
        // Used for a proper alignment of any scalar type
        max_align_t dummy;
        void *begin = static_cast<void *>(&dummy);

        void *pointers[NUM_MEMBERS] = {nullptr};
        size_t space = ~size_t(0);

        void *end = setPointers<0, typename PairTraits<Pairs>::Type...>(
            begin, pointers, space, num_elements);

        const size_t num_bytes = static_cast<size_t>(
            static_cast<char *>(end) - static_cast<char *>(begin));

        return num_bytes;
    }

    template <size_t UID> auto getPtr() const {
        constexpr size_t N = StructureOfArrays::linearIndex<UID, 0>(
            BoolAsType<UID == UIDS[0]>{});
        return getPointer<0, N, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == N>{}, pointers[N]);
    }

    template <size_t UID> auto getValue(size_t i) const {
        return getPtr<UID>()[i];
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &,
                                    const StructureOfArrays<Ts...> &);

  private:
    // Find the N for which UIDS[N] == UID
    template <size_t UID, size_t N>
    static constexpr size_t linearIndex(BoolAsType<false>) {
        constexpr size_t NEXT = N + 1;
        constexpr bool EQ = UID == UIDS[NEXT];
        return StructureOfArrays::linearIndex<UID, NEXT>(BoolAsType<EQ>{});
    }

    template <size_t UID, size_t N>
    static constexpr size_t linearIndex(BoolAsType<true>) {
        return N;
    }
};

template <typename... Pairs>
std::ostream &operator<<(std::ostream &os,
                         const StructureOfArrays<Pairs...> &soa) {
    const size_t n = StructureOfArrays<Pairs...>::NUM_MEMBERS;

    os << "StructureOfArrays {\n  ";
    os << "num members: " << n << "\n  ";
    os << "num elements: " << soa.num_elements << "\n  ";
    os << "memory requirement (bytes): "
       << StructureOfArrays<Pairs...>::getMemReq(soa.num_elements) << "\n  ";
    os << "*data: " << soa.data << "\n  ";
    os << "*pointers[]: " << '{';
    for (size_t i = 0; i < n - 1; i++) {
        os << soa.pointers[i] << ", ";
    }
    os << soa.pointers[n - 1] << "}\n  ";
    os << "member uids: " << '{';
    for (size_t i = 0; i < n - 1; i++) {
        os << StructureOfArrays<Pairs...>::UIDS[i] << ", ";
    }
    os << StructureOfArrays<Pairs...>::UIDS[n - 1] << "}\n";
    os << "}\n";

    return os;
}
} // namespace aosoa
