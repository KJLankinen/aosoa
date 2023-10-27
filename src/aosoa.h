#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

namespace aosoa {
// BoolAsType<true> is a distinct type from BoolAsType<false> and can be
// used to specialize functions
template <bool> struct BoolAsType {};

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
template <typename... Types> struct Tuple;
template <> struct Tuple<> {
  private:
    template <size_t I, size_t N> auto getter(BoolAsType<true>) = delete;
    template <size_t I, size_t N> auto getter(BoolAsType<false>) = delete;
    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<true>, U u) = delete;
    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<false>, U u) = delete;
    std::ostream &output(std::ostream &os) const { return os; }
};
template <typename T, typename... Types> struct Tuple<T, Types...> {
    // All instantiations are friends with each other
    template <typename... Ts> friend struct Tuple;

    T head;
    Tuple<Types...> tail;

    constexpr Tuple() {}
    constexpr Tuple(const Tuple<T, Types...> &tuple)
        : head(tuple.head), tail(tuple.tail) {}
    constexpr Tuple(T t, Tuple<Types...> tuple) : head(t), tail(tuple) {}
    template <typename... Args>
    constexpr Tuple(T t, Args... args) : head(t), tail(args...) {}

    template <size_t N> [[nodiscard]] auto get() const {
        constexpr bool LESS = 0 < N;
        return getter<0, N>(BoolAsType<LESS>{});
    }

    template <size_t N> void set(auto u) {
        constexpr bool LESS = 0 < N;
        setter<0, N>(BoolAsType<LESS>{}, u);
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Tuple<Ts...> &);

  private:
    // These getters/setters find the head from the correct depth of the tuple
    template <size_t I, size_t N>
    [[nodiscard]] auto getter(BoolAsType<true>) const {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        return tail.template getter<NEXT, N>(BoolAsType<LESS>{});
    }

    template <size_t I, size_t N>
    [[nodiscard]] auto getter(BoolAsType<false>) const {
        return head;
    }

    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<true>, U u) {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        tail.template setter<NEXT, N>(BoolAsType<LESS>{}, u);
    }

    template <size_t I, size_t N, typename U>
    void setter(BoolAsType<false>, U u) {
        head = u;
    }

    std::ostream &output(std::ostream &os) const {
        if constexpr (sizeof...(Types) == 0) {
            os << head;
        } else if constexpr (sizeof...(Types) > 0) {
            os << head << ", ";
            return tail.output(os);
        }

        return os;
    }
};

template <size_t I, typename T> struct IndexTypePair {};
template <typename> struct PairTraits;
template <size_t I, typename T> struct PairTraits<IndexTypePair<I, T>> {
    static constexpr size_t i = I;
    using Type = T;
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
[[nodiscard]] auto getPointer(BoolAsType<true>, void *ptr) {
    return static_cast<T *>(ptr);
}

template <size_t I, size_t N, typename T, typename... Types>
[[nodiscard]] auto getPointer(BoolAsType<false>, void *ptr) {
    constexpr size_t NEXT = I + 1;
    return getPointer<NEXT, N, Types...>(BoolAsType<NEXT == N>{}, ptr);
}

template <size_t I, size_t N> [[nodiscard]] auto toAos(void *const[], size_t) {
    return Tuple<>{};
}

template <size_t I, size_t N, typename T, typename... Types>
[[nodiscard]] auto toAos(void *const pointers[], size_t i) {
    const T *const t = static_cast<T *>(pointers[I]);
    constexpr size_t NEXT = I + 1;
    return Tuple<T, Types...>(t[i], toAos<NEXT, N, Types...>(pointers, i));
}

template <size_t I, size_t N> [[nodiscard]] auto toSoa(void *const[], size_t) {
    return Tuple<>{};
}

template <size_t I, size_t N, typename T, typename... Types>
[[nodiscard]] auto toSoa(void *const pointers[], size_t i) {
    T *const t = static_cast<T *>(pointers[I]);
    constexpr size_t NEXT = I + 1;
    return Tuple<T *, Types *...>(&t[i], toSoa<NEXT, N, Types...>(pointers, i));
}

template <size_t I, size_t N> void fromAos(void *[], size_t, const Tuple<> &) {}

template <size_t I, size_t N, typename T, typename... Types>
void fromAos(void *pointers[], size_t i, const Tuple<T, Types...> &tuple) {
    T *const t = static_cast<T *>(pointers[I]);
    t[i] = tuple.head;
    constexpr size_t NEXT = I + 1;
    fromAos<NEXT, N>(pointers, i, tuple.tail);
}

template <typename... Pairs> struct StructureOfArrays {
  private:
    static constexpr size_t NUM_MEMBERS = sizeof...(Pairs);
    static constexpr size_t UIDS[NUM_MEMBERS] = {PairTraits<Pairs>::i...};

    void *data = nullptr;
    void *pointers[NUM_MEMBERS] = {nullptr};
    const size_t num_elements = 0;

  public:
    typedef Tuple<typename PairTraits<Pairs>::Type...> Aos;
    typedef Tuple<typename PairTraits<Pairs>::Type *...> Soa;

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

    // Return a pointer for UID
    template <size_t UID> [[nodiscard]] auto get() const {
        constexpr size_t N = StructureOfArrays::linearIndex<UID, 0>(
            BoolAsType<UID == UIDS[0]>{});
        return getPointer<0, N, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == N>{}, pointers[N]);
    }

    // Return an element at index i of pointer UID
    template <size_t UID> [[nodiscard]] auto get(size_t i) const {
        return get<UID>()[i];
    }

    // Return a tuple
    [[nodiscard]] Aos get(size_t i) const {
        return toAos<0, NUM_MEMBERS, typename PairTraits<Pairs>::Type...>(
            pointers, i);
    }

    // Return a tuple of pointers
    [[nodiscard]] Soa getSoa(size_t i) const {
        return toSoa<0, NUM_MEMBERS, typename PairTraits<Pairs>::Type...>(
            pointers, i);
    }

    // Set a single value
    template <size_t UID> void set(size_t i, auto value) {
        get<UID>()[i] = value;
    }

    // Set by a tuple
    void set(size_t i, const Aos &aos) {
        fromAos<0, NUM_MEMBERS>(pointers, i, aos);
    }

    template <size_t I> [[nodiscard]] auto &operator[](size_t i) {
        return getPointer<0, I, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == I>{}, pointers[I])[i];
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &,
                                    const StructureOfArrays<Ts...> &);

  private:
    // Find the N for which UIDS[N] == UID
    template <size_t UID, size_t N>
    [[nodiscard]] static constexpr size_t linearIndex(BoolAsType<false>) {
        constexpr size_t NEXT = N + 1;
        constexpr bool EQ = UID == UIDS[NEXT];
        return StructureOfArrays::linearIndex<UID, NEXT>(BoolAsType<EQ>{});
    }

    template <size_t UID, size_t N>
    [[nodiscard]] static constexpr size_t linearIndex(BoolAsType<true>) {
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

template <typename... Types>
std::ostream &operator<<(std::ostream &os, const Tuple<Types...> &tuple) {
    os << "Tuple(";
    tuple.output(os);
    os << ")\n";

    return os;
}
} // namespace aosoa
