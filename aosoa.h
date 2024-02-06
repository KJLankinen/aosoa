/*
    Aosoa
    Copyright (C) 2024  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <utility>

#ifdef __NVCC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace aosoa {
// Hashing used to represent constexpr strings as uint32_t values
HOST DEVICE constexpr size_t operator""_idx(const char *str, size_t size) {
    constexpr uint32_t crc_table[256] = {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
        0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
        0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
        0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
        0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
        0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
        0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
        0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
        0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
        0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
        0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
        0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
        0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
        0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
        0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
        0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
        0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
        0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
        0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
        0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
        0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
        0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
        0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
        0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
        0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
        0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
        0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
        0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
        0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
        0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
        0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
        0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
        0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
        0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
        0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
        0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
        0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
        0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
        0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
        0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
        0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
        0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
        0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};

    uint32_t crc = 0xffffffff;
    for (size_t i = 0; i < size; i++)
        crc = (crc >> 8) ^ crc_table[(crc ^ (uint32_t)str[i]) & 0xff];
    return crc ^ 0xffffffff;
}

template <bool> struct BoolAsType {};

// This type is used as both the Array of Structures (Aos) type of values, as
// well as the Structure of Arrays (Soa) type of pointers to values for the
// AoSoa template type
template <typename... Types> struct Tuple;
template <> struct Tuple<> {
    bool operator==(const Tuple<> &) const { return true; }

  private:
    template <size_t I, size_t N>
    constexpr auto getter(BoolAsType<true>) = delete;
    template <size_t I, size_t N>
    constexpr auto getter(BoolAsType<false>) = delete;
    template <size_t I, size_t N, typename U>
    constexpr void setter(BoolAsType<true>, U u) = delete;
    template <size_t I, size_t N, typename U>
    constexpr void setter(BoolAsType<false>, U u) = delete;
    std::ostream &output(std::ostream &os) const { return os; }
};
template <typename T, typename... Types> struct Tuple<T, Types...> {
    // All instantiations are friends with each other
    template <typename... Ts> friend struct Tuple;

    T head = {};
    Tuple<Types...> tail = {};

    HOST DEVICE constexpr Tuple() {}
    HOST DEVICE constexpr Tuple(const Tuple<T, Types...> &tuple)
        : head(tuple.head), tail(tuple.tail) {}
    HOST DEVICE constexpr Tuple(T t, Tuple<Types...> tuple)
        : head(t), tail(tuple) {}
    template <typename... Args>
    HOST DEVICE constexpr Tuple(T t, Args... args) : head(t), tail(args...) {}

    template <size_t N> HOST DEVICE [[nodiscard]] constexpr auto get() const {
        constexpr bool LESS = 0 < N;
        return getter<0, N>(BoolAsType<LESS>{});
    }

    template <size_t N, typename U> HOST DEVICE constexpr void set(U u) {
        constexpr bool LESS = 0 < N;
        setter<0, N>(BoolAsType<LESS>{}, u);
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Tuple<Ts...> &);
    bool operator==(const Tuple<T, Types...> &rhs) const {
        return head == rhs.head && tail == rhs.tail;
    }

  private:
    // These getters/setters find the head from the correct depth of the tuple
    template <size_t I, size_t N>
    HOST DEVICE [[nodiscard]] constexpr auto getter(BoolAsType<true>) const {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        return tail.template getter<NEXT, N>(BoolAsType<LESS>{});
    }

    template <size_t I, size_t N>
    HOST DEVICE [[nodiscard]] constexpr auto getter(BoolAsType<false>) const {
        return head;
    }

    template <size_t I, size_t N, typename U>
    HOST DEVICE constexpr void setter(BoolAsType<true>, U u) {
        constexpr size_t NEXT = I + 1;
        constexpr bool LESS = NEXT < N;
        tail.template setter<NEXT, N>(BoolAsType<LESS>{}, u);
    }

    template <size_t I, size_t N, typename U>
    HOST DEVICE constexpr void setter(BoolAsType<false>, U u) {
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

template <typename... Types>
std::ostream &operator<<(std::ostream &os, const Tuple<Types...> &tuple) {
    os << "Tuple(";
    return tuple.output(os) << ")";
}

// Helper for matching a size_t I to a type T
template <size_t I, typename T> struct IndexTypePair {};
template <typename> struct PairTraits;
template <size_t I, typename T> struct PairTraits<IndexTypePair<I, T>> {
    static constexpr size_t i = I;
    using Type = T;
};

template <size_t MIN_ALIGN, typename... Pairs> struct AoSoa {
  private:
    // Very simple array impl: usable on devices
    template <typename T, size_t N> struct Array {
        T data[N] = {nullptr};
        HOST DEVICE constexpr T &operator[](size_t i) { return data[i]; }
        HOST DEVICE constexpr const T &operator[](size_t i) const {
            return data[i];
        }
    };
    // Type equality
    template <typename T, typename U> struct IsSame {
        constexpr static bool value = false;
    };
    template <typename T> struct IsSame<T, T> {
        constexpr static bool value = true;
    };

    static constexpr size_t NUM_POINTERS = sizeof...(Pairs);
    static constexpr size_t UIDS[NUM_POINTERS] = {PairTraits<Pairs>::i...};

    const size_t num_elements = 0;
    void *const data = nullptr;
    Array<void *, NUM_POINTERS> pointers;

  public:
    typedef Tuple<typename PairTraits<Pairs>::Type...> Aos;
    typedef Tuple<typename PairTraits<Pairs>::Type *...> Soa;

    HOST DEVICE constexpr AoSoa() {}
    AoSoa(size_t n, void *ptr)
        : num_elements(n), data(ptr),
          pointers(setPointers<0, typename PairTraits<Pairs>::Type...>(
              data, Array<void *, NUM_POINTERS>{}, ~0ul, num_elements)) {}

    [[nodiscard]] static size_t getMemReq(size_t num_elements) {
        // Get proper begin alignment: the strictest (largest) alignment
        // requirement between all the types and the MIN_ALIGN
        alignas(getAlignment()) uint8_t dummy = 0;
        constexpr size_t n = ~size_t(0);
        size_t space = n;
        [[maybe_unused]] const auto pointers =
            setPointers<0, typename PairTraits<Pairs>::Type...>(
                static_cast<void *>(&dummy), Array<void *, NUM_POINTERS>{},
                std::move(space), num_elements);

        const size_t num_bytes = n - space;
        // Require a block of (M + 1) * alignment bytes, where M is an integer.
        // The +1 is for manual alignment, if the memory allocation doesn't have
        // a strict enough alignment requirement.
        return num_bytes + bytesMissingFromAlignment(num_bytes) +
               getAlignment();
    }

    template <size_t UID1, size_t UID2> void swap() {
        constexpr size_t N1 =
            AoSoa::linearIndex<UID1, 0>(BoolAsType<UID1 == UIDS[0]>{});
        auto ptr1 = getPointer<0, N1, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == N1>{}, pointers[N1]);

        constexpr size_t N2 =
            AoSoa::linearIndex<UID2, 0>(BoolAsType<UID2 == UIDS[0]>{});
        auto ptr2 = getPointer<0, N2, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == N2>{}, pointers[N2]);

        if constexpr (IsSame<decltype(ptr1), decltype(ptr2)>::value) {
            std::swap(pointers[N1], pointers[N2]);
        } else {
            static_assert(IsSame<decltype(ptr1), decltype(ptr2)>::value,
                          "Pointers must have the same type to be swapped");
        }
    }

    // Return a pointer for UID
    template <size_t UID> HOST DEVICE [[nodiscard]] constexpr auto get() const {
        constexpr size_t N =
            AoSoa::linearIndex<UID, 0>(BoolAsType<UID == UIDS[0]>{});
        return getPointer<0, N, typename PairTraits<Pairs>::Type...>(
            BoolAsType<0 == N>{}, pointers[N]);
    }

    // Return an element at index I of pointer UID
    template <size_t UID, size_t I>
    HOST DEVICE [[nodiscard]] constexpr auto get() const {
        return get<UID>()[I];
    }

    // Return an element at index i of pointer UID
    template <size_t UID>
    HOST DEVICE [[nodiscard]] constexpr auto get(size_t i) const {
        return get<UID>()[i];
    }

    // Return a tuple
    template <typename T>
    HOST DEVICE [[nodiscard]] constexpr T get(size_t i) const {
        return toTuple<T, 0, NUM_POINTERS, typename PairTraits<Pairs>::Type...>(
            pointers.data, i);
    }

    // Set a single value
    template <size_t UID, typename T>
    HOST DEVICE constexpr void set(size_t i, T value) {
        get<UID>()[i] = value;
    }

    // Set by a tuple
    template <typename T> HOST DEVICE constexpr void set(size_t i, const T &t) {
        fromTuple<T, 0, NUM_POINTERS>(pointers.data, i, t);
    }

    template <size_t MA, typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const AoSoa<MA, Ts...> &);

  private:
    template <size_t>
    [[nodiscard]] static Array<void *, NUM_POINTERS>
    setPointers(void *ptr, Array<void *, NUM_POINTERS> pointers, size_t &&space,
                size_t) {
        // Align the end of last pointer to the getAlignment() byte boundary so
        // the memory requirement is a multiple of getAlignment()
        if (ptr) {
            ptr = std::align(getAlignment(), 1, ptr, space);
        }
        return pointers;
    }

    template <size_t I, typename T, typename... Types>
    [[nodiscard]] static Array<void *, NUM_POINTERS>
    setPointers(void *ptr, Array<void *, NUM_POINTERS> pointers, size_t &&space,
                size_t num_elements) {
        constexpr size_t size_of_type = sizeof(T);
        if (ptr) {
            ptr = std::align(getAlignment(), size_of_type, ptr, space);
            if (ptr) {
                pointers[I] = ptr;
                ptr = static_cast<void *>(static_cast<T *>(ptr) + num_elements);
                space -= size_of_type * num_elements;

                return setPointers<I + 1, Types...>(
                    ptr, pointers, std::move(space), num_elements);
            }

            return Array<void *, NUM_POINTERS>{};
        }

        return Array<void *, NUM_POINTERS>{};
    }

    template <size_t, size_t, typename T, typename...>
    HOST DEVICE [[nodiscard]] constexpr static auto
    getPointer(BoolAsType<true>, void *const ptr) {
        return static_cast<T *>(ptr);
    }

    template <size_t I, size_t N, typename T, typename... Types>
    HOST DEVICE [[nodiscard]] constexpr static auto
    getPointer(BoolAsType<false>, void *const ptr) {
        constexpr size_t NEXT = I + 1;
        return getPointer<NEXT, N, Types...>(BoolAsType<NEXT == N>{}, ptr);
    }

    template <typename TupleType, size_t, size_t>
    HOST DEVICE [[nodiscard]] constexpr static auto toTuple(void *const[],
                                                            size_t) {
        return Tuple<>{};
    }
    template <typename TupleType, size_t I, size_t N, typename T,
              typename... Types>
    HOST DEVICE [[nodiscard]] constexpr static auto
    toTuple(void *const pointers[], size_t i) {
        T *head = static_cast<T *>(pointers[I]) + i;
        auto tail = toTuple<TupleType, I + 1, N, Types...>(pointers, i);

        if constexpr (IsSame<TupleType, Aos>::value) {
            return Tuple<T, Types...>(*head, tail);
        } else {
            return Tuple<T *, Types *...>(head, tail);
        }
    }

    template <typename TupleType, size_t, size_t>
    HOST DEVICE constexpr static void fromTuple(void *const[], size_t,
                                                const Tuple<> &) {}

    template <typename TupleType, size_t I, size_t N, typename T,
              typename... Types>
    HOST DEVICE constexpr static void
    fromTuple(void *const pointers[], size_t i,
              const Tuple<T, Types...> &tuple) {
        if constexpr (IsSame<TupleType, Aos>::value) {
            static_cast<T *>(pointers[I])[i] = tuple.head;
        } else {
            static_cast<T *>(pointers[I])[i] = *tuple.head;
        }

        fromTuple<TupleType, I + 1, N>(pointers, i, tuple.tail);
    }

    [[nodiscard]] constexpr static size_t bytesMissingFromAlignment(size_t n) {
        return (getAlignment() - bytesOverAlignment(n)) & (getAlignment() - 1);
    }

    [[nodiscard]] constexpr static size_t bytesOverAlignment(size_t n) {
        return n & (getAlignment() - 1);
    }

    template <typename T, typename... Ts> constexpr static size_t maxAlign() {
        if constexpr (sizeof...(Ts) == 0) {
            return alignof(T);
        } else {
            return std::max(alignof(T), maxAlign<Ts...>());
        }
    }

    [[nodiscard]] constexpr static size_t getAlignment() {
        static_assert((MIN_ALIGN & (MIN_ALIGN - 1)) == 0,
                      "MIN_ALIGN isn't a power of two");
        // Aligned by the strictest (largest) alignment requirement between all
        // the types and the MIN_ALIGN template argument
        // N.B. A statement with multiple alignas declarations is supposed to
        // pick the strictest one, but GCC for some reason picks the last one
        // that is applied...
        // If it weren't for that bug, could use:
        // struct alignas(MIN_ALIGN) alignas(typename
        // PairTraits<Pairs>::Type...) Aligned {}; return alignof(Aligned);
        struct alignas(MIN_ALIGN) MinAligned {};
        return maxAlign<MinAligned, typename PairTraits<Pairs>::Type...>();
    }

    // Find the N for which UIDS[N] == UID
    template <size_t UID, size_t N>
    HOST DEVICE [[nodiscard]] constexpr static size_t
    linearIndex(BoolAsType<false>) {
        constexpr size_t NEXT = N + 1;
        constexpr bool EQ = UID == UIDS[NEXT];
        return AoSoa::linearIndex<UID, NEXT>(BoolAsType<EQ>{});
    }

    template <size_t, size_t N>
    HOST DEVICE [[nodiscard]] constexpr static size_t
    linearIndex(BoolAsType<true>) {
        return N;
    }

    // Fail compilation if two types have equal unique IDs either because they
    // have the same name or they hash to the same value
    template <size_t I, size_t J> struct ClashingIds {
        constexpr static bool value = UIDS[I] != UIDS[J];
    };

    template <size_t I, size_t J, size_t N>
    constexpr static bool assertUniqueIds() {
        if constexpr (I < N) {
            if constexpr (J < N) {
                static_assert(
                    ClashingIds<I, J>::value,
                    "Two types with indices I and J have the same unique ID. "
                    "This means they either have the same name, or they have "
                    "unique names but the names hash to the same number. In "
                    "any case, you should change the name of one of them.");

                assertUniqueIds<I, J + 1, N>();
            } else {
                assertUniqueIds<I + 1, I + 2, N>();
            }
        }

        return true;
    }

    static_assert(assertUniqueIds<0, 1, NUM_POINTERS>());
};

template <size_t A, typename... Pairs>
std::ostream &operator<<(std::ostream &os, const AoSoa<A, Pairs...> &aosoa) {
    typedef AoSoa<A, Pairs...> AoSoa;
    constexpr size_t n = AoSoa::NUM_POINTERS;

    os << "AoSoa {\n  ";
    os << "num members: " << n << "\n  ";
    os << "num elements: " << aosoa.num_elements << "\n  ";
    os << "memory requirement (bytes): " << AoSoa::getMemReq(aosoa.num_elements)
       << "\n  ";
    os << "*data: " << aosoa.data << "\n  ";
    os << "*pointers[]: " << '{';
    for (size_t i = 0; i < n - 1; i++) {
        os << aosoa.pointers[i] << ", ";
    }
    os << aosoa.pointers[n - 1] << "}\n  ";
    os << "member uids: " << '{';
    for (size_t i = 0; i < n - 1; i++) {
        os << AoSoa::UIDS[i] << ", ";
    }
    os << AoSoa::UIDS[n - 1] << "}\n";
    os << "}\n";

    return os;
}
} // namespace aosoa
