/*
    aosoa
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
#include <type_traits>
#include <utility>

#ifdef __NVCC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace {
// Use 'true' and 'false' as separate types to overload a function for each case
template <bool> struct BoolAsType {};

// Array usable on devices
template <typename T, size_t N> struct Array {
    T items[N] = {};

    HOST DEVICE constexpr T &operator[](size_t i) { return items[i]; }
    HOST DEVICE constexpr const T &operator[](size_t i) const {
        return items[i];
    }
    HOST DEVICE constexpr size_t size() const { return N; }
    HOST DEVICE constexpr T *data() const { return &items; }
};

// Type equality
template <typename T, typename U> struct IsSame {
    constexpr static bool value = false;
};

template <typename T> struct IsSame<T, T> {
    constexpr static bool value = true;
};

// Compile time string
template <size_t N> struct CompileTimeString {
    char str[N + 1] = {};

    consteval CompileTimeString(const char (&s)[N + 1]) {
        std::copy_n(s, N + 1, str);
    }

    consteval bool operator==(const CompileTimeString<N> rhs) const {
        return std::equal(rhs.str, rhs.str + N, str);
    }

    template <size_t M>
    consteval bool operator==(const CompileTimeString<M>) const {
        return false;
    }

    template <size_t M>
    consteval CompileTimeString<N + M>
    operator+(const CompileTimeString rhs) const {
        char out_str[N + 1 + M] = {};
        std::copy_n(str, N, out_str);
        std::copy_n(rhs.str, M + 1, out_str + N);
        return CompileTimeString<N + M>(out_str);
    }

    consteval char operator[](size_t i) const { return str[i]; }
    consteval char *data() const { return str; }
    consteval size_t size() const { return N - 1; }
};

template <size_t N, size_t M>
consteval bool operator==(const char (&lhs)[N], CompileTimeString<M> rhs) {
    return CompileTimeString<N>(lhs) == rhs;
}

template <size_t N, size_t M>
consteval bool operator==(CompileTimeString<N> lhs, const char (&rhs)[M]) {
    return lhs == CompileTimeString<N>(rhs);
}

template <size_t N, size_t M>
consteval auto operator+(const char (&lhs)[N + 1], CompileTimeString<M> rhs) {
    return CompileTimeString<N>(lhs) + rhs;
}

template <size_t N, size_t M>
consteval auto operator+(CompileTimeString<N> lhs, const char (&rhs)[M + 1]) {
    return lhs + CompileTimeString<M>(rhs);
}

// Deduction guide
template <size_t N>
CompileTimeString(const char (&)[N]) -> CompileTimeString<N - 1>;

template <CompileTimeString CTS> constexpr auto operator""_cts() { return CTS; }
} // namespace

namespace aosoa {
// Helper for matching a compile time string to a type T
template <typename, CompileTimeString> struct Variable {};
} // namespace aosoa

namespace {
// Used to extract the type and the name from a Variable containing both
template <typename> struct PairTraits;
template <typename T, CompileTimeString CTS>
struct PairTraits<aosoa::Variable<T, CTS>> {
    using Type = T;
    static constexpr CompileTimeString name = CTS;
};

// Get the Nth type from a parameter pack of types
template <size_t N, typename... Types> struct NthType {
  private:
    template <size_t I, typename Head, typename... Tail>
    consteval static auto ofType() {
        if constexpr (I == N) {
            return Head{};
        } else {
            return ofType<I + 1, Tail...>();
        }
    }

    static constexpr auto t = ofType<0, Types...>();

  public:
    using Type = std::remove_const_t<decltype(t)>;
};

// Find the index of string from a parameter pack
template <CompileTimeString MatchStr, CompileTimeString... Strings>
struct IndexOfString {
  private:
    template <size_t N> consteval static size_t index() { return N; }

    template <size_t N, CompileTimeString Head, CompileTimeString... Tail>
    consteval static size_t index() {
        if constexpr (MatchStr == Head) {
            return N;
        } else {
            return index<N + 1, Tail...>();
        }
    }

  public:
    constexpr static size_t i = index<0, Strings...>();
};
} // namespace

namespace aosoa {
/*
 * Row represents a row from a structure of arrays layout
 * Given a structure of arrays with N arrays, the aggregation of the ith value
 * of each array represent the ith row of the structure of arrays and can be
 * instantiated as a variable of type Row.
 *
 * Row is implemented as something between a tuple and a normal struct.
 * It's a tuple in the sense that it's a recursive type, but it's a struct in
 * the sense that you can (only) access it's members by name, using a
 * templated syntax like auto foo = row.get<"foo">();
 * */
template <typename... Vars> struct Row;

// Specialize for a single type
template <typename Var> struct Row<Var> {
    template <typename... Ts> friend struct Row;

    using Type = PairTraits<Var>::Type;
    static constexpr auto name = PairTraits<Var>::name;

  private:
    Type head = {};

  public:
    HOST DEVICE constexpr Row() {}
    HOST DEVICE constexpr Row(const Row<Var> &row) : head(row.head) {}
    HOST DEVICE constexpr Row(Type t) : head(t) {}

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] constexpr const auto &get() const {
        static_assert(EqualStrings<name, CTS>::value,
                      "No member with such name");
        return head;
    }

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] constexpr auto &get() {
        static_assert(EqualStrings<name, CTS>::value,
                      "No member with such name");
        return head;
    }

    template <CompileTimeString CTS, typename U>
    HOST DEVICE constexpr void set(U u) {
        get<CTS>() = u;
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Row<Ts...> &);

    bool operator==(const Row<Var> &rhs) const { return head == rhs.head; }

  private:
    std::ostream &output(std::ostream &os) const {
        os << PairTraits<Var>::name.str << ": " << head;
        return os;
    }

    template <CompileTimeString MemberName, CompileTimeString Candidate>
    struct EqualStrings {
        constexpr static bool value = MemberName == Candidate;
    };
};

// Specialize for two or more types
template <typename Var1, typename Var2, typename... Vars>
struct Row<Var1, Var2, Vars...> {
    template <typename... Ts> friend struct Row;

  private:
    using Type = PairTraits<Var1>::Type;
    static constexpr auto name = PairTraits<Var1>::name;

    using ThisType = Row<Var1, Var2, Vars...>;
    using TailType = Row<Var2, Vars...>;

    Type head = {};
    TailType tail = {};

  public:
    HOST DEVICE constexpr Row() {}
    HOST DEVICE constexpr Row(const ThisType &row)
        : head(row.head), tail(row.tail) {}
    HOST DEVICE constexpr Row(Type t, const TailType &row)
        : head(t), tail(row) {}
    template <typename... Args>
    HOST DEVICE constexpr Row(Type t, Args... args) : head(t), tail(args...) {}

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] constexpr const auto &get() const {
        if constexpr (CTS == name) {
            return head;
        } else {
            return tail.template get<CTS>();
        }
    }

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] constexpr auto &get() {
        if constexpr (CTS == name) {
            return head;
        } else {
            return tail.template get<CTS>();
        }
    }

    template <CompileTimeString CTS, typename U>
    HOST DEVICE constexpr void set(U u) {
        get<CTS>() = u;
    }

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Row<Ts...> &);
    bool operator==(const ThisType &rhs) const {
        return head == rhs.head && tail == rhs.tail;
    }

  private:
    std::ostream &output(std::ostream &os) const {
        os << PairTraits<Var1>::name.str << ": " << head << "\n  ";
        return tail.output(os);
    }

    // ==== Uniqueness of names ====
    // Asserting at compile time that all the names in the template parameters
    // are unique.
    //
    // IndexOfString takes a string to match and a parameter pack of strings and
    // finds the index of the match string from the  parameter pack. Given that
    // the parameter pack does not contain the match string, the returned index
    // should be the size of the parameter pack. Thus, if the index of string
    // is the size of the parameter pack, the first string is unique.
    template <size_t IndexOfStr, size_t SizeOfParamPack> struct EqualIndices {
        constexpr static bool value = IndexOfStr == SizeOfParamPack;
    };

    // +1 so the clashing index takes Var1 into account, i.e. i == 1 points to
    // Var2, not to the first of Vars...
    constexpr static size_t index_of_var1 =
        IndexOfString<PairTraits<Var1>::name, PairTraits<Var2>::name,
                      PairTraits<Vars>::name...>::i +
        1;

    // +1 because Var2 is not a part of Vars... but is a part of the string
    // parameter pack given to IndexOfString.
    // Another +1 so the clashing index takes Var1 into account, i.e. i == 1
    // points to Var2, not to the first of Vars...
    constexpr static size_t num_strings = sizeof...(Vars) + 2;

    // The tail structures do this to their own strings, i.e. Var2 is compared
    // to Vars... and so on recursively
    static_assert(EqualIndices<index_of_var1, num_strings>::value,
                  "Found a clashing name at the index I of EqualIndices<I, J>");

  public:
    // This helps StructureOfArrays assert it's names are unique by asserting
    // that the resulting Row type has unique names
    constexpr static bool unique_names =
        EqualIndices<index_of_var1, num_strings>::value;
};

template <typename... Vars>
std::ostream &operator<<(std::ostream &os, const Row<Vars...> &row) {
    os << "Row {\n  ";
    return row.output(os) << "\n}";
}

/*
 * StructureOfArrays helps store and access data in a structure of arrays layout
 *
 * Instead of storing N instantiations of a structure with M members in a single
 * array, it's more cache and GPU friendly to store the values of the struct in
 * M separate arrays with N values in each.
 *
 * However, it's sometimes convenient or necessary to access/inspect all (or
 * some) members of one particular instantiation of the struct at one go. The
 * Row type defined above gives access to the values in this way.
 *
 * One can access values stored in the StructureOfArrays either by getting a
 * full row, by getting an array of some particular members or by getting a
 * single value from an array.
 * */

template <size_t MIN_ALIGN, typename... Variables> struct StructureOfArrays {
    using FullRow = aosoa::Row<Variables...>;

  private:
    static_assert(FullRow::unique_names,
                  "StructureOfArrays struct has clashing names");

    const size_t num_elements = 0;
    void *const data = nullptr;
    static constexpr size_t num_pointers = sizeof...(Variables);

    using Pointers = Array<void *, num_pointers>;
    Pointers pointers;

  public:
    HOST DEVICE constexpr StructureOfArrays() {}
    StructureOfArrays(size_t n, void *ptr)
        : num_elements(n), data(ptr),
          pointers(
              makeAlignedPointers<0, typename PairTraits<Variables>::Type...>(
                  data, Pointers{}, ~0ul, num_elements)) {}

    [[nodiscard]] static size_t getMemReq(size_t num_elements) {
        // Get proper begin alignment: the strictest (largest) alignment
        // requirement between all the types and the MIN_ALIGN
        alignas(getAlignment()) uint8_t dummy = 0;
        constexpr size_t n = ~size_t(0);
        size_t space = n;
        [[maybe_unused]] const auto pointers =
            makeAlignedPointers<0, typename PairTraits<Variables>::Type...>(
                static_cast<void *>(&dummy), Pointers{}, std::move(space),
                num_elements);

        const size_t num_bytes = n - space;
        // Require a block of (M + 1) * alignment bytes, where M is an integer.
        // The +1 is for manual alignment, if the memory allocation doesn't have
        // a strict enough alignment requirement.
        return num_bytes + bytesMissingFromAlignment(num_bytes) +
               getAlignment();
    }

    template <CompileTimeString CTS1, CompileTimeString CTS2> void swap() {
        constexpr size_t i =
            IndexOfString<CTS1, PairTraits<Variables>::name...>::i;
        constexpr size_t j =
            IndexOfString<CTS2, PairTraits<Variables>::name...>::i;
        using TypeI =
            typename NthType<i, typename PairTraits<Variables>::Type...>::Type;
        using TypeJ =
            typename NthType<j, typename PairTraits<Variables>::Type...>::Type;

        static_assert(IsSame<TypeI, TypeJ>::value, "Mismatched types for swap");

        std::swap(pointers[i], pointers[j]);
    }

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] auto get() const {
        constexpr size_t i =
            IndexOfString<CTS, PairTraits<Variables>::name...>::i;
        using Type =
            typename NthType<i, typename PairTraits<Variables>::Type...>::Type;

        return static_cast<Type *>(pointers[i]);
    }

    template <CompileTimeString CTS, size_t I>
    HOST DEVICE [[nodiscard]] auto get() const {
        return get<CTS>()[I];
    }

    template <CompileTimeString CTS>
    HOST DEVICE [[nodiscard]] auto get(size_t i) const {
        return get<CTS>()[i];
    }

    HOST DEVICE [[nodiscard]] FullRow get(size_t i) const {
        return toRow<Variables...>(i);
    }

    template <CompileTimeString CTS, typename T>
    HOST DEVICE void set(size_t i, T value) const {
        get<CTS>()[i] = value;
    }

    HOST DEVICE void set(size_t i, const FullRow &t) const {
        fromRow<Variables...>(i, t);
    }

    HOST DEVICE void set(size_t i, FullRow &&t) const {
        fromRow<Variables...>(i, std::move(t));
    }

    template <size_t MA, typename... Ts>
    friend std::ostream &operator<<(std::ostream &,
                                    const StructureOfArrays<MA, Ts...> &);

    HOST DEVICE size_t size() const { return num_elements; }

  private:
    [[nodiscard]] constexpr static size_t bytesMissingFromAlignment(size_t n) {
        return (getAlignment() - bytesOverAlignment(n)) & (getAlignment() - 1);
    }

    [[nodiscard]] constexpr static size_t bytesOverAlignment(size_t n) {
        return n & (getAlignment() - 1);
    }

    template <typename T, typename... Ts> consteval static size_t maxAlign() {
        if constexpr (sizeof...(Ts) == 0) {
            return alignof(T);
        } else {
            return std::max(alignof(T), maxAlign<Ts...>());
        }
    }

    [[nodiscard]] consteval static size_t getAlignment() {
        static_assert((MIN_ALIGN & (MIN_ALIGN - 1)) == 0,
                      "MIN_ALIGN isn't a power of two");
        // Aligned by the strictest (largest) alignment requirement between all
        // the types and the MIN_ALIGN template argument
        // N.B. A statement with multiple alignas declarations is supposed to
        // pick the strictest one, but GCC for some reason picks the last one
        // that is applied...
        // If it weren't for that bug, could use:
        // struct alignas(MIN_ALIGN) alignas(typename
        // PairTraits<Variables>::Type...) Aligned {}; return alignof(Aligned);
        struct alignas(MIN_ALIGN) MinAligned {};
        return maxAlign<MinAligned, typename PairTraits<Variables>::Type...>();
    }

    template <size_t>
    [[nodiscard]] static Pointers
    makeAlignedPointers(void *ptr, Pointers pointers, size_t &&space, size_t) {
        // Align the end of last pointer to the getAlignment() byte boundary so
        // the memory requirement is a multiple of getAlignment()
        if (ptr) {
            ptr = std::align(getAlignment(), 1, ptr, space);
        }
        return pointers;
    }

    template <size_t I, typename Head, typename... Tail>
    [[nodiscard]] static Pointers
    makeAlignedPointers(void *ptr, Pointers pointers, size_t &&space,
                        size_t num_elements) {
        constexpr size_t size_of_type = sizeof(Head);
        if (ptr) {
            ptr = std::align(getAlignment(), size_of_type, ptr, space);
            if (ptr) {
                pointers[I] = ptr;
                ptr = static_cast<void *>(static_cast<Head *>(ptr) +
                                          num_elements);
                space -= size_of_type * num_elements;

                return makeAlignedPointers<I + 1, Tail...>(
                    ptr, pointers, std::move(space), num_elements);
            }
        }

        return Pointers{};
    }

    template <typename Head, typename... Tail>
    [[nodiscard]] HOST DEVICE auto toRow(size_t i) const {
        using H = PairTraits<Head>;
        if constexpr (sizeof...(Tail) > 0) {
            return Row<Head, Tail...>(get<H::name>(i), toRow<Tail...>(i));
        } else {
            return Row<Head>(get<H::name>(i));
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, const FullRow &row) const {
        using H = PairTraits<Head>;
        set<H::name, typename H::Type>(i, row.template get<H::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, row);
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, FullRow &&row) const {
        using H = PairTraits<Head>;
        set<H::name, typename H::Type>(i, row.template get<H::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, std::move(row));
        }
    }
};

template <size_t N, size_t I, typename Head, typename... Tail>
std::ostream &outputPointers(std::ostream &os,
                             const Array<void *, N> &pointers) {
    using H = PairTraits<Head>;
    os << typeid(typename H::Type).name() << " *" << H::name.str << ": "
       << pointers[I];

    if constexpr (sizeof...(Tail) == 0) {
        os << "\n  }";
        return os;
    } else {
        os << "\n    ";

        return outputPointers<N, I + 1, Tail...>(os, pointers);
    }
}

template <size_t A, typename... Variables>
std::ostream &operator<<(std::ostream &os,
                         const StructureOfArrays<A, Variables...> &soa) {
    typedef StructureOfArrays<A, Variables...> Soa;
    constexpr size_t n = Soa::num_pointers;

    os << "Soa {\n  ";
    os << "num members: " << n << "\n  ";
    os << "num elements: " << soa.num_elements << "\n  ";
    os << "memory requirement (bytes): " << Soa::getMemReq(soa.num_elements)
       << "\n  ";
    os << "*data: " << soa.data << "\n  ";
    os << "*pointers[]: {\n    ";
    outputPointers<n, 0, Variables...>(os, soa.pointers);
    os << "\n}";

    return os;
}
} // namespace aosoa
