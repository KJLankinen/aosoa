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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __NVCC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

// TODO:
// - add cuda/hip/sycl memory ops

namespace detail {
// This namespace contains utility types and functions used by Row and
// StructureOfArrays

// ==== Type equality ====
template <typename T, typename U> struct IsSame {
    constexpr static bool value = false;
};

template <typename T> struct IsSame<T, T> {
    constexpr static bool value = true;
};

// ==== Compile time string ====
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
    operator+(const CompileTimeString<M> rhs) const {
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
    return CompileTimeString<N - 1>(lhs) == rhs;
}

template <size_t N, size_t M>
consteval bool operator==(CompileTimeString<N> lhs, const char (&rhs)[M]) {
    return lhs == CompileTimeString<M - 1>(rhs);
}

template <size_t N, size_t M>
consteval auto operator+(const char (&lhs)[N], CompileTimeString<M> rhs) {
    return CompileTimeString<N - 1>(lhs) + rhs;
}

template <size_t N, size_t M>
consteval auto operator+(CompileTimeString<N> lhs, const char (&rhs)[M]) {
    return lhs + CompileTimeString<M - 1>(rhs);
}

// Deduction guide
template <size_t N>
CompileTimeString(const char (&)[N]) -> CompileTimeString<N - 1>;

template <CompileTimeString Cts> constexpr auto operator""_cts() { return Cts; }

// ==== Variable ====
// - Binds a type and a CompileTimeString together
template <typename, CompileTimeString> struct Variable {};

// ==== ExtractFrom ====
// - Used to extract the name and the type from a Variable<Type, Name>
template <typename> struct ExtractFrom;
template <typename T, CompileTimeString Cts>
struct ExtractFrom<Variable<T, Cts>> {
    using Type = T;
    static constexpr CompileTimeString name = Cts;
};

// ==== NthType ====
// - Get the Nth type from a parameter pack of types
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

  public:
    using Type = std::invoke_result_t<decltype(ofType<0, Types...>)>;
};

// ==== IndexOfString ====
// - Find the index of string from a parameter pack
template <CompileTimeString MatchStr, CompileTimeString... Strings>
struct IndexOfString {
  private:
    template <size_t N> consteval static size_t index() { return ~0ul; }

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

// ==== FindString ====
// - Check if the string is in the parameter pack
template <CompileTimeString MatchStr, CompileTimeString... Strings>
struct FindString {
    constexpr static bool value =
        IndexOfString<MatchStr, Strings...>::i != IndexOfString<MatchStr>::i;
};

// ==== GetType ====
// - Get index and type corresponding to Cts
template <CompileTimeString Cts, typename... Variables> struct GetType {
    static constexpr size_t i =
        IndexOfString<Cts, ExtractFrom<Variables>::name...>::i;
    using Type =
        typename NthType<i, typename ExtractFrom<Variables>::Type...>::Type;
};

// ==== MemoryOperations ====
// - These are used by StructureOfArrays to perform
//     - memory allocation at construction
//     - deallocation at unique_ptr destruction
//     - memcpy and memset between pointers
//     - update of the remote accessor
// - One should create similar functors for different APIs like Cuda, Hip and
//   Sycl
template <bool HostAccessRequiresCopy, typename Allocate, typename Free,
          typename Copy, typename Set>
struct MemoryOperations {
    using Deallocate = Free;
    static constexpr bool host_access_requires_copy = HostAccessRequiresCopy;
    Allocate allocate = {};
    Copy memcpy = {};
    Set memset = {};
};

struct CAllocator {
    void *operator()(size_t bytes) const noexcept { return std::malloc(bytes); }
};

struct CDeallocator {
    void operator()(void *ptr) const noexcept { std::free(ptr); }
};

struct CMemcpy {
    void operator()(void *dst, const void *src, size_t bytes) const noexcept {
        std::memcpy(dst, src, bytes);
    }
};

struct CMemset {
    void operator()(void *dst, int pattern, size_t bytes) const noexcept {
        std::memset(dst, pattern, bytes);
    }
};

typedef MemoryOperations<false, CAllocator, CDeallocator, CMemcpy, CMemset>
    CMemoryOperations;

} // namespace detail

namespace aosoa {
using namespace detail;
// ==== Row ====
// - Row represents a row from a structure of arrays layout
// - Given a structure of arrays with N arrays, the aggregation of the ith value
//   of each array represent the ith row of the structure of arrays and can be
//   instantiated as a variable of type Row.
//
// - Row is implemented as something between a tuple and a normal struct.
// - It's a tuple in the sense that it's a recursive type, but it's a struct in
//   the sense that you can (only) access it's members by name, using a
//   templated syntax like auto foo = row.get<"foo">();
template <typename... Vars> struct Row;
template <typename T, typename... Vars> void to_json(T &, const Row<Vars...> &);
template <typename T, typename... Vars>
void from_json(const T &, Row<Vars...> &);

// Specialize Row for a single type
template <typename Var> struct Row<Var> {
    template <typename... Ts> friend struct Row;

    template <typename T, typename... Ts>
    friend void to_json(T &, const Row<Ts...> &);

    template <typename T, typename... Ts>
    friend void from_json(const T &, Row<Ts...> &);

    template <typename T>
    friend std::ostream &operator<<(std::ostream &, const Row<T> &);

    using Type = ExtractFrom<Var>::Type;
    static constexpr auto name = ExtractFrom<Var>::name;

  private:
    Type head = {};

  public:
    HOST DEVICE constexpr Row() {}
    HOST DEVICE constexpr Row(const Row<Var> &row) : head(row.head) {}
    HOST DEVICE constexpr Row(Type t) : head(t) {}

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] constexpr const auto &get() const {
        static_assert(EqualStrings<name, Cts>::value,
                      "No member with such name");
        return head;
    }

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] constexpr auto &get() {
        static_assert(EqualStrings<name, Cts>::value,
                      "No member with such name");
        return head;
    }

    template <CompileTimeString Cts, typename U>
    HOST DEVICE constexpr void set(U u) {
        get<Cts>() = u;
    }

    bool operator==(const Row<Var> &rhs) const { return head == rhs.head; }

  private:
    std::ostream &output(std::ostream &os) const {
        os << name.str << ": " << head;
        return os;
    }

    template <typename T> void convert_to_json(T &j) const {
        j[name.str] = head;
    }

    template <typename T> void construct_from_json(const T &j) {
        j.at(name.str).get_to(head);
    }

    template <CompileTimeString MemberName, CompileTimeString Candidate>
    struct EqualStrings {
        constexpr static bool value = MemberName == Candidate;
    };
};

// Specialize Row for two or more types
template <typename Var1, typename Var2, typename... Vars>
struct Row<Var1, Var2, Vars...> {
    template <typename... Ts> friend struct Row;

    template <typename T, typename... Ts>
    friend void to_json(T &, const Row<Ts...> &);

    template <typename T, typename... Ts>
    friend void from_json(const T &, Row<Ts...> &);

    template <typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Row<Ts...> &);

    using Type = ExtractFrom<Var1>::Type;
    static constexpr auto name = ExtractFrom<Var1>::name;

  private:
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

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] constexpr const auto &get() const {
        if constexpr (Cts == name) {
            return head;
        } else {
            return tail.template get<Cts>();
        }
    }

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] constexpr auto &get() {
        if constexpr (Cts == name) {
            return head;
        } else {
            return tail.template get<Cts>();
        }
    }

    template <CompileTimeString Cts, typename U>
    HOST DEVICE constexpr void set(U u) {
        get<Cts>() = u;
    }

    bool operator==(const ThisType &rhs) const {
        return head == rhs.head && tail == rhs.tail;
    }

  private:
    std::ostream &output(std::ostream &os) const {
        os << name.str << ": " << head << "\n  ";
        return tail.output(os);
    }

    template <typename T> void convert_to_json(T &j) const {
        j[name.str] = head;
        tail.convert_to_json(j);
    }

    template <typename T> void construct_from_json(const T &j) {
        j.at(name.str).get_to(head);
        tail.construct_from_json(j);
    }

    // ==== Uniqueness of names ====
    // Asserting at compile time that all the names in the template parameters
    // are unique.
    static_assert(!FindString<ExtractFrom<Var1>::name, ExtractFrom<Var2>::name,
                              ExtractFrom<Vars>::name...>::value,
                  "Found a clashing name");

  public:
    // This helps Accessor assert it's names are unique by asserting
    // that the resulting Row type has unique names
    constexpr static bool unique_names =
        !FindString<ExtractFrom<Var1>::name, ExtractFrom<Var2>::name,
                    ExtractFrom<Vars>::name...>::value;
};

template <typename... Vars>
std::ostream &operator<<(std::ostream &os, const Row<Vars...> &row) {
    os << "Row {\n  ";
    return row.output(os) << "\n}";
}

// From Row to json
template <typename T, typename... Vars>
void to_json(T &j, const Row<Vars...> &from) {
    from.convert_to_json(j);
}

// From json to Row
template <typename T, typename... Vars>
void from_json(const T &j, Row<Vars...> &to) {
    to.construct_from_json(j);
}

// ==== AlignedPointers ====
// - An array of void *
// - Each pointer is aligned by the maximum alignment of all the types in
//   Variables... and MIN_ALIGN
template <size_t MIN_ALIGN, typename... Variables> struct AlignedPointers {
    static constexpr size_t size = sizeof...(Variables);

  private:
    void *pointers[size] = {};

  public:
    // Construct a target for copy
    HOST DEVICE AlignedPointers() {}

    AlignedPointers(size_t n, void *ptr) {
        alignPointers<size, 0, typename ExtractFrom<Variables>::Type...>(
            ptr, pointers, ~0ul, n);
    }

    HOST DEVICE constexpr auto &operator[](size_t i) { return pointers[i]; }

    HOST DEVICE constexpr const auto &operator[](size_t i) const {
        return pointers[i];
    }

    // ==== Construction related functionality ====
    [[nodiscard]] static size_t getMemReq(size_t n) {
        // Get proper begin alignment: the strictest (largest) alignment
        // requirement between all the types and MIN_ALIGN
        alignas(getAlignment()) uint8_t dummy = 0;
        constexpr size_t max_size = ~size_t(0);
        size_t space = max_size;
        void *pointers[size] = {};
        alignPointers<size, 0, typename ExtractFrom<Variables>::Type...>(
            static_cast<void *>(&dummy), pointers, std::move(space), n);

        const size_t num_bytes = max_size - space;
        // Require a block of (M + 1) * alignment bytes, where M is an integer.
        // The +1 is for manual alignment, if the memory allocation doesn't have
        // a strict enough alignment requirement.
        return num_bytes + bytesMissingFromAlignment(num_bytes) +
               getAlignment();
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
        // ExtractFrom<Variables>::Type...) Aligned {}; return
        // alignof(Aligned);
        struct alignas(MIN_ALIGN) MinAligned {};
        return maxAlign<MinAligned, typename ExtractFrom<Variables>::Type...>();
    }

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

    template <size_t N, size_t>
    static void alignPointers(void *ptr, void *(&)[N], size_t &&space, size_t) {
        // Align the end of last pointer to the getAlignment() byte boundary so
        // the memory requirement is a multiple of getAlignment()
        if (ptr) {
            ptr = std::align(getAlignment(), 1, ptr, space);
        }
    }

    template <size_t N, size_t I, typename Head, typename... Tail>
    static void alignPointers(void *ptr, void *(&pointers)[N], size_t &&space,
                              size_t n) {
        constexpr size_t size_of_type = sizeof(Head);
        if (ptr) {
            ptr = std::align(getAlignment(), size_of_type, ptr, space);
            if (ptr) {
                pointers[I] = ptr;
                ptr = static_cast<void *>(static_cast<Head *>(ptr) + n);
                space -= size_of_type * n;

                alignPointers<N, I + 1, Tail...>(ptr, pointers,
                                                 std::move(space), n);
            }
        }
    }
};

// ==== Accessor ====
// - Used to access data stored in the AlignedPointers array
template <size_t MIN_ALIGN, typename... Variables> struct Accessor {
    using FullRow = Row<Variables...>;

  private:
    using Pointers = AlignedPointers<MIN_ALIGN, Variables...>;
    size_t num_elements = 0;
    Pointers pointers;

  public:
    [[nodiscard]] static size_t getMemReq(size_t n) {
        return Pointers::getMemReq(n);
    }

    [[nodiscard]] consteval static size_t getAlignment() {
        return Pointers::getAlignment();
    }

    HOST DEVICE Accessor() {}

    Accessor(size_t n, void *ptr) : num_elements(n), pointers(n, ptr) {}

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] auto get() const {
        using G = GetType<Cts, Variables...>;
        return static_cast<G::Type *>(pointers[G::i]);
    }

    template <size_t I> HOST DEVICE [[nodiscard]] auto get() const {
        using Type =
            typename NthType<I, typename ExtractFrom<Variables>::Type...>::Type;
        return static_cast<Type *>(pointers[I]);
    }

    template <CompileTimeString Cts, size_t I>
    HOST DEVICE [[nodiscard]] auto get() const {
        return get<Cts>()[I];
    }

    template <CompileTimeString Cts>
    HOST DEVICE [[nodiscard]] auto get(size_t i) const {
        return get<Cts>()[i];
    }

    HOST DEVICE [[nodiscard]] FullRow get(size_t i) const {
        return toRow<Variables...>(i);
    }

    template <CompileTimeString Cts> HOST DEVICE [[nodiscard]] void *&getRef() {
        using G = GetType<Cts, Variables...>;
        return pointers[G::i];
    }

    template <CompileTimeString Cts, typename T>
    HOST DEVICE void set(size_t i, T value) const {
        get<Cts>()[i] = value;
    }

    HOST DEVICE void set(size_t i, const FullRow &t) const {
        fromRow<Variables...>(i, t);
    }

    HOST DEVICE void set(size_t i, FullRow &&t) const {
        fromRow<Variables...>(i, std::move(t));
    }

    HOST DEVICE size_t &size() { return num_elements; }
    HOST DEVICE const size_t &size() const { return num_elements; }

  private:
    template <typename Head, typename... Tail>
    [[nodiscard]] HOST DEVICE auto toRow(size_t i) const {
        using Exctracted = ExtractFrom<Head>;
        if constexpr (sizeof...(Tail) > 0) {
            return Row<Head, Tail...>(get<Exctracted::name>(i),
                                      toRow<Tail...>(i));
        } else {
            return Row<Head>(get<Exctracted::name>(i));
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, const FullRow &row) const {
        using Exctracted = ExtractFrom<Head>;
        set<Exctracted::name, typename Exctracted::Type>(
            i, row.template get<Exctracted::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, row);
        }
    }

    template <typename Head, typename... Tail>
    HOST DEVICE void fromRow(size_t i, FullRow &&row) const {
        using Exctracted = ExtractFrom<Head>;
        set<Exctracted::name, typename Exctracted::Type>(
            i, row.template get<Exctracted::name>());
        if constexpr (sizeof...(Tail) > 0) {
            fromRow<Tail...>(i, std::move(row));
        }
    }
};

// ==== StructureOfArrays ====
// - Used on the host to manage the accessor and the memory
template <size_t MIN_ALIGN, typename MemOps, typename... Variables>
struct StructureOfArrays {
    using ThisAccessor = Accessor<MIN_ALIGN, Variables...>;
    using FullRow = ThisAccessor::FullRow;

  private:
    static_assert(Row<Variables...>::unique_names,
                  "StructureOfArrays has clashing names");

    using CSoa = StructureOfArrays<MIN_ALIGN, CMemoryOperations, Variables...>;

    template <size_t MA, typename M, typename... V>
    friend struct StructureOfArrays;

    const MemOps &memory_ops;
    const size_t max_num_elements;
    std::unique_ptr<uint8_t, typename MemOps::Deallocate> memory;
    ThisAccessor local_accessor;

  public:
    [[nodiscard]] static size_t getMemReq(size_t n) {
        return ThisAccessor::getMemReq(n);
    }

    StructureOfArrays(const MemOps &mem_ops, size_t n,
                      ThisAccessor *accessor = nullptr)
        : memory_ops(mem_ops), max_num_elements(n),
          memory(static_cast<uint8_t *>(
              memory_ops.allocate(getMemReq(max_num_elements)))),
          local_accessor(max_num_elements, static_cast<void *>(memory.get())) {
        updateAccessor(accessor);
        memory_ops.memset(static_cast<void *>(memory.get()), 0,
                          getMemReq(max_num_elements));
    }

    // If the number of elements is very large, use the above constructor and
    // initialize the values in place to avoid running out of memory
    StructureOfArrays(const MemOps &mem_ops, const std::vector<FullRow> &rows,
                      ThisAccessor *accessor = nullptr)
        : StructureOfArrays(mem_ops, rows.size(), accessor) {
        if (memory_ops.host_access_requires_copy) {
            CMemoryOperations c_mem_ops;
            CSoa host_soa(c_mem_ops, rows);

            memory_ops.memcpy(local_accessor.template get<0>(),
                              host_soa.local_accessor.template get<0>(),
                              getAlignedBlockSize());
        } else {
            size_t i = 0;
            for (const auto &row : rows) {
                local_accessor.set(i++, row);
            }
        }
    }

    void decreaseBy(size_t n, ThisAccessor *accessor = nullptr) {
        local_accessor.size() -= std::min(n, local_accessor.size());
        updateAccessor(accessor);
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2>
    void swap(ThisAccessor *accessor = nullptr) {
        using G1 = GetType<Cts1, Variables...>;
        using G2 = GetType<Cts2, Variables...>;

        static_assert(IsSame<typename G1::Type, typename G2::Type>::value,
                      "Mismatched types for swap");

        std::swap(local_accessor.template getRef<Cts1>(),
                  local_accessor.template getRef<Cts2>());

        updateAccessor(accessor);
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2,
              CompileTimeString Cts3, CompileTimeString... Tail>
    void swap(ThisAccessor *accessor = nullptr) {
        swap<Cts1, Cts2>();
        swap<Cts3, Tail...>();
        updateAccessor(accessor);
    }

    void updateAccessor(ThisAccessor *accessor) const {
        updateAccessor(memory_ops.memcpy, accessor);
    }

    template <typename F>
    void updateAccessor(F f, ThisAccessor *accessor) const {
        if (accessor != nullptr) {
            f(static_cast<void *>(accessor),
              static_cast<const void *>(&local_accessor), sizeof(ThisAccessor));
        }
    }

    std::vector<FullRow> getRows() const {
        // This is an expensive function: it copies all the memory twice, if the
        // memory recides on device. The first copy is the raw data from device
        // to host, the second is from soa (= current) layout to aos (= vector
        // of FullRow) layout
        if (memory_ops.host_access_requires_copy) {
            // Create this structure backed by host memory, then call it's
            // version of this function
            CMemoryOperations c_mem_ops;
            CSoa host_soa(c_mem_ops, max_num_elements);
            memory_ops.memcpy(host_soa.local_accessor.template get<0>(),
                              local_accessor.template get<0>(),
                              getAlignedBlockSize());
            host_soa.local_accessor.size() = local_accessor.size();

            return host_soa.getRows();
        } else {
            // Just convert to a vector of rows
            std::vector<FullRow> rows(local_accessor.size());
            std::generate(rows.begin(), rows.end(), [i = 0ul, this]() mutable {
                return local_accessor.get(i++);
            });
            return rows;
        }
    }

    const ThisAccessor &getAccess() const { return local_accessor; }
    ThisAccessor &getAccess() { return local_accessor; }

    // Internal to internal
    template <CompileTimeString DstName, CompileTimeString SrcName>
    void memcpy() const {
        memcpy<DstName, SrcName>(memory_ops.memcpy);
    }

    // Internal to internal
    template <CompileTimeString DstName, CompileTimeString SrcName, typename F>
    void memcpy(F f) const {
        using Dst = GetType<DstName, Variables...>;
        using Src = GetType<SrcName, Variables...>;
        static_assert(IsSame<typename Dst::Type, typename Src::Type>::value,
                      "Mismatched types for memcpy");
        static_assert(DstName != SrcName, "DstName and SrcName are the same");

        f(local_accessor.template get<DstName>(),
          local_accessor.template get<SrcName>(), getMemReq<DstName>());
    }

    // External to internal
    template <CompileTimeString DstName, typename SrcType>
    void memcpy(const SrcType *src) const {
        memcpy<DstName>(memory_ops.memcpy, src);
    }

    // External to internal
    template <CompileTimeString DstName, typename SrcType, typename F>
    void memcpy(F f, const SrcType *src) const {
        using Dst = GetType<DstName, Variables...>;
        static_assert(IsSame<typename Dst::Type, SrcType>::value,
                      "Mismatched types for memcpy");
        f(local_accessor.template get<DstName>(),
          static_cast<const void *>(src), getMemReq<DstName>());
    }

    // Internal to external
    template <CompileTimeString SrcName, typename DstType>
    void memcpy(DstType *dst) const {
        memcpy<SrcName>(memory_ops.memcpy, dst);
    }

    // Internal to external
    template <CompileTimeString SrcName, typename DstType, typename F>
    void memcpy(F f, DstType *dst) const {
        using Src = GetType<SrcName, Variables...>;
        static_assert(IsSame<typename Src::Type, DstType>::value,
                      "Mismatched types for memcpy");

        f(static_cast<void *>(dst), local_accessor.template get<SrcName>(),
          getMemReq<SrcName>());
    }

    // Memset column
    template <CompileTimeString DstName> void memset(int pattern) const {
        memset<DstName>(memory_ops.memset, pattern);
    }

    template <CompileTimeString DstName, typename F>
    void memset(F f, int pattern) const {
        f(local_accessor.template get<DstName>(), pattern,
          getMemReq<DstName>());
    }

  private:
    template <CompileTimeString Cts> [[nodiscard]] size_t getMemReq() const {
        return local_accessor.size() *
               sizeof(typename GetType<Cts, Variables...>::Type);
    }

    [[nodiscard]] uintptr_t getAlignedBlockSize() const {
        return getMemReq(max_num_elements) - ThisAccessor::getAlignment();
    }
};
} // namespace aosoa
