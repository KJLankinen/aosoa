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

namespace detail {
// This namespace contains utility types and functions used by Row and
// StructureOfArrays

// ==== Array ====
// - Usable on devices
template <typename T, size_t N> struct Array {
    T items[N] = {};

    HOST DEVICE constexpr T &operator[](size_t i) { return items[i]; }
    HOST DEVICE constexpr const T &operator[](size_t i) const {
        return items[i];
    }
    HOST DEVICE constexpr size_t size() const { return N; }
    HOST DEVICE constexpr T *data() const { return &items; }
};

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

template <CompileTimeString Cts> constexpr auto operator""_cts() { return Cts; }

// ==== Variable ====
// - Binds a type and a CompileTimeString together
template <typename, CompileTimeString> struct Variable {};

// ==== Abstract MemoryOps interface ====
// - Override this for C, Cuda, Hip, Sycl and others as needed
struct MemoryOps {
    virtual void *allocate(size_t bytes) const = 0;
    virtual void deallocate(void *ptr) const = 0;
    virtual void memcpy(void *dst, const void *src, size_t bytes,
                        bool synchronous = true) const = 0;
    virtual void memset(void *dst, int pattern, size_t bytes,
                        bool synchronous = true) const = 0;
    virtual void update(void *dst, const void *src, size_t bytes) const = 0;
    virtual bool accessOnHostRequiresMemcpy() const = 0;
};

// ==== CMemoryOps ====
struct CMemoryOps : MemoryOps {
    void *allocate(size_t bytes) const { return malloc(bytes); }
    void deallocate(void *ptr) const { free(ptr); }
    void memcpy(void *dst, const void *src, size_t bytes, bool) const {
        std::memcpy(dst, src, bytes);
    }
    void memset(void *dst, int pattern, size_t bytes, bool) const {
        std::memset(dst, pattern, bytes);
    }
    void update(void *dst, const void *src, size_t bytes) const {
        std::memcpy(dst, src, bytes);
    }
    bool accessOnHostRequiresMemcpy() const { return false; }
};

// ==== PairTraits ====
// - Used to extract the name and the type from a Variable<Type, Name>
template <typename> struct PairTraits;
template <typename T, CompileTimeString Cts>
struct PairTraits<Variable<T, Cts>> {
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

    static constexpr auto t = ofType<0, Types...>();

  public:
    // no std?
    using Type = std::remove_const_t<decltype(t)>;
};

// ==== IndexOfString ====
// - Find the index of string from a parameter pack
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

// ==== Buffer ====
// - A very simple buffer that allocates/deallocates memory with the given
// MemoryOps
struct Buffer {
    const MemoryOps &memory_ops;
    void *const data = nullptr;

    Buffer(const MemoryOps &mem_ops, size_t bytes)
        : memory_ops(mem_ops), data(memory_ops.allocate(bytes)) {}
    ~Buffer() { memory_ops.deallocate(data); }
};
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

// Specialize Row for a single type
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

// Specialize Row for two or more types
template <typename Var1, typename Var2, typename... Vars>
struct Row<Var1, Var2, Vars...> {
    template <typename... Ts> friend struct Row;

    using Type = PairTraits<Var1>::Type;
    static constexpr auto name = PairTraits<Var1>::name;

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
    // This helps Accessor assert it's names are unique by asserting
    // that the resulting Row type has unique names
    constexpr static bool unique_names =
        EqualIndices<index_of_var1, num_strings>::value;
};

template <typename... Vars>
std::ostream &operator<<(std::ostream &os, const Row<Vars...> &row) {
    os << "Row {\n  ";
    return row.output(os) << "\n}";
}

// ==== StructureOfArrays ====
template <size_t MIN_ALIGN, typename... Variables> struct StructureOfArrays {
  private:
    static_assert(Row<Variables...>::unique_names,
                  "StructureOfArrays has clashing names");

    static constexpr size_t num_pointers = sizeof...(Variables);
    using Pointers = Array<void *, num_pointers>;

  public:
    template <CompileTimeString Cts> struct GetType {
        static constexpr size_t i =
            IndexOfString<Cts, PairTraits<Variables>::name...>::i;
        using Type =
            typename NthType<i, typename PairTraits<Variables>::Type...>::Type;
    };

    using FullRow = Row<Variables...>;

    struct Accessor {
      private:
        template <size_t I, typename... T> friend struct StructureOfArrays;
        size_t num_elements = 0;
        Pointers pointers = {};

      public:
        HOST DEVICE Accessor() {}
        HOST DEVICE Accessor(size_t n, const Pointers &p)
            : num_elements(n), pointers(p) {}

        template <CompileTimeString Cts>
        HOST DEVICE [[nodiscard]] auto get() const {
            using G = GetType<Cts>;
            return static_cast<G::Type *>(pointers[G::i]);
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

        HOST DEVICE size_t size() const { return num_elements; }

      private:
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

  private:
    const MemoryOps &memory_ops;
    const size_t max_num_elements;
    const Buffer buffer;
    Accessor local_accessor = {};
    Accessor *const remote_accessor = nullptr;

  public:
    StructureOfArrays(const MemoryOps &mem_ops,
                      const std::vector<FullRow> &rows, Accessor *accessor)
        : memory_ops(mem_ops), max_num_elements(rows.size()),
          buffer(memory_ops, getMemReq(max_num_elements)),
          local_accessor(
              max_num_elements,
              makeAlignedPointers<0, typename PairTraits<Variables>::Type...>(
                  buffer.data, Pointers{}, ~0ul, max_num_elements)),
          remote_accessor(accessor) {
        // Setup the accessor
        updateAccessor();

        // Initialize the memory with the given vector
        if (memory_ops.accessOnHostRequiresMemcpy()) {
            CMemoryOps c_mem_ops;
            Accessor a;
            StructureOfArrays<MIN_ALIGN, Variables...> host_soa(c_mem_ops, rows,
                                                                &a);
            memory_ops.memcpy(local_accessor.pointers[0],
                              host_soa.local_accessor.pointers[0],
                              getAlignedBlockSize(), true);
        } else {
            size_t i = 0;
            for (const auto &row : rows) {
                local_accessor.set(i++, row);
            }
        }
    }

    StructureOfArrays(const MemoryOps &mem_ops, size_t n, Accessor *accessor)
        : StructureOfArrays(mem_ops, std::vector<FullRow>(n), accessor) {}

    [[nodiscard]] static size_t getMemReq(size_t n) {
        // Get proper begin alignment: the strictest (largest) alignment
        // requirement between all the types and the MIN_ALIGN
        alignas(getAlignment()) uint8_t dummy = 0;
        constexpr size_t max_size = ~size_t(0);
        size_t space = max_size;
        [[maybe_unused]] const auto pointers =
            makeAlignedPointers<0, typename PairTraits<Variables>::Type...>(
                static_cast<void *>(&dummy), Pointers{}, std::move(space), n);

        const size_t num_bytes = max_size - space;
        // Require a block of (M + 1) * alignment bytes, where M is an integer.
        // The +1 is for manual alignment, if the memory allocation doesn't have
        // a strict enough alignment requirement.
        return num_bytes + bytesMissingFromAlignment(num_bytes) +
               getAlignment();
    }

    template <CompileTimeString Cts> [[nodiscard]] size_t getMemReq() const {
        return local_accessor.num_elements *
               sizeof(typename GetType<Cts>::Type);
    }

    [[nodiscard]] uintptr_t getAlignedBlockSize() const {
        return getMemReq(max_num_elements) - getAlignment();
    }

    [[nodiscard]] uintptr_t getAlignmentBytes() const {
        // How many bytes from the beginning of the data pointer are unused due
        // to alignment requirement
        return static_cast<uintptr_t>(
            static_cast<uint8_t *>(local_accessor.pointers[0]) -
            static_cast<uint8_t *>(buffer.data));
    }

    [[nodiscard]] void *data() const { return buffer.data; }

    void decreaseBy(size_t n, bool update_accessor = false) {
        local_accessor.num_elements -= std::min(n, local_accessor.num_elements);
        if (update_accessor) {
            updateAccessor();
        }
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2>
    void swap(bool update_accessor = false) {
        using G1 = GetType<Cts1>;
        using G2 = GetType<Cts2>;

        static_assert(IsSame<typename G1::Type, typename G2::Type>::value,
                      "Mismatched types for swap");

        std::swap(local_accessor.pointers[G1::i],
                  local_accessor.pointers[G2::i]);

        if (update_accessor) {
            updateAccessor();
        }
    }

    template <CompileTimeString Cts1, CompileTimeString Cts2,
              CompileTimeString Cts3, CompileTimeString... Tail>
    void swap(bool update_accessor = false) {
        swap<Cts1, Cts2>();
        swap<Cts3, Tail...>();

        if (update_accessor) {
            updateAccessor();
        }
    }

    template <typename F, typename... Args>
    auto updateAccessor(F f, Args... args) const {
        return f(static_cast<void *>(remote_accessor),
                 static_cast<const void *>(&local_accessor), sizeof(Accessor),
                 args...);
    }

    auto updateAccessor() const {
        return memory_ops.update(static_cast<void *>(remote_accessor),
                                 static_cast<const void *>(&local_accessor),
                                 sizeof(Accessor));
    }

    std::vector<FullRow> getRows() const {
        // This is an expensive function: it copies all the memory twice, if the
        // memory recides on device. The first copy is the raw data from device
        // to host, the second is from soa (= current) layout to aos (= vector
        // of FullRow) layout
        if (memory_ops.accessOnHostRequiresMemcpy()) {
            // Create this structure backed by host memory, then call it's
            // version of this function
            CMemoryOps c_mem_ops;
            Accessor a;
            StructureOfArrays<MIN_ALIGN, Variables...> host_soa(
                c_mem_ops, max_num_elements, &a);
            memory_ops.memcpy(host_soa.local_accessor.pointers[0],
                              local_accessor.pointers[0], getAlignedBlockSize(),
                              true);
            host_soa.local_accessor.num_elements = local_accessor.num_elements;

            return host_soa.getRows();
        } else {
            // Just convert to a vector of rows
            std::vector<FullRow> rows(local_accessor.num_elements);
            std::generate(rows.begin(), rows.end(), [i = 0ul, this]() mutable {
                return local_accessor.get(i++);
            });
            return rows;
        }
    }

    // Internal to internal
    template <CompileTimeString Dst, CompileTimeString Src, typename F,
              typename... Args>
    auto memcpy(F f, Args... args) {
        using Gd = GetType<Dst>;
        using Gs = GetType<Src>;
        static_assert(IsSame<typename Gd::Type, typename Gs::Type>::value,
                      "Mismatched types for memcpy");

        return memcpy(f, local_accessor.pointers[Gd::i],
                      local_accessor.pointers[Gs::i], getMemReq<Dst>(),
                      args...);
    }

    // Internal to internal
    template <CompileTimeString Dst, CompileTimeString Src> auto memcpy() {
        using Gd = GetType<Dst>;
        using Gs = GetType<Src>;
        static_assert(IsSame<typename Gd::Type, typename Gs::Type>::value,
                      "Mismatched types for memcpy");

        return memcpy(local_accessor.pointers[Gd::i],
                      local_accessor.pointers[Gs::i], getMemReq<Dst>());
    }

    // External to internal
    template <CompileTimeString Dst, typename SrcType, typename F,
              typename... Args>
    auto memcpy(F f, const SrcType *src, Args... args) {
        using G = GetType<Dst>;
        static_assert(IsSame<typename G::Type, SrcType>::value,
                      "Mismatched types for memcpy");
        return memcpy(f, local_accessor.pointers[G::i],
                      static_cast<const void *>(src), getMemReq<Dst>(),
                      args...);
    }

    // External to internal
    template <CompileTimeString Dst, typename SrcType>
    auto memcpy(const SrcType *src) {
        using G = GetType<Dst>;
        static_assert(IsSame<typename G::Type, SrcType>::value,
                      "Mismatched types for memcpy");
        return memcpy(local_accessor.pointers[G::i],
                      static_cast<const void *>(src), getMemReq<Dst>());
    }

    // Internal to external
    template <CompileTimeString Src, typename DstType, typename F,
              typename... Args>
    auto memcpy(F f, DstType *dst, Args... args) {
        using G = GetType<Src>;
        static_assert(IsSame<typename G::Type, DstType>::value,
                      "Mismatched types for memcpy");
        return memcpy(f, static_cast<void *>(dst),
                      local_accessor.pointers[G::i], getMemReq<Src>(), args...);
    }

    // Internal to external
    template <CompileTimeString Src, typename DstType>
    auto memcpy(DstType *dst) {
        using G = GetType<Src>;
        static_assert(IsSame<typename G::Type, DstType>::value,
                      "Mismatched types for memcpy");
        return memcpy(static_cast<void *>(dst), local_accessor.pointers[G::i],
                      getMemReq<Src>());
    }

    template <CompileTimeString Dst, typename F, typename... Args>
    auto memset(F f, int pattern, Args... args) {
        using G = GetType<Dst>;
        return f(local_accessor.pointers[G::i], pattern, getMemReq<Dst>(),
                 args...);
    }

    template <CompileTimeString Dst> auto memset(int pattern) {
        using G = GetType<Dst>;
        return memset(local_accessor.pointers[G::i], pattern, getMemReq<Dst>());
    }

  private:
    template <typename F, typename... Args>
    auto memcpy(F f, void *dst, const void *src, size_t bytes,
                Args... args) const {
        return f(dst, src, bytes, args...);
    }

    auto memcpy(void *dst, const void *src, size_t bytes) const {
        return memory_ops.memcpy(dst, src, bytes);
    }

    template <typename F, typename... Args>
    auto memset(F f, void *dst, int pattern, size_t bytes, Args... args) const {
        return f(dst, pattern, bytes, args...);
    }

    auto memset(void *dst, int pattern, size_t bytes) const {
        return memory_ops.memset(dst, pattern, bytes);
    }

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
                        size_t n) {
        constexpr size_t size_of_type = sizeof(Head);
        if (ptr) {
            ptr = std::align(getAlignment(), size_of_type, ptr, space);
            if (ptr) {
                pointers[I] = ptr;
                ptr = static_cast<void *>(static_cast<Head *>(ptr) + n);
                space -= size_of_type * n;

                return makeAlignedPointers<I + 1, Tail...>(ptr, pointers,
                                                           std::move(space), n);
            }
        }

        return Pointers{};
    }
};
} // namespace aosoa
