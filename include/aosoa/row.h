/*
    MIT License

    Copyright (c) 2024 Juhana Lankinen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#pragma once

#include <ostream>

#include "compile_time_string.h"
#include "definitions.h"
#include "type_operations.h"

namespace aosoa {
// - Row represents a row from a structure of arrays layout
// - Given a structure of arrays with N arrays, the aggregation of the ith value
//   of each array represent the ith row of the structure of arrays and can be
//   instantiated as a variable of type Row.
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

    using Type = typename Var::Type;
    static constexpr auto name = Var::name;

  private:
    Type head = {};

  public:
    HOST DEVICE constexpr Row() {}
    HOST DEVICE constexpr Row(const Row<Var> &row) : head(row.head) {}
    HOST DEVICE constexpr Row(Type t) : head(t) {}

    template <CompileTimeString Cts>
    [[nodiscard]] HOST DEVICE constexpr const auto &get() const {
        static_assert(EqualStrings<name, Cts>::value,
                      "No member with such name");
        return head;
    }

    template <CompileTimeString Cts>
    [[nodiscard]] HOST DEVICE constexpr auto &get() {
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

    template <typename T>
    friend std::ostream &operator<<(std::ostream &, const Row<T> &);

    template <typename T1, typename T2, typename... Ts>
    friend std::ostream &operator<<(std::ostream &, const Row<T1, T2, Ts...> &);

    using Type = typename Var1::Type;
    static constexpr auto name = Var1::name;

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
    [[nodiscard]] HOST DEVICE constexpr const auto &get() const {
        if constexpr (Cts == name) {
            return head;
        } else {
            return tail.template get<Cts>();
        }
    }

    template <CompileTimeString Cts>
    [[nodiscard]] HOST DEVICE constexpr auto &get() {
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
    static_assert(
        !Is<Var1::name>::template ContainedIn<Var2::name, Vars::name...>::value,
        "Found a clashing name");

  public:
    // This helps Accessor assert it's names are unique by asserting
    // that the resulting Row type has unique names
    constexpr static bool unique_names =
        !Is<Var1::name>::template ContainedIn<Var2::name, Vars::name...>::value;
};

template <typename Var1, typename Var2, typename... Vars>
std::ostream &operator<<(std::ostream &os,
                         const Row<Var1, Var2, Vars...> &row) {
    os << "Row {\n  ";
    return row.output(os) << "\n}";
}

template <typename Var>
std::ostream &operator<<(std::ostream &os, const Row<Var> &row) {
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
} // namespace aosoa
