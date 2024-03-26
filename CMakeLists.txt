cmake_minimum_required(VERSION 3.17)

project(aosoa LANGUAGES CXX VERSION 0.1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(gcc "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(debug_gcc "$<AND:$<CONFIG:Debug>,${gcc}>")

set(gcc_warn -Wall;-Werror;-Wextra;-Wconversion;-Wsign-conversion;)
set(gcc_warn ${gcc_warn};-pedantic-errors;-Wcast-qual;-Wwrite-strings;)
set(gcc_warn ${gcc_warn};-Wcast-align=strict;-Wparentheses;)
set(gcc_warn ${gcc_warn};-Wlogical-op;-Wlogical-not-parentheses;)
set(gcc_warn ${gcc_warn};-Wredundant-decls;-Wformat=2;)
set(gcc_warn ${gcc_warn};-Wformat-security;-Wformat-nonliteral;)
set(gcc_warn ${gcc_warn};-Wnull-dereference;-Winit-self;-Wuninitialized;)
set(gcc_warn ${gcc_warn};-Warith-conversion;-Wduplicated-branches;)
set(gcc_warn ${gcc_warn};-Wpointer-arith;-Wundef;)
set(gcc_warn ${gcc_warn};-Wduplicated-cond;-Wformat-signedness;)

set(gcc_deb_opt -Og;)
set(gcc_rel_opt -O3;)

add_compile_options(
    "$<${gcc}:${gcc_warn}>"
    "$<IF:${debug_gcc},${gcc_deb_opt},${gcc_rel_opt}>"
    )

option(package_tests "Build tests" ON)
if(package_tests)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()

option(package_samples "Build samples" OFF)
if(package_samples)
    add_subdirectory(samples)
endif()

add_subdirectory(src)
