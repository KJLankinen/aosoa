#!/bin/bash

if [ $# -eq 0 ] || [ ! -d "$1" ]
then
    echo "Give the base directory of the project"
    exit 1
fi

base_dir=$1

out_bin=aosoa_tests

g++ $base_dir/tests/tests.cpp \
    -o $out_bin \
    --std=c++20 \
    -Wall \
    -Werror \
    -Wextra \
    -Wshadow \
    -Wsign-conversion \
    -Wconversion \
    -O2 \
    -isystem "tests/include/tabulate/include" \
    -fdiagnostics-show-template-tree \
    && ./$out_bin

rm -rf $out_bin
