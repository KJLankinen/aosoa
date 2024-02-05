#!/bin/bash

OUT_BIN=aosoa_tests

g++ main.cpp \
    -o $OUT_BIN \
    --std=c++17 \
    -Wall \
    -Werror \
    -Wextra \
    -Wshadow \
    -Wsign-conversion \
    -Wconversion \
    -O2 \
    && ./$OUT_BIN \
    && rm -rf $OUT_BIN \
