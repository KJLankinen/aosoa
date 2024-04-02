#!/bin/bash
cmake \
    -DCMAKE_INSTALL_PREFIX="." \
    -DSYCL_LIBRARY_DIR=/opt/intel/oneapi/2024.1/lib \
    -Dpackage_samples=ON \
    -S . \
    -B build/

cmake --build build/ --target install 
cp build/compile_commands.json .
