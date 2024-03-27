#!/bin/bash
cmake \
    -DCMAKE_INSTALL_PREFIX="." \
    -Dpackage_samples=ON \
    -S . \
    -B build/

cmake --build build/ --target install 
cp build/compile_commands.json .
