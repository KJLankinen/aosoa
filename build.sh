#!/bin/bash
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX="."
cmake --build build/ --target install 
