#!/bin/sh
set -e

rm -rf build
cd shaders/
./compile.sh
cd ../
meson build
ninja -C build

