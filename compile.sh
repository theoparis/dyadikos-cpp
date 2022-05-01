#!/bin/sh
set -e

rm -rf build
meson build
ninja -C build

