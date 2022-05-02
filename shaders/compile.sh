#!/bin/sh
set -e

glslangValidator -V shader.vert
glslangValidator -V shader.frag
