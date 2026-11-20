#!/bin/sh
# Copies the repository to a Mac, builds a C file there and runs it.
#   tools/mac.sh <user@host> <source.c> [program arguments]
# Environment: CFLAGS, LDFLAGS.
set -e
host=$1; src=$2; shift 2
tar cf - src test bench | ssh -o BatchMode=yes "$host" 'rm -rf twofft-dev && mkdir -p twofft-dev && tar xf - -C twofft-dev'
name=$(basename "$src" .c)
ssh -o BatchMode=yes "$host" "cd twofft-dev && mkdir -p build && cc -O2 -Wall -Wextra -std=c11 ${CFLAGS} $src -o build/$name ${LDFLAGS} && build/$name $*"
