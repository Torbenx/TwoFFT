#!/bin/sh
# Builds a C file for Android and runs it on a device connected via adb.
#   tools/android.sh <device serial> <cpu mask or -> <source.c> [program arguments]
# Environment: NDK (path of the Android NDK), CFLAGS, LDFLAGS.
set -e
NDK=${NDK:-$(ls -d "$HOME"/opt/android-ndk-* | tail -1)}
CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"
serial=$1; mask=$2; src=$3; shift 3
out=$(mktemp -d)/$(basename "$src" .c)
"$CC" -O2 -Wall -Wextra -std=c11 ${CFLAGS} "$src" -o "$out" ${LDFLAGS} -lm
adb -s "$serial" push "$out" /data/local/tmp/ >/dev/null
name=/data/local/tmp/$(basename "$out")
if [ "$mask" = "-" ]; then adb -s "$serial" shell "$name" "$@"; else adb -s "$serial" shell taskset "$mask" "$name" "$@"; fi
