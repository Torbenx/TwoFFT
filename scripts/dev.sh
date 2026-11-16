#!/usr/bin/env bash
# Development helper: builds the test / benchmark for a target and runs it there.
#
#   scripts/dev.sh <target> test  [test args]
#   scripts/dev.sh <target> bench [bench args]
#   scripts/dev.sh <target> tune  [tune args]     (see bench/tune.c)
#
# Targets:
#   host            this machine
#   android:<serial> an arm64 Android device reachable through adb (needs the NDK, see NDK below)
#   ssh:<user@host>  a machine reachable through ssh with a C compiler (e.g. an Apple Silicon Mac)
#
# Environment:
#   NDK      path of the Android NDK (default: .toolchains/android-ndk-r27c)
#   CFLAGS   extra compiler flags
#   BENCH_DEFS / BENCH_LIBS  extra flags to compile comparison libraries into the benchmark
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:?target}
what=${2:?test or bench}
shift 2

NDK=${NDK:-$root/.toolchains/android-ndk-r27c}
flags="-O2 -std=c11 -Wall -Wextra -I$root/include -I$root/src ${CFLAGS:-}"
case $what in
test) src="$root/src/twofft.c $root/tests/test_twofft.c" ;;
bench) src="$root/src/twofft.c $root/bench/bench.c ${BENCH_DEFS:-}" ;;
tune) src="-DTWOFFT_TUNING $root/src/twofft.c $root/bench/tune.c" ;;
*) echo "unknown action $what (test, bench or tune)" >&2; exit 1 ;;
esac
mkdir -p "$root/build"

case $target in
host)
    out=$root/build/${what}_host
    ${CC:-cc} $flags -D_POSIX_C_SOURCE=200809L $src ${BENCH_LIBS:-} -lm -o "$out"
    "$out" "$@"
    ;;
android:*)
    serial=${target#android:}
    out=$root/build/${what}_android_$serial
    "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang" $flags $src ${BENCH_LIBS:-} -lm -o "$out"
    adb -s "$serial" push "$out" /data/local/tmp/twofft_$what >/dev/null
    adb -s "$serial" shell /data/local/tmp/twofft_$what "$@"
    ;;
ssh:*)
    host=${target#ssh:}
    ssh "$host" 'mkdir -p twofft_dev'
    rsync -a --delete "$root/include" "$root/src" "$root/tests" "$root/bench" "$host:twofft_dev/"
    rflags="-O2 -std=c11 -Wall -Wextra -Iinclude -Isrc ${CFLAGS:-}"
    rsrc=${src//$root\//}
    ssh "$host" "cd twofft_dev && cc $rflags $rsrc ${BENCH_LIBS:-} -lm -o twofft_$what && ./twofft_$what $*"
    ;;
*) echo "unknown target $target" >&2; exit 1 ;;
esac
