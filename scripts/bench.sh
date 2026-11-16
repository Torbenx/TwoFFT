#!/usr/bin/env bash
# Builds the benchmark together with the comparison libraries and runs it on a target.
#
#   scripts/bench.sh host             [bench args]
#   scripts/bench.sh android:<serial> [bench args]
#   scripts/bench.sh ssh:<user@host>  [bench args]     (macOS; adds Apple vDSP)
#
# Requires scripts/fetch_deps.sh and, for host / android, scripts/build_deps.sh <platform>.
# On an ssh target the libraries are built remotely on first use.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:?target}
shift
NDK=${NDK:-$root/.toolchains/android-ndk-r27c}
flags="-O2 -std=c11 -Wall -Wextra ${CFLAGS:-}"
defs="-DBENCH_FFTW -DBENCH_PFFFT -DBENCH_KISSFFT"
mkdir -p "$root/build"

case $target in
host | android:*)
    if [ "$target" = host ]; then
        platform=host
        cc="${CC:-cc} -D_GNU_SOURCE"
    else
        platform=android
        cc=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang
    fi
    deps=$root/.deps/$platform
    libs="$deps/lib/pffft.o $deps/lib/kiss_fft.o $deps/lib/libfftw3f.a"
    if [ -f "$deps/lib/libkfr_capi.so" ]; then
        defs="$defs -DBENCH_KFR"
        libs="$libs -L$deps/lib -lkfr_capi"
    fi
    out=$root/build/bench_all_$platform
    $cc $flags $defs -I"$root/include" -I"$root/src" -I"$deps/include" "$root/src/twofft.c" "$root/bench/bench.c" \
        $libs -lm -o "$out"
    if [ "$platform" = host ]; then
        LD_LIBRARY_PATH=$deps/lib "$out" "$@"
    else
        serial=${target#android:}
        adb -s "$serial" push "$out" /data/local/tmp/twofft_bench_all >/dev/null
        [ -f "$deps/lib/libkfr_capi.so" ] && adb -s "$serial" push "$deps/lib/libkfr_capi.so" /data/local/tmp/ >/dev/null
        adb -s "$serial" shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/twofft_bench_all $*"
    fi
    ;;
ssh:*)
    host=${target#ssh:}
    ssh "$host" 'mkdir -p twofft_dev twofft_deps'
    rsync -a --delete "$root/include" "$root/src" "$root/tests" "$root/bench" "$host:twofft_dev/"
    if ! ssh "$host" 'test -f twofft_deps/lib/libfftw3f.a -a -f twofft_deps/lib/kiss_fft.o'; then
        rsync -a "$root/.deps/src/fftw-3.3.10" "$root/.deps/src/pffft" "$root/.deps/src/kissfft" "$host:twofft_deps/src/"
        ssh "$host" 'set -e; cd twofft_deps; p=$PWD; mkdir -p lib include build-fftw; cd build-fftw
            ../src/fftw-3.3.10/configure --prefix=$p --enable-float --disable-fortran --disable-shared --enable-neon >/dev/null
            make -j8 >/dev/null 2>&1; make install >/dev/null; cd ..
            cc -O3 -w -c src/pffft/pffft.c -o lib/pffft.o; cc -O3 -w -c src/kissfft/kiss_fft.c -o lib/kiss_fft.o
            cp src/pffft/pffft.h src/kissfft/kiss_fft.h include/'
    fi
    kfr=""
    if ssh "$host" 'test -f twofft_deps/lib/libkfr_capi.dylib'; then
        kfr="-DBENCH_KFR -Ltwofft_deps/lib -lkfr_capi -Wl,-rpath,\$HOME/twofft_deps/lib"
    fi
    ssh "$host" "cc $flags $defs -DBENCH_VDSP -Itwofft_dev/include -Itwofft_dev/src -Itwofft_deps/include \
        twofft_dev/src/twofft.c twofft_dev/bench/bench.c twofft_deps/lib/pffft.o twofft_deps/lib/kiss_fft.o \
        twofft_deps/lib/libfftw3f.a $kfr -framework Accelerate -lm -o twofft_dev/bench_all && twofft_dev/bench_all $*"
    ;;
*) echo "unknown target $target" >&2; exit 1 ;;
esac
