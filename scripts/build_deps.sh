#!/usr/bin/env bash
# Builds the comparison libraries fetched by fetch_deps.sh for one platform:
#
#   scripts/build_deps.sh host      -> .deps/host
#   scripts/build_deps.sh android   -> .deps/android   (arm64, needs the NDK)
#
# The result is used by scripts/bench.sh.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
platform=${1:?host or android}
src=$root/.deps/src
prefix=$root/.deps/$platform
build=$root/.deps/build-$platform
NDK=${NDK:-$root/.toolchains/android-ndk-r27c}
ndkbin=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
jobs=$(nproc)
mkdir -p "$prefix" "$build"

# ---- FFTW (single precision, all SIMD variants of the platform) ----
if [ ! -f "$prefix/lib/libfftw3f.a" ]; then
    rm -rf "$build/fftw" && mkdir -p "$build/fftw" && cd "$build/fftw"
    case $platform in
    host)
        "$src/fftw-3.3.10/configure" --prefix="$prefix" --enable-float --disable-fortran --disable-shared \
            --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 --enable-fma >/dev/null
        ;;
    android)
        "$src/fftw-3.3.10/configure" --prefix="$prefix" --host=aarch64-linux-android --enable-float \
            --disable-fortran --disable-shared --enable-neon \
            CC="$ndkbin/aarch64-linux-android24-clang" AR="$ndkbin/llvm-ar" RANLIB="$ndkbin/llvm-ranlib" >/dev/null
        ;;
    esac
    make -j"$jobs" >/dev/null 2>&1
    make install >/dev/null
fi

# ---- KFR (C API, static) ----
if [ ! -f "$prefix/lib/libkfr_capi.a" ] && [ -z "${SKIP_KFR:-}" ]; then
    rm -rf "$build/kfr" && mkdir -p "$build/kfr" && cd "$build/kfr"
    common="-DCMAKE_BUILD_TYPE=Release -DKFR_ENABLE_CAPI_BUILD=ON -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    case $platform in
    host)
        cmake -G Ninja $common -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "$src/kfr" >/dev/null
        ;;
    android)
        cmake -G Ninja $common -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
            -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 "$src/kfr" >/dev/null
        ;;
    esac
    ninja >/dev/null
    ninja install >/dev/null
fi

# ---- pffft and KissFFT (plain C files) ----
case $platform in
host) cc=${CC:-cc} ;;
android) cc=$ndkbin/aarch64-linux-android24-clang ;;
esac
mkdir -p "$prefix/lib" "$prefix/include"
$cc -O3 -w -c "$src/pffft/pffft.c" -o "$prefix/lib/pffft.o"
$cc -O3 -w -c "$src/kissfft/kiss_fft.c" -o "$prefix/lib/kiss_fft.o"
cp "$src/pffft/pffft.h" "$src/kissfft/kiss_fft.h" "$prefix/include/"

echo "comparison libraries are in $prefix"
