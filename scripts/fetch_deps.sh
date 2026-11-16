#!/usr/bin/env bash
# Downloads the libraries the benchmark compares against into .deps/src.
# They are only needed for benchmarking, never for building or using TwoFFT.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/.deps/src"
cd "$root/.deps/src"

if [ ! -d fftw-3.3.10 ]; then
    curl -sSL https://www.fftw.org/fftw-3.3.10.tar.gz | tar xz
fi
[ -d pffft ] || git clone -q --depth 1 https://bitbucket.org/jpommier/pffft.git
[ -d kissfft ] || git clone -q --depth 1 https://github.com/mborgerding/kissfft.git
[ -d kfr ] || git clone -q --depth 1 --branch 6.3.1 https://github.com/kfrlib/kfr.git
echo "dependencies are in $root/.deps/src"
