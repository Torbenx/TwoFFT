# Development Plan

This is the development plan for the project. It started as a rough outline and is updated
while implementing; the "Design" and "Log" sections record the decisions taken on the way.

## Plan

Status: `[x]` done, `[~]` in progress, `[ ]` open.

### A. ARM-64

- [x] A1. Design and plan TwoFFT implementation for ARM-64
- [x] A2. Implement TwoFFT for initial size range
- [~] A3. Review implementation and run tests and benchmarks
- [~] A4. Iteratively design, implement, test, benchmark and improve new sizes
- [ ] A5. Review entire implementation and apply suite wide refactors/improvements

### X. X86-64

- [ ] X1. Design and plan implementation for AVX2
- [ ] X2. Implement AVX2 plan for an initial range of sizes and run tests
- [ ] X3. Setup benchmark suite and compare against other libraries
- [ ] X4. Run benchmark suite on extended hardware (ask user)
- [ ] X5. Iteratively design, implement, test, benchmark and improve new sizes for AVX2
- [ ] X6. Design and plan implementation for SSE variant (128 bit registers)
- [ ] X7. Iteratively design, implement, test, benchmark and improve new sizes for SSE
- [ ] X8. Design and plan implementation for AVX512 variant
- [ ] X9. Iteratively design, implement, test, benchmark and improve new sizes for AVX512

### Release

- [ ] R1. Review entire code base and apply refactors/improvements
- [ ] R2. Design library interface and distribution
    * Use descriptive names and document potential additional overhead of call.
      For example "Invoking a size for the first time will allocate and compute the weight table"
      or "This will allocate and compute all weight tables".
    * Assume that users will use shared objects (such as a context holding weights) from multiple
      threads. Rely on single threaded-ness only when the library user has explicitly opted in.
    * Interface suggestion: C and C++ interfaces. (Large) C functions defined out-of-line in .c
      file, C++ functions inline and forward to C interface.
    * Distribution suggestion: Generate a single .c and .h (tracked in git) that
      can be copied directly or included via build scripts. Build scripts for cmake + bazel.
- [ ] R3. Implement user interface with documentation
- [ ] R4. Setup github CI for correctness tests
- [ ] R5. Write further documentation (README, etc)
- [ ] R6. Final review and polish

## Design

### Scope

Complex-to-complex, single precision, power-of-two, out-of-place, forward and (unscaled)
inverse. Input and output are interleaved `(re, im)` arrays that must not overlap.

### Algorithm

Decimation in time with natural-order input and output. The transform of `N = 2^n` points
is computed in passes over the data:

1. **Leaf pass** (`in -> out`). Being out-of-place gives exactly one pass that may permute
   for free, and the input is read-only, so it has to be the first one. The leaf pass merges
   the bit reversal permutation with the first 3 to 6 butterfly levels:
   * the input is read through `L` sequential streams (`L` = 8 .. 64, the leaf size),
   * small DFTs are computed *vertically*: every vector lane holds a different
     sub-sequence, so no shuffles and no twiddle loads are needed,
   * the result registers are transposed (the only shuffles of the whole transform) so
     that every lane's results become a contiguous run of the output, and the runs are
     stored at bit reversed positions.

   A vertical DFT of 16 or more points needs more registers than exist (16 complex
   vectors = 32 registers), so larger leaves are done in two fused stages per tile: a
   vertical DFT of 4 or 8 points, and a radix 4 / 8 butterfly on the runs just written
   whose twiddles are constants held in registers.
2. **Middle passes** (in place on `out`). Radix 4 or radix 8 butterflies with twiddles
   read sequentially from per-pass tables. Blocks that exceed the cache are processed
   depth first.
3. **Final pass** (in place on `out`). Same butterflies, but stores interleaved complex.

Between the leaf pass and the final pass the data is kept in *split block* format: for
each group of `VW` complex numbers (`VW` = vector lanes), `VW` real parts followed by `VW`
imaginary parts. Butterflies therefore use plain vector loads and stores.

The inverse uses `ifft(x) = swap(fft(swap(x)))` (swap exchanges re and im). Swapping is
free while (de)interleaving, so only the leaf and the final pass exist twice.

Sizes below 32 use fixed-size transforms.

Considered and rejected: "twiddles by block" (bit reversed twiddle order, as used by
NTT libraries). It avoids twiddle table traffic in large passes, but the first levels
then need table loads and cannot skip trivial multiplications; for the in-cache sizes
that matter most it costs about 10% more arithmetic.

### Code structure

* `src/twofft_simd_<isa>.h`: a small vector abstraction (load, store, add, fma,
  deinterleaving load, transpose, ...) per instruction set.
* `src/twofft_kernels.h`: all kernels, written once against that abstraction and
  instantiated per instruction set by repeated inclusion.
* `src/twofft.c`: tables, plans (how a size is decomposed), context, dispatch, API.
* `tests/`: correctness against an independent double precision reference.
* `bench/bench.c`: comparison with other libraries; `bench/tune.c`: enumerates and times
  all decompositions of a size (used to pick the built-in decompositions).
* `scripts/`: build/run helpers for the test devices (`dev.sh`, `bench.sh`) and for the
  comparison libraries (`fetch_deps.sh`, `build_deps.sh`).

## Log

* A1/A2: first version (vertical 8 and 16 point leaves, radix 4/8 passes) correct on first
  device run; already faster than FFTW, pffft and (mostly) KFR on Cortex-X4.
* A4: the 16 point vertical leaf spilled ~130 register values per tile; replaced by the
  two-stage fused leaves (4x4, 4x8, 8x4, 8x8).
