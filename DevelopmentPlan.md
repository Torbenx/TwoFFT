# Development Plan

This is the rough development plan for the project. It is updated while implementing: finished
steps are marked, findings and decisions are recorded below the plan.

## Plan

### A. ARM-64

- [x] A1. Design and plan TwoFFT implementation for ARM-64
- [x] A2. Implement TwoFFT for initial size range (64 .. 2^16)
- [x] A3. Review implementation and run tests and benchmarks
- [x] A4. Iteratively design, implement, test, benchmark and improve new sizes
      (1 .. 32 with dedicated kernels, large sizes with cache aware first pass and blocking)
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
      can be copied directly or include via build scripts. Build scripts for cmake + bazel.
- [ ] R3. Implement user interface with documentation
- [ ] R4. Setup github CI for correctness tests
- [ ] R5. Write further documentation (README, etc)
- [ ] R6. Final review and polish

### Possible later work

- ARMv8.3 (FCMA) backend working on interleaved complex numbers: same number of floating point
  operations, but no deinterleaving / transposition shuffles (about 15% fewer uops at N = 1024)
  and room for radix-16 passes.
- Scaled ("tangent") butterflies for cores that eliminate vector register moves (Apple).

## Design

The design is documented where it is implemented:

* `src/tw_weights.h`  pass structure, weight tables, order of the first pass
* `src/tw_scalar.h`   portable implementation of exactly the same passes; reference for the tests
* `src/tw_arm64.h`    register conventions and kernels of the ARM64 backend

Summary: a transform of size N is split by the first pass into 8 sub-FFTs of size N/8 that live
in the 8 lanes of 64 byte "lines" (split real / imaginary parts). The first pass reads the
interleaved input, also does the bit-reversal and, depending on log2(N) mod 3, the first 0..2
stages of the sub-FFTs. All further passes are in-place radix-8 passes that need only 7 scalar
weights per butterfly, because all lanes of a line share them. The last pass writes interleaved
output. The backward transform is the forward transform with real and imaginary parts exchanged
on input and output, which costs nothing.

## Findings

### ARM64 (A1 - A4)

Test devices and their cores (measured with `bench/uarch.c`):

| core                         | FP ops / cycle | notes                                           |
|------------------------------|----------------|-------------------------------------------------|
| Kryo 360 Gold (A75), Pixel 3a| 1              | `ldp q` takes 4 cycles                          |
| Cortex-A73, Galaxy A50       | 1              | 2 cycles per vector store                       |
| Cortex-A720, Galaxy S24      | 2              |                                                 |
| Cortex-X4, Galaxy S24        | 4              | `st2` takes 3 cycles, vector `mov` not free     |
| Apple M2 (P core)            | 4 (fmla: 3)    | vector `mov` is free, slow scattered stores     |

* The kernels run at the floating point throughput limit of the cores: on the M2 the radix-8
  butterfly (84 FP operations per 32 complex numbers) takes 22.7 cycles, the limit is 21.
  Faster kernels therefore need fewer operations, not better scheduling.
* The operation count per stage is the same for radix 4, 8 and 16 (about 3.5 vector operations
  per complex vector and stage); radix 8 was chosen because it is the largest radix that fits
  32 registers and minimises loads and stores.
* Butterflies with fused multiply-add and scaled weights (2.75 operations) need a register copy
  per butterfly on ARM because `fmla` is destructive. This is a loss on all Cortex cores.
* `st2` was replaced by `zip1/zip2 + stp` (never slower, much faster on Cortex-X4).
* Large transforms: the radix-8 passes stay fast at any size (depth first blocking on top).
  The first pass was the problem: its 8..32 input streams are a power of two apart and compete
  for the same cache lines, and it writes the output in bit-reversed order. It now reads the
  input through a 16 KiB stack buffer in chunks and, on Apple cores only, visits the tiles in
  a blocked order (`TW_FIRST_LOG2_PLACES_*`), which makes the output contiguous runs.
  Software prefetching of the output did not help anywhere.

State after A4, time per transform in ns (FFTW 3.3.10 with FFTW_MEASURE, NEON enabled):

| N     | M2: TwoFFT | FFTW  | vDSP | pffft | S24 (X4): TwoFFT | FFTW  | pffft |
|-------|-----------:|------:|-----:|------:|-----------------:|------:|------:|
| 16    | 9.1        | 14.6  | 11.2 | 17.7  | 9.7              | 19.6  | 20.3  |
| 64    | 39.7       | 34.5  | 45.8 | 58.9  | 45.8             | 54.3  | 74.4  |
| 256   | 176        | 157   | 236  | 235   | 247              | 238   | 330   |
| 1024  | 828        | 747   | 791  | 1107  | 1138             | 1280  | 1539  |
| 4096  | 3884       | 4080  | 3460 | 5142  | 5141             | 6991  | 7301  |
| 2^14  | 22201      | 22289 |      | 29701 | 28816            | 39373 | 35956 |
| 2^16  | 94844      | 121906|      | 135852| 121238           | 196506| 154750|
| 2^18  | 484203     | 567453|      | 613544| 568220           | 988101| 845352|

On the older phones (Pixel 3a, Galaxy A50) TwoFFT and FFTW are within 10% of each other up to
N = 1024, TwoFFT is 10..30% faster above; pffft is 20..45% slower than TwoFFT everywhere.

## Tools

* `tools/android.sh`  build a C file with the NDK and run it on a phone via adb
* `tools/mac.sh`      copy the sources to a Mac, build and run there
* `bench/bench.c`     comparison with other libraries (pffft, FFTW, vDSP)
* `bench/bench_passes.c`  cycles per element of each pass (ARM64)
* `bench/uarch.c`     latency / throughput of the instructions used by the kernels (ARM64)
