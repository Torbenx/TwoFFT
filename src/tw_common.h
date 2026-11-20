/* TwoFFT - internal definitions shared by all backends. */
#ifndef TW_COMMON_H
#define TW_COMMON_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define TW_INLINE static inline __attribute__((always_inline))
#define TW_NOINLINE __attribute__((noinline))
#else
#error "TwoFFT requires GCC or Clang (inline assembly in GNU syntax)"
#endif

/*
 * Working layout
 * --------------
 * After the first pass the output buffer holds N/8 "lines" of 64 bytes. A
 * line holds the same coefficient of 8 independent sub-FFTs c = 0..7 as
 * split real / imaginary parts in two halves of 4 lanes:
 *
 *     [re c0..c3] [im c0..c3] [re c4..c7] [im c4..c7]
 *
 * The last pass rewrites every line in place as 8 interleaved complex
 * numbers, which is the final output order.
 */
#define TW_LINE_FLOATS 16
#define TW_LINE_BYTES 64

/* Position m of a radix-8 butterfly is weighted with w^TW_E8(m). */
#define TW_E8(m) ((unsigned)(0x73516240u >> (4 * (m))) & 7u)

#define TW_SQRT1_2 0.70710678118654752440f

/* Smallest transform size handled by the generic pass structure. */
#define TW_LOG2_GENERIC_MIN 6

/* Number of lines of a transform of size 2^log2n. */
TW_INLINE size_t tw_lines(unsigned log2n) { return (size_t)1 << (log2n - 3); }

/* Reverses the low `bits` bits of x. */
TW_INLINE size_t tw_bitrev(size_t x, unsigned bits) {
    size_t r = 0;
    for (unsigned i = 0; i < bits; i++) r |= ((x >> i) & 1) << (bits - 1 - i);
    return r;
}

#endif /* TW_COMMON_H */
