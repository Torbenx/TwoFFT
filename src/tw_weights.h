/* TwoFFT - weight (twiddle factor) tables. */
#ifndef TW_WEIGHTS_H
#define TW_WEIGHTS_H

#include "tw_common.h"

/*
 * Pass structure (log2n >= 6)
 * ---------------------------
 * first pass   Splits the transform into 8 sub-FFTs of size N/8 with a
 *              radix-8 decimation-in-frequency step, weights the results and
 *              scatters them in bit-reversed order into lines. Depending on
 *              log2n mod 3 it also performs the first 0, 1 or 2 stages of the
 *              sub-FFTs, so that the number of remaining stages is a multiple
 *              of three:
 *                  TW_FIRST_A  log2n % 3 == 0   3 stages
 *                  TW_FIRST_B  log2n % 3 == 1   4 stages
 *                  TW_FIRST_C  log2n % 3 == 2   5 stages
 * radix-8      In-place decimation-in-time passes over the lines. All 8
 *              lanes of a line share the same weights, so these need a single
 *              set of 7 scalar weights per butterfly. The pass with stride 1
 *              (directly after TW_FIRST_A) has unit weights and no table.
 *
 * Weight table layout for one transform size:
 *     first pass  N/32 tiles of 14 vectors of 4 floats: for c = 1..7 the
 *                 real and imaginary parts of W_N^(c*k) for the 4 values of
 *                 k a tile processes (see tw_first_k), in the order the
 *                 tiles are visited (see tw_first_tile).
 *     radix-8     for each stride s = s0*8^i (s > 1, 8*s <= N/8), s sets of
 *                 16 floats: (re, im) of w^4, w^2, w^6, w^1, w^5, w^3, w^7
 *                 for w = W_(8s)^j, followed by sqrt(1/2) twice.
 */

enum { TW_FIRST_A = 0, TW_FIRST_B = 1, TW_FIRST_C = 2 };

#define TW_FIRST_TILE_FLOATS 56
#define TW_R8_SET_FLOATS 16

TW_INLINE int tw_first_variant(unsigned log2n) { return (int)(log2n % 3); }

/* Stride (in lines) of the first radix-8 pass after the first pass. */
TW_INLINE size_t tw_first_stride(unsigned log2n) { return (size_t)1 << (log2n % 3); }

/* Offset in floats of the weights of the radix-8 pass with stride s to the
 * start of the radix-8 weights. s0 is the stride of the first radix-8 pass. */
TW_INLINE size_t tw_r8_weights_offset(size_t s0, size_t s) {
    /* strides s0, 8*s0, ..., s/8 precede; the pass with stride 1 has no table */
    return TW_R8_SET_FLOATS * ((s - s0) / 7 - (s0 == 1 && s > 1 ? 1 : 0));
}

/* The index k of the sub-FFT input computed in `lane` of first pass tile `tile`. */
TW_INLINE size_t tw_first_k(unsigned log2n, size_t tile, unsigned lane) {
    const size_t m = tw_lines(log2n);
    switch (tw_first_variant(log2n)) {
        case TW_FIRST_A: return 4 * tile + lane;
        case TW_FIRST_B: return 2 * tile + (lane & 1) + (lane >> 1) * (m / 2);
        default: return tile + (lane & 1) * (m / 2) + (lane >> 1) * (m / 4);
    }
}

/*
 * Order of the first pass tiles
 * -----------------------------
 * Tile t writes its lines to position bitrev(t). Visiting the tiles in
 * natural order therefore scatters the output over the whole buffer, which
 * is slow as soon as the output does not fit into the L1 cache. Large
 * transforms (log2n >= TW_FIRST_BLOCKED_LOG2N) visit the tiles in chunks of
 * TW_FIRST_CHUNK_TILES = R * P tiles instead:
 *
 *     t = chunk * R + r + p * (tiles / P)      r < R, p < P
 *
 * ordered by (chunk, p, r): a chunk reads runs of R consecutive tiles at P
 * places of each input stream. Its output consists of contiguous runs of P
 * positions. More places give longer output runs but split the input into
 * more concurrently read streams. The weight table is stored in the order
 * the tiles are visited.
 *
 * The best P depends on the first pass variant and the processor. Apple
 * cores are slow at storing to scattered cache lines, all other cores we
 * measured prefer the input to be read sequentially (P = 1).
 */
#ifndef TW_FIRST_BLOCKED_LOG2N
#define TW_FIRST_BLOCKED_LOG2N 14
#endif
#ifndef TW_FIRST_LOG2_PLACES_A
#if defined(__APPLE__)
#define TW_FIRST_LOG2_PLACES_A 5
#define TW_FIRST_LOG2_PLACES_B 3
#define TW_FIRST_LOG2_PLACES_C 2
#else
#define TW_FIRST_LOG2_PLACES_A 0
#define TW_FIRST_LOG2_PLACES_B 0
#define TW_FIRST_LOG2_PLACES_C 0
#endif
#endif
#define TW_FIRST_CHUNK_TILES 64

TW_INLINE int tw_first_blocked(unsigned log2n) { return log2n >= TW_FIRST_BLOCKED_LOG2N; }

/* log2 of the number of places P of a chunk (blocked order only). */
TW_INLINE unsigned tw_first_log2_places(unsigned log2n) {
    switch (tw_first_variant(log2n)) {
        case TW_FIRST_A: return TW_FIRST_LOG2_PLACES_A;
        case TW_FIRST_B: return TW_FIRST_LOG2_PLACES_B;
        default: return TW_FIRST_LOG2_PLACES_C;
    }
}

/* The tile visited at position `index`. */
TW_INLINE size_t tw_first_tile(unsigned log2n, size_t index) {
    if (!tw_first_blocked(log2n)) return index;
    const unsigned log2_places = tw_first_log2_places(log2n);
    const size_t run = TW_FIRST_CHUNK_TILES >> log2_places;
    const size_t distance = (((size_t)1 << log2n) / 32) >> log2_places;
    const size_t r = index % run;
    const size_t p = (index % TW_FIRST_CHUNK_TILES) / run;
    const size_t chunk = index / TW_FIRST_CHUNK_TILES;
    return chunk * run + r + p * distance;
}

/* Number of floats in the weight table of a transform of size 2^log2n. */
TW_INLINE size_t tw_weights_floats(unsigned log2n) {
    if (log2n < TW_LOG2_GENERIC_MIN) return 0;
    const size_t m = tw_lines(log2n);
    size_t count = (m / 4) * TW_FIRST_TILE_FLOATS;
    for (size_t s = tw_first_stride(log2n); 8 * s <= m; s *= 8)
        if (s > 1) count += s * TW_R8_SET_FLOATS;
    return count;
}

/* sin and cos for x in [0, pi/4], accurate to double precision. Having our
 * own implementation makes the tables bit-identical on all platforms and
 * avoids a dependency on libm. */
static void tw_sincos(double x, double* s, double* c) {
    const double x2 = x * x;
    double sp = 1.0, cp = 1.0;
    for (int i = 19; i >= 3; i -= 2) {
        sp = 1.0 - sp * x2 / (double)(i * (i - 1));         /* sin: x^i / i! */
        cp = 1.0 - cp * x2 / (double)((i - 1) * (i - 2));   /* cos: x^(i-1) / (i-1)! */
    }
    *s = x * sp;
    *c = cp;
}

/* Computes exp(-2*pi*i * num / den) for a power of two den. Symmetries are
 * exact: the result is reduced to the first octant. */
static void tw_root(uint64_t num, uint64_t den, float* re, float* im) {
    static const double quarter_pi = 0.78539816339744830962;
    num = (num & (den - 1)) * 8;
    const unsigned octant = (unsigned)(num / den);
    uint64_t frac = num - octant * den;
    if (octant & 1) frac = den - frac;
    double s, c;
    tw_sincos(quarter_pi * ((double)frac / (double)den), &s, &c);
    double x, y; /* exp(+i*angle) */
    switch (octant) {
        case 0: x = c, y = s; break;
        case 1: x = s, y = c; break;
        case 2: x = -s, y = c; break;
        case 3: x = -c, y = s; break;
        case 4: x = -c, y = -s; break;
        case 5: x = -s, y = -c; break;
        case 6: x = s, y = -c; break;
        default: x = c, y = -s; break;
    }
    *re = (float)x;
    *im = (float)-y;
}

/* Fills the weight table of a transform of size 2^log2n. `w` must have room
 * for tw_weights_floats(log2n) floats. */
static void tw_weights_fill(unsigned log2n, float* w) {
    if (log2n < TW_LOG2_GENERIC_MIN) return;
    const size_t m = tw_lines(log2n);
    const uint64_t n = (uint64_t)1 << log2n;
    for (size_t index = 0; index < m / 4; index++) {
        const size_t tile = tw_first_tile(log2n, index);
        for (unsigned c = 1; c < 8; c++) {
            for (unsigned lane = 0; lane < 4; lane++)
                tw_root(c * (uint64_t)tw_first_k(log2n, tile, lane), n, &w[lane], &w[4 + lane]);
            w += 8;
        }
    }
    for (size_t s = tw_first_stride(log2n); 8 * s <= m; s *= 8) {
        if (s == 1) continue;
        for (size_t j = 0; j < s; j++) {
            for (unsigned pos = 1; pos < 8; pos++)
                tw_root(TW_E8(pos) * (uint64_t)j, 8 * (uint64_t)s, &w[2 * pos - 2], &w[2 * pos - 1]);
            w[14] = w[15] = TW_SQRT1_2;
            w += TW_R8_SET_FLOATS;
        }
    }
}

#endif /* TW_WEIGHTS_H */
