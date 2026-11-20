/* TwoFFT - portable C implementation.
 *
 * Implements exactly the same pass structure, working layout and weight
 * tables as the SIMD backends. It is the reference the assembly kernels are
 * tested against and the fallback for other architectures. */
#ifndef TW_SCALAR_H
#define TW_SCALAR_H

#include "tw_common.h"
#include "tw_weights.h"

typedef struct tw_cf {
    float re, im;
} tw_cf;

TW_INLINE tw_cf tw_cadd(tw_cf a, tw_cf b) { return (tw_cf){a.re + b.re, a.im + b.im}; }
TW_INLINE tw_cf tw_csub(tw_cf a, tw_cf b) { return (tw_cf){a.re - b.re, a.im - b.im}; }
TW_INLINE tw_cf tw_cmul(tw_cf a, tw_cf w) {
    return (tw_cf){a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
}
/* a - i*b and a + i*b */
TW_INLINE tw_cf tw_csubj(tw_cf a, tw_cf b) { return (tw_cf){a.re + b.im, a.im - b.re}; }
TW_INLINE tw_cf tw_caddj(tw_cf a, tw_cf b) { return (tw_cf){a.re - b.im, a.im + b.re}; }

/* 8 point DFT. Input in bit-reversed (decimation-in-time) order, output in
 * natural order. All radix-8 butterflies of the library are this function
 * applied to weighted inputs. */
TW_INLINE void tw_dft8(tw_cf* v) {
    const tw_cf a0 = tw_cadd(v[0], v[1]), a1 = tw_csub(v[0], v[1]);
    const tw_cf a2 = tw_cadd(v[2], v[3]), a3 = tw_csub(v[2], v[3]);
    const tw_cf a4 = tw_cadd(v[4], v[5]), a5 = tw_csub(v[4], v[5]);
    const tw_cf a6 = tw_cadd(v[6], v[7]), a7 = tw_csub(v[6], v[7]);

    const tw_cf b0 = tw_cadd(a0, a2), b2 = tw_csub(a0, a2);
    const tw_cf b1 = tw_csubj(a1, a3), b3 = tw_caddj(a1, a3);
    const tw_cf b4 = tw_cadd(a4, a6), b6 = tw_csub(a4, a6);
    const tw_cf b5 = tw_csubj(a5, a7), b7 = tw_caddj(a5, a7);

    /* W8 * b5 = h * (s5 + i*d5) and W8^3 * b7 = h * (d7 - i*s7) */
    const float s5 = (b5.re + b5.im) * TW_SQRT1_2, d5 = (b5.im - b5.re) * TW_SQRT1_2;
    const float s7 = (b7.re + b7.im) * TW_SQRT1_2, d7 = (b7.im - b7.re) * TW_SQRT1_2;

    v[0] = tw_cadd(b0, b4);
    v[4] = tw_csub(b0, b4);
    v[1] = (tw_cf){b1.re + s5, b1.im + d5};
    v[5] = (tw_cf){b1.re - s5, b1.im - d5};
    v[2] = tw_csubj(b2, b6);
    v[6] = tw_caddj(b2, b6);
    v[3] = (tw_cf){b3.re + d7, b3.im - s7};
    v[7] = (tw_cf){b3.re - d7, b3.im + s7};
}

/* Lane c of a line: real part at TW_LANE(c), imaginary part 4 floats later. */
#define TW_LANE(c) ((c) + ((c) & 4))

TW_INLINE tw_cf tw_line_get(const float* line, unsigned c) {
    return (tw_cf){line[TW_LANE(c)], line[TW_LANE(c) + 4]};
}
TW_INLINE void tw_line_set(float* line, unsigned c, tw_cf v) {
    line[TW_LANE(c)] = v.re;
    line[TW_LANE(c) + 4] = v.im;
}

/* First pass, see tw_weights.h. `swap` exchanges real and imaginary parts of
 * the input, which together with the same exchange on the output turns the
 * forward into the backward transform. */
static void tw_scalar_first(unsigned log2n, const float* w, const float* in, float* out, int swap) {
    const size_t m = tw_lines(log2n);
    const int variant = tw_first_variant(log2n);
    for (size_t index = 0; index < m / 4; index++, w += TW_FIRST_TILE_FLOATS) {
        const size_t tile = tw_first_tile(log2n, index);
        tw_cf y[4][8]; /* [lane][c] */
        size_t k[4];
        for (unsigned lane = 0; lane < 4; lane++) {
            k[lane] = tw_first_k(log2n, tile, lane);
            tw_cf* v = y[lane];
            for (unsigned pos = 0; pos < 8; pos++) {
                const float* x = in + 2 * (k[lane] + TW_E8(pos) * m);
                v[pos] = swap ? (tw_cf){x[1], x[0]} : (tw_cf){x[0], x[1]};
            }
            tw_dft8(v);
            for (unsigned c = 1; c < 8; c++)
                v[c] = tw_cmul(v[c], (tw_cf){w[8 * (c - 1) + lane], w[8 * (c - 1) + 4 + lane]});
        }
        const size_t pos = tw_bitrev(k[0], log2n - 3);
        for (unsigned c = 0; c < 8; c++) {
            if (variant == TW_FIRST_A) {
                for (unsigned lane = 0; lane < 4; lane++)
                    tw_line_set(out + TW_LINE_FLOATS * tw_bitrev(k[lane], log2n - 3), c, y[lane][c]);
            } else if (variant == TW_FIRST_B) {
                float* lo = out + TW_LINE_FLOATS * pos;
                float* hi = lo + TW_LINE_FLOATS * (m / 2);
                tw_line_set(lo, c, tw_cadd(y[0][c], y[2][c]));
                tw_line_set(lo + TW_LINE_FLOATS, c, tw_csub(y[0][c], y[2][c]));
                tw_line_set(hi, c, tw_cadd(y[1][c], y[3][c]));
                tw_line_set(hi + TW_LINE_FLOATS, c, tw_csub(y[1][c], y[3][c]));
            } else {
                float* o = out + TW_LINE_FLOATS * pos;
                const tw_cf a0 = tw_cadd(y[0][c], y[1][c]), a1 = tw_csub(y[0][c], y[1][c]);
                const tw_cf a2 = tw_cadd(y[2][c], y[3][c]), a3 = tw_csub(y[2][c], y[3][c]);
                tw_line_set(o, c, tw_cadd(a0, a2));
                tw_line_set(o + TW_LINE_FLOATS, c, tw_csubj(a1, a3));
                tw_line_set(o + 2 * TW_LINE_FLOATS, c, tw_csub(a0, a2));
                tw_line_set(o + 3 * TW_LINE_FLOATS, c, tw_caddj(a1, a3));
            }
        }
    }
}

/* Rewrites a line as 8 interleaved complex numbers. */
TW_INLINE void tw_scalar_interleave(float* line, int swap) {
    float t[TW_LINE_FLOATS];
    for (unsigned i = 0; i < TW_LINE_FLOATS; i++) t[i] = line[i];
    for (unsigned c = 0; c < 8; c++) {
        line[2 * c + (swap ? 1 : 0)] = t[TW_LANE(c)];
        line[2 * c + (swap ? 0 : 1)] = t[TW_LANE(c) + 4];
    }
}

/* Radix-8 pass with stride `s` lines over `blocks` blocks of 8*s lines.
 * `w` is the weight table of the pass or NULL for unit weights (s == 1).
 * With `last` set lines are converted to the interleaved output format. */
static void tw_scalar_r8(float* data, const float* w, size_t s, size_t blocks, int last, int swap) {
    for (size_t b = 0; b < blocks; b++, data += 8 * s * TW_LINE_FLOATS) {
        for (size_t j = 0; j < s; j++) {
            const float* wj = w ? w + j * TW_R8_SET_FLOATS : NULL;
            float* line[8];
            for (unsigned pos = 0; pos < 8; pos++) line[pos] = data + (j + pos * s) * TW_LINE_FLOATS;
            for (unsigned c = 0; c < 8; c++) {
                tw_cf v[8];
                v[0] = tw_line_get(line[0], c);
                for (unsigned pos = 1; pos < 8; pos++) {
                    v[pos] = tw_line_get(line[pos], c);
                    if (wj) v[pos] = tw_cmul(v[pos], (tw_cf){wj[2 * pos - 2], wj[2 * pos - 1]});
                }
                tw_dft8(v);
                for (unsigned pos = 0; pos < 8; pos++) tw_line_set(line[pos], c, v[pos]);
            }
            if (last)
                for (unsigned pos = 0; pos < 8; pos++) tw_scalar_interleave(line[pos], swap);
        }
    }
}

/* Transforms of size 1 to 32 by direct evaluation of small DFTs. */
static void tw_scalar_small(unsigned log2n, const float* in, float* out, int swap) {
    const size_t n = (size_t)1 << log2n;
    tw_cf v[32], t[32];
    for (size_t i = 0; i < n; i++) v[i] = swap ? (tw_cf){in[2 * i + 1], in[2 * i]} : (tw_cf){in[2 * i], in[2 * i + 1]};
    /* decimation in time on bit-reversed input, radix 2 */
    for (size_t i = 0; i < n; i++) t[tw_bitrev(i, log2n)] = v[i];
    for (size_t h = 1; h < n; h *= 2) {
        for (size_t b = 0; b < n; b += 2 * h) {
            for (size_t j = 0; j < h; j++) {
                tw_cf wj;
                tw_root(j, 2 * h, &wj.re, &wj.im);
                const tw_cf a = t[b + j], c = tw_cmul(t[b + j + h], wj);
                t[b + j] = tw_cadd(a, c);
                t[b + j + h] = tw_csub(a, c);
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        out[2 * i + (swap ? 1 : 0)] = t[i].re;
        out[2 * i + (swap ? 0 : 1)] = t[i].im;
    }
}

/* Out-of-place transform of size 2^log2n with the weights from tw_weights_fill. */
static void tw_scalar_fft(unsigned log2n, const float* w, const float* in, float* out, int backward) {
    if (log2n < TW_LOG2_GENERIC_MIN) {
        tw_scalar_small(log2n, in, out, backward);
        return;
    }
    const size_t m = tw_lines(log2n);
    tw_scalar_first(log2n, w, in, out, backward);
    w += (m / 4) * TW_FIRST_TILE_FLOATS;
    for (size_t s = tw_first_stride(log2n); 8 * s <= m; s *= 8) {
        tw_scalar_r8(out, s > 1 ? w : NULL, s, m / (8 * s), 8 * s == m, backward);
        if (s > 1) w += s * TW_R8_SET_FLOATS;
    }
}

#endif /* TW_SCALAR_H */
