/* TwoFFT - fast power-of-two single precision FFT. */

#include "twofft.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#define TWOFFT_ARCH_ARM64 1
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#define TWOFFT_ARCH_X86_64 1
#include <immintrin.h>
#else
#error "TwoFFT supports arm64 and x86-64 only"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define TWOFFT_FORCE_INLINE static __forceinline
#define restrict __restrict
#else
#define TWOFFT_FORCE_INLINE static inline __attribute__((always_inline))
#endif

#define TWOFFT_SQRT1_2 0.70710678118654752440f
#define TWOFFT_COS_PI_8 0.92387953251128675613f
#define TWOFFT_SIN_PI_8 0.38268343236508977173f

/* Blocks of up to 2^TWOFFT_BLOCK_LOG2 points are transformed pass by pass, larger ones
 * depth first. */
#ifndef TWOFFT_BLOCK_LOG2
#define TWOFFT_BLOCK_LOG2 12
#endif

#define TWOFFT_MIN_PLAN_LOG2 5 /* smaller sizes use fixed size transforms */
#define TWOFFT_MAX_PASSES 14

/* How a size 2^log2n transform is decomposed. Pass j combines sub-transforms of
 * 2^(size_log[j] - radix_log[j]) points into transforms of 2^size_log[j] points. */
typedef struct twofft_plan {
    uint8_t log2n;
    uint8_t leaf_log; /* 3 or 4 */
    uint8_t npasses;  /* >= 1; the last pass is the final pass */
    uint8_t radix_log[TWOFFT_MAX_PASSES];
    uint8_t size_log[TWOFFT_MAX_PASSES];
    const float *tw[TWOFFT_MAX_PASSES];
} twofft_plan;

/* Reverses the lowest `bits` bits of v. */
static inline size_t twofft_bitrev(size_t v, unsigned bits)
{
    size_t r = 0;
    unsigned b;
    for (b = 0; b < bits; b++)
        r |= ((v >> b) & 1) << (bits - 1 - b);
    return r;
}

/* Given rev = bitrev(m) for a counter m < count (a power of two), returns bitrev(m + 1). */
static inline size_t twofft_bitrev_next(size_t rev, size_t count)
{
    size_t bit = count >> 1;
    while (rev & bit) {
        rev ^= bit;
        bit >>= 1;
    }
    return rev | bit;
}

/* exp(-2 pi i n q / N) for the fixed size transforms; [0]: real parts, [1]: imaginary parts */
static const float twofft_w8[2][4] = {
    {1.0f, TWOFFT_SQRT1_2, 0.0f, -TWOFFT_SQRT1_2},
    {0.0f, -TWOFFT_SQRT1_2, -1.0f, -TWOFFT_SQRT1_2},
};
static const float twofft_w16[3][2][4] = {
    {{1.0f, TWOFFT_COS_PI_8, TWOFFT_SQRT1_2, TWOFFT_SIN_PI_8},
     {0.0f, -TWOFFT_SIN_PI_8, -TWOFFT_SQRT1_2, -TWOFFT_COS_PI_8}},
    {{1.0f, TWOFFT_SQRT1_2, 0.0f, -TWOFFT_SQRT1_2}, {0.0f, -TWOFFT_SQRT1_2, -1.0f, -TWOFFT_SQRT1_2}},
    {{1.0f, TWOFFT_SIN_PI_8, -TWOFFT_SQRT1_2, -TWOFFT_COS_PI_8},
     {0.0f, -TWOFFT_COS_PI_8, -TWOFFT_SQRT1_2, TWOFFT_SIN_PI_8}},
};

/* ---------------------------------------------------------------------------------------
 * Kernels, instantiated once per instruction set.
 * ------------------------------------------------------------------------------------ */

#if TWOFFT_ARCH_ARM64
#include "twofft_simd_neon.h"
#include "twofft_kernels.h"
#include "twofft_simd_undef.h"
#define TWOFFT_VW_NATIVE 4
#endif

#if TWOFFT_ARCH_X86_64
#include "twofft_simd_sse2.h"
#include "twofft_kernels.h"
#include "twofft_simd_undef.h"
#define TWOFFT_VW_NATIVE 4
#endif

/* ---------------------------------------------------------------------------------------
 * Tables and plans.
 * ------------------------------------------------------------------------------------ */

static void *twofft_alloc(size_t bytes)
{
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&p, 64, bytes) != 0)
        p = NULL;
#endif
    return p;
}

static void twofft_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Twiddle table of a radix 2^radix_log pass with sub-transform size 2^sub_log for vectors
 * of vw lanes: for every vector position the powers w^1 .. w^(radix-1), each stored as vw
 * real parts followed by vw imaginary parts. */
static float *twofft_make_twiddles(unsigned radix_log, unsigned sub_log, unsigned vw)
{
    const size_t radix = (size_t)1 << radix_log, S = (size_t)1 << sub_log;
    const double step = -6.283185307179586476925286766559 / (double)(radix * S);
    float *tab = (float *)twofft_alloc(2 * (radix - 1) * S * sizeof(float));
    float *t = tab;
    size_t p, j, l;
    if (!tab)
        return NULL;
    for (p = 0; p < S; p += vw) {
        for (j = 1; j < radix; j++) {
            for (l = 0; l < vw; l++) {
                const double a = step * (double)(j * (p + l));
                t[l] = (float)cos(a);
                t[vw + l] = (float)sin(a);
            }
            t += 2 * vw;
        }
    }
    return tab;
}

struct twofft_context {
    unsigned vw;
    float *twiddles[2][TWOFFT_MAX_LOG2_SIZE + 1]; /* [radix_log - 2][sub_log] */
    twofft_plan *plans[TWOFFT_MAX_LOG2_SIZE + 1];
};

static const float *twofft_get_twiddles(twofft_context *ctx, unsigned radix_log, unsigned sub_log)
{
    float **slot = &ctx->twiddles[radix_log - 2][sub_log];
    float *tab = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (!tab) {
        float *expected = NULL;
        tab = twofft_make_twiddles(radix_log, sub_log, ctx->vw);
        if (tab && !__atomic_compare_exchange_n(slot, &expected, tab, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            twofft_free(tab);
            tab = expected;
        }
    }
    return tab;
}

/* Chooses the decomposition of a size 2^n transform. */
static void twofft_decompose(twofft_plan *pl, unsigned n)
{
    unsigned rest, k = 0, size;
    pl->log2n = (uint8_t)n;
    /* 16 point leaves unless that leaves a single level */
    pl->leaf_log = (uint8_t)(n == 5 ? 3 : 4);
    rest = n - pl->leaf_log;
    size = pl->leaf_log;
    /* radix 8 passes where possible, radix 4 passes first */
    while (rest > 0) {
        const unsigned rl = (rest % 3 == 0) ? 3 : 2;
        size += rl;
        pl->radix_log[k] = (uint8_t)rl;
        pl->size_log[k] = (uint8_t)size;
        rest -= rl;
        k++;
    }
    pl->npasses = (uint8_t)k;
}

static const twofft_plan *twofft_get_plan(twofft_context *ctx, unsigned n)
{
    twofft_plan **slot = &ctx->plans[n];
    twofft_plan *pl = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (!pl) {
        twofft_plan *expected = NULL;
        unsigned k;
        pl = (twofft_plan *)malloc(sizeof(twofft_plan));
        if (!pl)
            return NULL;
        twofft_decompose(pl, n);
        for (k = 0; k < pl->npasses; k++) {
            pl->tw[k] = twofft_get_twiddles(ctx, pl->radix_log[k], pl->size_log[k] - pl->radix_log[k]);
            if (!pl->tw[k]) {
                free(pl);
                return NULL;
            }
        }
        if (!__atomic_compare_exchange_n(slot, &expected, pl, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            free(pl);
            pl = expected;
        }
    }
    return pl;
}

twofft_context *twofft_context_create(void)
{
    twofft_context *ctx = (twofft_context *)calloc(1, sizeof(twofft_context));
    if (ctx)
        ctx->vw = TWOFFT_VW_NATIVE;
    return ctx;
}

void twofft_context_destroy(twofft_context *ctx)
{
    unsigned i, j;
    if (!ctx)
        return;
    for (i = 0; i <= TWOFFT_MAX_LOG2_SIZE; i++) {
        for (j = 0; j < 2; j++)
            twofft_free(ctx->twiddles[j][i]);
        free(ctx->plans[i]);
    }
    free(ctx);
}

/* ---------------------------------------------------------------------------------------
 * Transforms.
 * ------------------------------------------------------------------------------------ */

static void twofft_tiny(const float *restrict in, float *restrict out, unsigned n, int inv)
{
    if (n == 0) {
        out[0] = in[0], out[1] = in[1];
    } else if (n == 1) {
        const float ar = in[0], ai = in[1], br = in[2], bi = in[3];
        out[0] = ar + br, out[1] = ai + bi;
        out[2] = ar - br, out[3] = ai - bi;
    } else {
        const float s0r = in[0] + in[4], s0i = in[1] + in[5], d0r = in[0] - in[4], d0i = in[1] - in[5];
        const float s1r = in[2] + in[6], s1i = in[3] + in[7], d1r = in[2] - in[6], d1i = in[3] - in[7];
        const float er = inv ? -d1i : d1i, ei = inv ? d1r : -d1r; /* -+ i * d1 */
        out[0] = s0r + s1r, out[1] = s0i + s1i;
        out[2] = d0r + er, out[3] = d0i + ei;
        out[4] = s0r - s1r, out[5] = s0i - s1i;
        out[6] = d0r - er, out[7] = d0i - ei;
    }
}

static int twofft_execute(twofft_context *ctx, const float *restrict in, float *restrict out, unsigned n,
                          int inv)
{
    if (n > TWOFFT_MAX_LOG2_SIZE)
        return -1;
    if (n >= TWOFFT_MIN_PLAN_LOG2) {
        const twofft_plan *pl = twofft_get_plan(ctx, n);
        if (!pl)
            return -1;
#if TWOFFT_ARCH_ARM64
        twofft_neon_transform(pl, in, out, inv);
#else
        twofft_sse2_transform(pl, in, out, inv);
#endif
        return 0;
    }
    if (n <= 2) {
        twofft_tiny(in, out, n, inv);
    } else if (n == 3) {
#if TWOFFT_ARCH_ARM64
        if (inv) twofft_neon_fft8_inv(in, out); else twofft_neon_fft8_fwd(in, out);
#else
        if (inv) twofft_sse2_fft8_inv(in, out); else twofft_sse2_fft8_fwd(in, out);
#endif
    } else {
#if TWOFFT_ARCH_ARM64
        if (inv) twofft_neon_fft16_inv(in, out); else twofft_neon_fft16_fwd(in, out);
#else
        if (inv) twofft_sse2_fft16_inv(in, out); else twofft_sse2_fft16_fwd(in, out);
#endif
    }
    return 0;
}

int twofft_forward(twofft_context *ctx, const float *in, float *out, unsigned log2_size)
{
    return twofft_execute(ctx, in, out, log2_size, 0);
}

int twofft_inverse(twofft_context *ctx, const float *in, float *out, unsigned log2_size)
{
    return twofft_execute(ctx, in, out, log2_size, 1);
}
