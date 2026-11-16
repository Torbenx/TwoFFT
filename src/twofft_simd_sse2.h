/* TwoFFT vector abstraction: x86-64 SSE2 (128 bit, 4 float lanes, no FMA).
 * See twofft_simd_neon.h for the description of the macro set. */

#define TWOFFT_ISA sse2
#define TWOFFT_TARGET /* SSE2 is part of the x86-64 baseline */
#define TWOFFT_VW 4
#define TWOFFT_VW_LOG2 2

#define vf __m128

#define v_load(p) _mm_loadu_ps(p)
#define v_store(p, v) _mm_storeu_ps((p), (v))
#define v_set1(x) _mm_set1_ps(x)
#define v_add(a, b) _mm_add_ps((a), (b))
#define v_sub(a, b) _mm_sub_ps((a), (b))
#define v_mul(a, b) _mm_mul_ps((a), (b))
#define v_neg(a) _mm_xor_ps((a), _mm_set1_ps(-0.0f))
#define v_fmadd(a, b, c) _mm_add_ps((a), _mm_mul_ps((b), (c)))
#define v_fnmadd(a, b, c) _mm_sub_ps((a), _mm_mul_ps((b), (c)))

#define V_LOAD2(p, re, im)                                            \
    do {                                                              \
        __m128 ld0_ = _mm_loadu_ps(p), ld1_ = _mm_loadu_ps((p) + 4);  \
        (re) = _mm_shuffle_ps(ld0_, ld1_, _MM_SHUFFLE(2, 0, 2, 0));   \
        (im) = _mm_shuffle_ps(ld0_, ld1_, _MM_SHUFFLE(3, 1, 3, 1));   \
    } while (0)
#define V_STORE2(p, re, im)                                 \
    do {                                                    \
        __m128 st0_ = (re), st1_ = (im);                    \
        _mm_storeu_ps((p), _mm_unpacklo_ps(st0_, st1_));    \
        _mm_storeu_ps((p) + 4, _mm_unpackhi_ps(st0_, st1_)); \
    } while (0)
#define TWOFFT_LANE_ELEM(c) (c)

#define V_TRANSPOSE(v)                                                                      \
    do {                                                                                    \
        __m128 t0_ = _mm_unpacklo_ps((v)[0], (v)[1]), t1_ = _mm_unpackhi_ps((v)[0], (v)[1]); \
        __m128 t2_ = _mm_unpacklo_ps((v)[2], (v)[3]), t3_ = _mm_unpackhi_ps((v)[2], (v)[3]); \
        (v)[0] = _mm_movelh_ps(t0_, t2_);                                                   \
        (v)[1] = _mm_movehl_ps(t2_, t0_);                                                   \
        (v)[2] = _mm_movelh_ps(t1_, t3_);                                                   \
        (v)[3] = _mm_movehl_ps(t3_, t1_);                                                   \
    } while (0)

#define v4_lo64(a, b) _mm_movelh_ps((a), (b))
#define v4_hi64(a, b) _mm_movehl_ps((b), (a))
#define v4_lohi(a, b) _mm_shuffle_ps((a), (b), _MM_SHUFFLE(3, 2, 1, 0))
#define v4_even(a, b) _mm_shuffle_ps((a), (b), _MM_SHUFFLE(2, 0, 2, 0))
#define v4_odd(a, b) _mm_shuffle_ps((a), (b), _MM_SHUFFLE(3, 1, 3, 1))
