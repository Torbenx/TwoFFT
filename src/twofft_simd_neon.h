/* TwoFFT vector abstraction: ARM64 NEON (128 bit, 4 float lanes).
 *
 * Every twofft_simd_*.h header defines the same set of macros; twofft_kernels.h is
 * written purely in terms of them and is instantiated once per instruction set.
 */

#define TWOFFT_ISA neon
#define TWOFFT_TARGET /* NEON is part of the arm64 baseline */
#define TWOFFT_VW 4 /* float lanes per vector */
#define TWOFFT_VW_LOG2 2

#define vf float32x4_t

#define v_load(p) vld1q_f32(p)
#define v_store(p, v) vst1q_f32((p), (v))
#define v_set1(x) vdupq_n_f32(x)
#define v_add(a, b) vaddq_f32((a), (b))
#define v_sub(a, b) vsubq_f32((a), (b))
#define v_mul(a, b) vmulq_f32((a), (b))
#define v_neg(a) vnegq_f32(a)
/* a + b * c and a - b * c */
#define v_fmadd(a, b, c) vfmaq_f32((a), (b), (c))
#define v_fnmadd(a, b, c) vfmsq_f32((a), (b), (c))

/* Load / store TWOFFT_VW interleaved complex numbers as separate real / imaginary vectors.
 * Lane c of the loaded vectors holds element TWOFFT_LANE_ELEM(c). */
#define V_LOAD2(p, re, im)                 \
    do {                                   \
        float32x4x2_t ld2_ = vld2q_f32(p); \
        (re) = ld2_.val[0];                \
        (im) = ld2_.val[1];                \
    } while (0)
#define V_STORE2(p, re, im)     \
    do {                        \
        float32x4x2_t st2_;     \
        st2_.val[0] = (re);     \
        st2_.val[1] = (im);     \
        vst2q_f32((p), st2_);   \
    } while (0)
#define TWOFFT_LANE_ELEM(c) (c)

/* In-place transpose of the TWOFFT_VW x TWOFFT_VW matrix held in v[0 .. TWOFFT_VW-1]. */
#define V_TRANSPOSE(v)                                                                  \
    do {                                                                                \
        float32x4_t t0_ = vtrn1q_f32((v)[0], (v)[1]), t1_ = vtrn2q_f32((v)[0], (v)[1]); \
        float32x4_t t2_ = vtrn1q_f32((v)[2], (v)[3]), t3_ = vtrn2q_f32((v)[2], (v)[3]); \
        (v)[0] = v4_lo64(t0_, t2_);                                                     \
        (v)[1] = v4_lo64(t1_, t3_);                                                     \
        (v)[2] = v4_hi64(t0_, t2_);                                                     \
        (v)[3] = v4_hi64(t1_, t3_);                                                     \
    } while (0)

/* 4-lane helpers, only used by the tiny fixed-size transforms. */
#define v4_lo64(a, b) /* [a0 a1 b0 b1] */ \
    vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)))
#define v4_hi64(a, b) /* [a2 a3 b2 b3] */ \
    vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)))
#define v4_lohi(a, b) /* [a0 a1 b2 b3] */ vcombine_f32(vget_low_f32(a), vget_high_f32(b))
#define v4_even(a, b) /* [a0 a2 b0 b2] */ vuzp1q_f32((a), (b))
#define v4_odd(a, b) /* [a1 a3 b1 b3] */ vuzp2q_f32((a), (b))
