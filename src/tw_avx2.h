/* TwoFFT - x86-64 AVX2 + FMA kernels.
 *
 * Pass structure (log2n >= 6)
 * ---------------------------
 * As on ARM64 (see tw_weights.h) the first pass splits the transform into 8
 * sub-FFTs of size N/8 that live in the 8 lanes of 64 byte lines, and all
 * further passes are in-place passes over the lines with scalar weights. The
 * differences are dictated by having 16 registers of 8 floats:
 *
 * line        [re x 8] [im x 8], one ymm register each. Lane l holds sub-FFT
 *             (0, 2, 4, 6, 1, 3, 5, 7)[l]: the two 128 bit halves hold the
 *             even and the odd sub-FFTs.
 * first pass  A tile is 64 complex numbers x[k + p*N/8], p = 0..7, for 8
 *             values of k (one per lane). It is processed in two halves of 4
 *             complex vectors: the sums x_p + x_(p+4) yield the even
 *             sub-FFTs, the differences the odd ones. Each half ends with a
 *             transposition within the 128 bit halves of the registers,
 *             after which a register holds 4 lanes of two different lines,
 *             which are stored separately.
 *                 TW_AVX2_FIRST_A  log2n odd    3 stages
 *                 TW_AVX2_FIRST_B  log2n even   4 stages (first stage of
 *                                               the sub-FFTs included)
 * radix-4     In-place decimation-in-time passes. A butterfly combines 4
 *             lines with 3 scalar weights. The pass with stride 1 (directly
 *             after TW_AVX2_FIRST_A) has unit weights and no table.
 *
 * Weight table layout for one transform size:
 *     first pass  N/64 tiles of 14 vectors of 8 floats: real and imaginary
 *                 parts of W_N^(c*k) for c = 2, 4, 6, 1, 3, 5, 7 and the 8
 *                 values of k of the tile (see tw_avx2_first_k).
 *     radix-4     for each stride s = s0*4^i (s > 1, 4*s <= N/8), s sets of
 *                 8 floats: (re, im) of w^2, w, w^3 for w = W_(4s)^j, 0, 0.
 *     positions   N/64 uint32: line index of the output of each tile.
 */
#ifndef TW_AVX2_H
#define TW_AVX2_H

#include <string.h>

#include "tw_common.h"
#include "tw_small.h"
#include "tw_weights.h"

#define TW_AVX2 __attribute__((target("avx2,fma")))

enum { TW_AVX2_FIRST_A = 0, TW_AVX2_FIRST_B = 1 };

#define TW_AVX2_TILE_FLOATS 112
#define TW_AVX2_R4_SET_FLOATS 8

TW_INLINE int tw_avx2_first_variant(unsigned log2n) { return (log2n & 1) ? TW_AVX2_FIRST_A : TW_AVX2_FIRST_B; }

/* Stride (in lines) of the first radix-4 pass after the first pass. */
TW_INLINE size_t tw_avx2_first_stride(unsigned log2n) { return (log2n & 1) ? 1 : 2; }

/* The index k of the sub-FFT input computed in `lane` of first pass tile `tile`. */
TW_INLINE size_t tw_avx2_first_k(unsigned log2n, size_t tile, unsigned lane) {
    /* Deinterleaving two vectors of 4 complex numbers yields this lane order. */
    static const unsigned char order[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    if (tw_avx2_first_variant(log2n) == TW_AVX2_FIRST_A) return 8 * tile + order[lane];
    /* B: the second vector starts half the sub-FFT size later. */
    return 4 * tile + (order[lane] & 3) + (order[lane] >> 2) * (tw_lines(log2n) / 2);
}

/* Offset in floats of the weights of the radix-4 pass with stride s to the
 * start of the radix-4 weights. s0 is the stride of the first radix-4 pass. */
TW_INLINE size_t tw_avx2_r4_weights_offset(size_t s0, size_t s) {
    /* strides s0, 4*s0, ..., s/4 precede; the pass with stride 1 has no table */
    return TW_AVX2_R4_SET_FLOATS * ((s - s0) / 3 - (s0 == 1 && s > 1 ? 1 : 0));
}

/* Number of floats of the first pass and radix-4 weights. */
TW_INLINE size_t tw_avx2_weights_floats(unsigned log2n) {
    const size_t m = tw_lines(log2n);
    size_t count = (m / 8) * TW_AVX2_TILE_FLOATS;
    for (size_t s = tw_avx2_first_stride(log2n); 4 * s <= m; s *= 4)
        if (s > 1) count += s * TW_AVX2_R4_SET_FLOATS;
    return count;
}

/* Size in bytes of the weight table of a transform of size 2^log2n. */
TW_INLINE size_t tw_avx2_weights_bytes(unsigned log2n) {
    if (log2n < TW_LOG2_GENERIC_MIN) return 0;
    return tw_avx2_weights_floats(log2n) * sizeof(float) + (tw_lines(log2n) / 8) * sizeof(uint32_t);
}

TW_INLINE const uint32_t* tw_avx2_positions(unsigned log2n, const float* w) {
    return (const uint32_t*)(const void*)(w + tw_avx2_weights_floats(log2n));
}

/* Fills the weight table. `table` must be aligned to 4 bytes and have room
 * for tw_avx2_weights_bytes(log2n) bytes. */
static void tw_avx2_weights_fill(unsigned log2n, void* table) {
    if (log2n < TW_LOG2_GENERIC_MIN) return;
    static const unsigned char power[7] = {2, 4, 6, 1, 3, 5, 7};
    const size_t m = tw_lines(log2n);
    const uint64_t n = (uint64_t)1 << log2n;
    float* w = (float*)table;
    uint32_t* position = (uint32_t*)(void*)(w + tw_avx2_weights_floats(log2n));
    for (size_t tile = 0; tile < m / 8; tile++) {
        for (unsigned i = 0; i < 7; i++, w += 16)
            for (unsigned lane = 0; lane < 8; lane++)
                tw_root(power[i] * (uint64_t)tw_avx2_first_k(log2n, tile, lane), n, &w[lane], &w[8 + lane]);
        /* A: the 8 lines go to the same position of the 8 eighths of the
         * output. B: 4 pairs of lines go to the 4 quarters. */
        const uint32_t u = (uint32_t)tw_bitrev(tile, log2n - 6);
        position[tile] = tw_avx2_first_variant(log2n) == TW_AVX2_FIRST_A ? u : 2 * u;
    }
    for (size_t s = tw_avx2_first_stride(log2n); 4 * s <= m; s *= 4) {
        if (s == 1) continue;
        for (size_t j = 0; j < s; j++, w += TW_AVX2_R4_SET_FLOATS) {
            tw_root(2 * (uint64_t)j, 4 * (uint64_t)s, &w[0], &w[1]);
            tw_root(j, 4 * (uint64_t)s, &w[2], &w[3]);
            tw_root(3 * (uint64_t)j, 4 * (uint64_t)s, &w[4], &w[5]);
            w[6] = w[7] = 0.0f;
        }
    }
}

/* ---- instructions ------------------------------------------------------ */
/* Register arguments are ymm register numbers as string literals, memory
 * arguments are AT&T memory operands. The destination comes first. */

#define AVX_R(r) "%%ymm" r
#define AVX_OP3(op, d, a, b) op " %%ymm" b ", %%ymm" a ", %%ymm" d "\n"
#define AVX_OP3M(op, d, a, mem) op " " mem ", %%ymm" a ", %%ymm" d "\n"
#define AVX_ADD(d, a, b) AVX_OP3("vaddps", d, a, b)        /* d = a + b */
#define AVX_SUB(d, a, b) AVX_OP3("vsubps", d, a, b)        /* d = a - b */
#define AVX_MUL(d, a, b) AVX_OP3("vmulps", d, a, b)        /* d = a * b */
#define AVX_ADDM(d, a, mem) AVX_OP3M("vaddps", d, a, mem)
#define AVX_SUBM(d, a, mem) AVX_OP3M("vsubps", d, a, mem)
#define AVX_FMADD231(d, a, b) AVX_OP3("vfmadd231ps", d, a, b)   /* d = d + a * b */
#define AVX_FNMADD231(d, a, b) AVX_OP3("vfnmadd231ps", d, a, b) /* d = d - a * b */
#define AVX_FMADD213(d, a, b) AVX_OP3("vfmadd213ps", d, a, b)   /* d = a * d + b */
#define AVX_FMSUB213(d, a, b) AVX_OP3("vfmsub213ps", d, a, b)   /* d = a * d - b */
#define AVX_FNMADD213(d, a, b) AVX_OP3("vfnmadd213ps", d, a, b) /* d = b - a * d */
#define AVX_UNPCKLPS(d, a, b) AVX_OP3("vunpcklps", d, a, b)
#define AVX_UNPCKHPS(d, a, b) AVX_OP3("vunpckhps", d, a, b)
#define AVX_UNPCKLPD(d, a, b) AVX_OP3("vunpcklpd", d, a, b)
#define AVX_UNPCKHPD(d, a, b) AVX_OP3("vunpckhpd", d, a, b)
#define AVX_MOV(d, a) "vmovaps %%ymm" a ", %%ymm" d "\n"
#define AVX_LOAD(d, mem) "vmovups " mem ", %%ymm" d "\n"
#define AVX_STORE(mem, a) "vmovups %%ymm" a ", " mem "\n"
#define AVX_STORE_LO(mem, a) "vmovups %%xmm" a ", " mem "\n"
#define AVX_STORE_HI(mem, a) "vextractf128 $1, %%ymm" a ", " mem "\n"
#define AVX_BROADCAST(d, mem) "vbroadcastss " mem ", %%ymm" d "\n"
/* d = the bit pattern `bits` as float in all lanes, using eax. */
#define AVX_SET1(d, bits) "movl $" bits ", %%eax\n vmovd %%eax, %%xmm" d "\n vbroadcastss %%xmm" d ", %%ymm" d "\n"
#define AVX_SQRT1_2 "0x3f3504f3"
#define AVX_TWO "0x40000000"

/* (zr, zi) *= (wr, wi) using the scratch registers t1, t2. */
#define AVX_CMUL(zr, zi, wr, wi, t1, t2)        \
    AVX_MUL(t1, zi, wi) AVX_MUL(t2, zr, wi)     \
    AVX_FMSUB213(zr, wr, t1) AVX_FMADD213(zi, wr, t2)

/* Transposes the 4x4 matrices in the two 128 bit halves of q0..q3: afterwards
 * q_j holds lane j (low half) and lane j + 4 (high half) of the former q0..q3. */
#define AVX_TRANSPOSE(q0, q1, q2, q3, t0, t1, t2, t3)       \
    AVX_UNPCKLPS(t0, q0, q1) AVX_UNPCKHPS(t1, q0, q1)       \
    AVX_UNPCKLPS(t2, q2, q3) AVX_UNPCKHPS(t3, q2, q3)       \
    AVX_UNPCKLPD(q0, t0, t2) AVX_UNPCKHPD(q1, t0, t2)       \
    AVX_UNPCKLPD(q2, t1, t3) AVX_UNPCKHPD(q3, t1, t3)

#define AVX_CLOBBER                                                                          \
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", \
    "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory", "cc"

/* ---- first pass -------------------------------------------------------- */

/*
 * Registers: rdi, r8 stream pointers of x_0 and x_4 (rsi: distance of two
 * streams, rdx: 3 * rsi), rcx weights, r11 positions (r15: their end), rbx
 * output position, r13, r14: 1x and 3x the distance of the output places.
 * Variant A: r9 out, r12 = rbx + 4 * r13. Variant B: r9, r10 = rdi, r8 + rsi/2
 * (the streams of the second vector), r12 out.
 *
 * FWD / BWD: the immediates selecting the real and imaginary parts when
 * deinterleaving. Exchanging them turns the forward into the backward
 * transform (together with the same exchange in the last pass).
 */
#define AVX_FIRST_FWD "0x88", "0xdd"
#define AVX_FIRST_BWD "0xdd", "0x88"

/* Loads the sum or difference (OP) of the streams p and p + 4 and
 * deinterleaves it: 4 complex numbers at LO and 4 at HI. */
#define AVX_FIRST_LOAD_(OP, re, im, lo, lo4, hi, hi4, SRE, SIM)       \
    AVX_LOAD("12", lo) OP("12", "12", lo4)                            \
    AVX_LOAD("13", hi) OP("13", "13", hi4)                            \
    "vshufps $" SRE ", %%ymm13, %%ymm12, %%ymm" re "\n"               \
    "vshufps $" SIM ", %%ymm13, %%ymm12, %%ymm" im "\n"
#define AVX_FIRST_LOAD(OP, re, im, lo, lo4, hi, hi4, DIR) AVX_FIRST_LOAD_(OP, re, im, lo, lo4, hi, hi4, DIR)

/* Loads the 4 streams of a half tile into (0,1) (2,3) (4,5) (6,7). */
#define AVX_FIRST_LOADS_A(OP, DIR)                                                              \
    AVX_FIRST_LOAD(OP, "0", "1", "(%%rdi)", "(%%r8)", "32(%%rdi)", "32(%%r8)", DIR)             \
    AVX_FIRST_LOAD(OP, "2", "3", "(%%rdi,%%rsi)", "(%%r8,%%rsi)", "32(%%rdi,%%rsi)",            \
                   "32(%%r8,%%rsi)", DIR)                                                       \
    AVX_FIRST_LOAD(OP, "4", "5", "(%%rdi,%%rsi,2)", "(%%r8,%%rsi,2)", "32(%%rdi,%%rsi,2)",      \
                   "32(%%r8,%%rsi,2)", DIR)                                                     \
    AVX_FIRST_LOAD(OP, "6", "7", "(%%rdi,%%rdx)", "(%%r8,%%rdx)", "32(%%rdi,%%rdx)",            \
                   "32(%%r8,%%rdx)", DIR)
#define AVX_FIRST_LOADS_B(OP, DIR)                                                              \
    AVX_FIRST_LOAD(OP, "0", "1", "(%%rdi)", "(%%r8)", "(%%r9)", "(%%r10)", DIR)                 \
    AVX_FIRST_LOAD(OP, "2", "3", "(%%rdi,%%rsi)", "(%%r8,%%rsi)", "(%%r9,%%rsi)",               \
                   "(%%r10,%%rsi)", DIR)                                                        \
    AVX_FIRST_LOAD(OP, "4", "5", "(%%rdi,%%rsi,2)", "(%%r8,%%rsi,2)", "(%%r9,%%rsi,2)",         \
                   "(%%r10,%%rsi,2)", DIR)                                                      \
    AVX_FIRST_LOAD(OP, "6", "7", "(%%rdi,%%rdx)", "(%%r8,%%rdx)", "(%%r9,%%rdx)",               \
                   "(%%r10,%%rdx)", DIR)

/* Multiplies (zr, zi) by the next weight vector of the table. */
#define AVX_FIRST_WEIGHT(zr, zi, t2)                                  \
    AVX_LOAD("12", "(%%rcx)") AVX_LOAD("13", "32(%%rcx)")             \
    "add $64, %%rcx\n"                                                \
    AVX_CMUL(zr, zi, "12", "13", "14", t2)

/*
 * Even half: s_p = x_p + x_(p+4) in (0,1) (2,3) (4,5) (6,7).
 * z_c = sum_p s_p * (-i)^(p*c/2) for c = 0, 2, 4, 6, then y_c = z_c * W^(c*k).
 * Result: real parts of y_0, y_2, y_4, y_6 in 4, 6, 8, 0, imaginary parts in
 * 5, 7, 9, 1.
 */
#define AVX_FIRST_EVEN                                                \
    AVX_ADD("8", "0", "4") AVX_ADD("9", "1", "5")     /* a0 = s0 + s2 */ \
    AVX_SUB("0", "0", "4") AVX_SUB("1", "1", "5")     /* a1 = s0 - s2 */ \
    AVX_ADD("10", "2", "6") AVX_ADD("11", "3", "7")   /* a2 = s1 + s3 */ \
    AVX_SUB("2", "2", "6") AVX_SUB("3", "3", "7")     /* a3 = s1 - s3 */ \
    AVX_ADD("4", "8", "10") AVX_ADD("5", "9", "11")   /* z0 = a0 + a2 */ \
    AVX_SUB("8", "8", "10") AVX_SUB("9", "9", "11")   /* z4 = a0 - a2 */ \
    AVX_ADD("6", "0", "3") AVX_SUB("7", "1", "2")     /* z2 = a1 - i*a3 */ \
    AVX_SUB("0", "0", "3") AVX_ADD("1", "1", "2")     /* z6 = a1 + i*a3 */ \
    AVX_FIRST_WEIGHT("6", "7", "10")                                  \
    AVX_FIRST_WEIGHT("8", "9", "10")                                  \
    AVX_FIRST_WEIGHT("0", "1", "10")                                  \
    AVX_TRANSPOSE("4", "6", "8", "0", "12", "13", "14", "2")          \
    AVX_TRANSPOSE("5", "7", "9", "1", "12", "13", "14", "2")

/*
 * Odd half: x_p - x_(p+4) in (0,1) (2,3) (4,5) (6,7), ymm15 = sqrt(1/2).
 * With d_p = (x_p - x_(p+4)) * W8^p: z_c = sum_p d_p * (-i)^(p*(c-1)/2) for
 * c = 1, 3, 5, 7, then y_c = z_c * W^(c*k).
 * Result: real parts of y_1, y_3, y_5, y_7 in 4, 10, 2, 7, imaginary parts in
 * 5, 11, 3, 6.
 */
#define AVX_FIRST_ODD                                                 \
    AVX_ADD("8", "0", "5") AVX_SUB("9", "1", "4")     /* a0 = d0 + d2 = x0 - i*x2 */ \
    AVX_SUB("0", "0", "5") AVX_ADD("1", "1", "4")     /* a1 = d0 - d2 */ \
    /* d1 = h * (s1 + i*t1), d3 = h * (t3 - i*s3): s = re + im, t = im - re */ \
    AVX_ADD("10", "2", "3") AVX_SUB("11", "3", "2")                   \
    AVX_ADD("4", "6", "7") AVX_SUB("5", "7", "6")                     \
    AVX_ADD("2", "10", "5") AVX_SUB("3", "11", "4")   /* a2 / h = d1 + d3 */ \
    AVX_SUB("6", "10", "5") AVX_ADD("7", "11", "4")   /* a3 / h = d1 - d3 */ \
    AVX_MOV("4", "2") AVX_FMADD213("4", "15", "8")    /* z1 = a0 + a2 */ \
    AVX_MOV("5", "3") AVX_FMADD213("5", "15", "9")                    \
    AVX_FNMADD213("2", "15", "8") AVX_FNMADD213("3", "15", "9")       /* z5 = a0 - a2 */ \
    AVX_MOV("10", "7") AVX_FMADD213("10", "15", "0")  /* z3 = a1 - i*a3 */ \
    AVX_MOV("11", "6") AVX_FNMADD213("11", "15", "1")                 \
    AVX_FNMADD213("7", "15", "0") AVX_FMADD213("6", "15", "1")        /* z7 = a1 + i*a3 */ \
    AVX_FIRST_WEIGHT("4", "5", "0")                                   \
    AVX_FIRST_WEIGHT("10", "11", "0")                                 \
    AVX_FIRST_WEIGHT("2", "3", "0")                                   \
    AVX_FIRST_WEIGHT("7", "6", "0")                                   \
    AVX_TRANSPOSE("4", "10", "2", "7", "12", "13", "14", "0")         \
    AVX_TRANSPOSE("5", "11", "3", "6", "12", "13", "14", "0")

/* Variant A: lanes are k = 8*tile + (0, 1, 4, 5, 2, 3, 6, 7). Stores the 4
 * lanes of sub-FFTs in t0..t3 at offset o of the 8 lines, which are at the
 * same position of the 8 eighths of the output. */
#define AVX_FIRST_STORE_A(t0, t1, t2, t3, o)                                          \
    AVX_STORE_LO(o "(%%rbx)", t0) AVX_STORE_HI(o "(%%rbx,%%r13,2)", t0)               \
    AVX_STORE_LO(o "(%%r12)", t1) AVX_STORE_HI(o "(%%r12,%%r13,2)", t1)               \
    AVX_STORE_LO(o "(%%rbx,%%r13)", t2) AVX_STORE_HI(o "(%%rbx,%%r14)", t2)           \
    AVX_STORE_LO(o "(%%r12,%%r13)", t3) AVX_STORE_HI(o "(%%r12,%%r14)", t3)

#define AVX_FIRST_A(DIR)                                              \
    AVX_SET1("15", AVX_SQRT1_2)                                       \
    "lea (%%rdi,%%rsi,4), %%r8\n"                                     \
    "lea (%%rsi,%%rsi,2), %%rdx\n"                                    \
    "lea (%%r13,%%r13,2), %%r14\n"                                    \
    "1:\n"                                                            \
    "movl (%%r11), %%eax\n"                                           \
    "add $4, %%r11\n"                                                 \
    "shl $6, %%rax\n"                                                 \
    "lea (%%r9,%%rax), %%rbx\n"                                       \
    "lea (%%rbx,%%r13,4), %%r12\n"                                    \
    AVX_FIRST_LOADS_A(AVX_ADDM, DIR)                                  \
    AVX_FIRST_EVEN                                                    \
    AVX_FIRST_STORE_A("4", "6", "8", "0", "0")                        \
    AVX_FIRST_STORE_A("5", "7", "9", "1", "32")                       \
    AVX_FIRST_LOADS_A(AVX_SUBM, DIR)                                  \
    AVX_FIRST_ODD                                                     \
    AVX_FIRST_STORE_A("4", "10", "2", "7", "16")                      \
    AVX_FIRST_STORE_A("5", "11", "3", "6", "48")                      \
    "add $64, %%rdi\n"                                                \
    "add $64, %%r8\n"                                                 \
    "cmp %%r15, %%r11\n"                                              \
    "jne 1b\n"

/* Variant B: lanes are k = 4*tile + (0, 1, h, h+1, 2, 3, h+2, h+3) with h =
 * N/16. After the transposition t0, t1 hold the lines of k = (0 | 2) and
 * (1 | 3), t2, t3 their partners in the first stage of the sub-FFTs, which is
 * done here. The 4 pairs of lines go to the same position of the 4 quarters
 * of the output. Uses ymm12, ymm13. */
#define AVX_FIRST_STORE_B(t0, t1, t2, t3, o, o64)                                     \
    AVX_ADD("12", t0, t2) AVX_SUB(t0, t0, t2)                                         \
    AVX_ADD("13", t1, t3) AVX_SUB(t1, t1, t3)                                         \
    AVX_STORE_LO(o "(%%rbx)", "12") AVX_STORE_HI(o "(%%rbx,%%r13)", "12")             \
    AVX_STORE_LO(o64 "(%%rbx)", t0) AVX_STORE_HI(o64 "(%%rbx,%%r13)", t0)             \
    AVX_STORE_LO(o "(%%rbx,%%r13,2)", "13") AVX_STORE_HI(o "(%%rbx,%%r14)", "13")     \
    AVX_STORE_LO(o64 "(%%rbx,%%r13,2)", t1) AVX_STORE_HI(o64 "(%%rbx,%%r14)", t1)

#define AVX_FIRST_B(DIR)                                              \
    AVX_SET1("15", AVX_SQRT1_2)                                       \
    "lea (%%rdi,%%rsi,4), %%r8\n"                                     \
    "lea (%%rsi,%%rsi,2), %%rdx\n"                                    \
    "lea (%%r13,%%r13,2), %%r14\n"                                    \
    "mov %%rsi, %%rax\n"                                              \
    "shr $1, %%rax\n"                                                 \
    "lea (%%rdi,%%rax), %%r9\n"                                       \
    "lea (%%r8,%%rax), %%r10\n"                                       \
    "1:\n"                                                            \
    "movl (%%r11), %%eax\n"                                           \
    "add $4, %%r11\n"                                                 \
    "shl $6, %%rax\n"                                                 \
    "lea (%%r12,%%rax), %%rbx\n"                                      \
    AVX_FIRST_LOADS_B(AVX_ADDM, DIR)                                  \
    AVX_FIRST_EVEN                                                    \
    AVX_FIRST_STORE_B("4", "6", "8", "0", "0", "64")                  \
    AVX_FIRST_STORE_B("5", "7", "9", "1", "32", "96")                 \
    AVX_FIRST_LOADS_B(AVX_SUBM, DIR)                                  \
    AVX_FIRST_ODD                                                     \
    AVX_FIRST_STORE_B("4", "10", "2", "7", "16", "80")                \
    AVX_FIRST_STORE_B("5", "11", "3", "6", "48", "112")               \
    "add $32, %%rdi\n"                                                \
    "add $32, %%r8\n"                                                 \
    "add $32, %%r9\n"                                                 \
    "add $32, %%r10\n"                                                \
    "cmp %%r15, %%r11\n"                                              \
    "jne 1b\n"

/* in: start of the first stream, stride: distance of two streams in bytes,
 * place: distance in bytes of the places a tile writes to. */
#define AVX_DEFINE_FIRST(name, BODY, OUT, UNUSED, DIR)                                  \
    TW_AVX2 static TW_NOINLINE void name(const float* in_, size_t stride_, float* out_, \
                                         const float* w_, const uint32_t* position_,    \
                                         const uint32_t* position_end_,                 \
                                         size_t place_) {                               \
        register const float* in __asm__("rdi") = in_;                                  \
        register size_t stride __asm__("rsi") = stride_;                                \
        register const float* w __asm__("rcx") = w_;                                    \
        register const uint32_t* position __asm__("r11") = position_;                   \
        register const uint32_t* position_end __asm__("r15") = position_end_;           \
        register size_t place __asm__("r13") = place_;                                  \
        register float* out __asm__(OUT) = out_;                                        \
        __asm__ volatile(BODY(DIR)                                                      \
                         : "+r"(in), "+r"(w), "+r"(position)                            \
                         : "r"(stride), "r"(position_end), "r"(place), "r"(out)         \
                         : "rax", "rbx", "rdx", "r8", "r10", "r14", UNUSED, AVX_CLOBBER); \
    }

AVX_DEFINE_FIRST(tw_avx2_first_a_fwd, AVX_FIRST_A, "r9", "r12", AVX_FIRST_FWD)
AVX_DEFINE_FIRST(tw_avx2_first_a_bwd, AVX_FIRST_A, "r9", "r12", AVX_FIRST_BWD)
AVX_DEFINE_FIRST(tw_avx2_first_b_fwd, AVX_FIRST_B, "r12", "r9", AVX_FIRST_FWD)
AVX_DEFINE_FIRST(tw_avx2_first_b_bwd, AVX_FIRST_B, "r12", "r9", AVX_FIRST_BWD)

/* ---- radix-4 passes ---------------------------------------------------- */

/* Output of the lines of a butterfly: STORE(re, im, address of the line,
 * scratch registers t0, t1). */
#define AVX_STORE_SPLIT(re, im, mem, t0, t1) AVX_STORE(mem, re) AVX_STORE("32+" mem, im)
/* Interleaves real and imaginary parts and restores the order of the sub-FFTs
 * (0, 2, 4, 6, 1, 3, 5, 7) -> (0, 1, 2, .. 7). */
#define AVX_STORE_INTERLEAVED_(lo, hi, mem, t0, t1)                   \
    AVX_UNPCKLPS(t0, lo, hi) AVX_UNPCKHPS(t1, lo, hi)                 \
    "vpermpd $0xd8, %%ymm" t0 ", %%ymm" t0 "\n"                       \
    "vpermpd $0xd8, %%ymm" t1 ", %%ymm" t1 "\n"                       \
    AVX_STORE(mem, t0) AVX_STORE("32+" mem, t1)
#define AVX_STORE_INTERLEAVED_FWD(re, im, mem, t0, t1) AVX_STORE_INTERLEAVED_(re, im, mem, t0, t1)
#define AVX_STORE_INTERLEAVED_BWD(re, im, mem, t0, t1) AVX_STORE_INTERLEAVED_(im, re, mem, t0, t1)

/*
 * Radix-4 butterfly of the lines a0..a3 at rdi + (0, 1, 2, 3) * rsi (rdx = 3 *
 * rsi) with the weights w2, w1, w3 = (8,9) (10,11) (12,13) and ymm15 = 2:
 *
 *     b1 = w2*a1, b2 = w1*a2, b3 = w3*a3
 *     y0 = (a0 + b1) + (b2 + b3)        y1 = (a0 - b1) - i*(b2 - b3)
 *     y2 = (a0 + b1) - (b2 + b3)        y3 = (a0 - b1) + i*(b2 - b3)
 *
 * Products are accumulated with fused multiply-adds and differences obtained
 * as 2*x - (x + y), which takes 16 multiply-type and 8 add-type operations
 * instead of 12 and 16: Intel cores up to Skylake execute both on the same
 * ports (Haswell has a single adder), on AMD cores the mix keeps both types
 * of ports busy. Register copies are free on all AVX2 capable cores.
 */
#define AVX_R4_BUTTERFLY(STORE)                                                       \
    AVX_LOAD("0", "(%%rdi)") AVX_LOAD("1", "32(%%rdi)")                               \
    AVX_LOAD("2", "(%%rdi,%%rsi)") AVX_LOAD("3", "32(%%rdi,%%rsi)")                   \
    AVX_MOV("4", "0") AVX_MOV("5", "1")                                               \
    AVX_FMADD231("0", "2", "8") AVX_FMADD231("1", "2", "9")                           \
    AVX_FNMADD231("0", "3", "9") AVX_FMADD231("1", "3", "8")       /* (0,1) = a0 + b1 */ \
    AVX_FMSUB213("4", "15", "0") AVX_FMSUB213("5", "15", "1")      /* (4,5) = a0 - b1 */ \
    AVX_LOAD("2", "(%%rdi,%%rsi,2)") AVX_LOAD("3", "32(%%rdi,%%rsi,2)")               \
    AVX_MUL("6", "2", "10") AVX_MUL("7", "2", "11")                                   \
    AVX_FNMADD231("6", "3", "11") AVX_FMADD231("7", "3", "10")     /* (6,7) = b2 */   \
    AVX_LOAD("2", "(%%rdi,%%rdx)") AVX_LOAD("3", "32(%%rdi,%%rdx)")                   \
    AVX_MOV("8", "6") AVX_MOV("9", "7")                                               \
    AVX_FMADD231("8", "2", "12") AVX_FMADD231("9", "2", "13")                         \
    AVX_FNMADD231("8", "3", "13") AVX_FMADD231("9", "3", "12")     /* (8,9) = b2 + b3 */ \
    AVX_FMSUB213("6", "15", "8") AVX_FMSUB213("7", "15", "9")      /* (6,7) = b2 - b3 */ \
    AVX_ADD("2", "0", "8") AVX_ADD("3", "1", "9")                  /* y0 */           \
    AVX_SUB("0", "0", "8") AVX_SUB("1", "1", "9")                  /* y2 */           \
    AVX_ADD("10", "4", "7") AVX_SUB("11", "5", "6")                /* y1 */           \
    AVX_SUB("4", "4", "7") AVX_ADD("5", "5", "6")                  /* y3 */           \
    STORE("2", "3", "(%%rdi)", "12", "13")                                            \
    STORE("10", "11", "(%%rdi,%%rsi)", "12", "13")                                    \
    STORE("0", "1", "(%%rdi,%%rsi,2)", "12", "13")                                    \
    STORE("4", "5", "(%%rdi,%%rdx)", "12", "13")

/* Radix-4 pass: rdi data, r8 weights, r10 number of blocks, r11 stride s in
 * lines. Each block consists of 4*s lines and s butterflies. */
#define AVX_R4(STORE)                                                 \
    AVX_SET1("15", AVX_TWO)                                           \
    "mov %%r11, %%rsi\n"                                              \
    "shl $6, %%rsi\n"                                                 \
    "lea (%%rsi,%%rsi,2), %%rdx\n"                                    \
    "1:\n"                                                            \
    "mov %%r8, %%rcx\n"                                               \
    "mov %%r11, %%r9\n"                                               \
    "2:\n"                                                            \
    AVX_BROADCAST("8", "(%%rcx)") AVX_BROADCAST("9", "4(%%rcx)")      \
    AVX_BROADCAST("10", "8(%%rcx)") AVX_BROADCAST("11", "12(%%rcx)")  \
    AVX_BROADCAST("12", "16(%%rcx)") AVX_BROADCAST("13", "20(%%rcx)") \
    "add $32, %%rcx\n"                                                \
    AVX_R4_BUTTERFLY(STORE)                                           \
    "add $64, %%rdi\n"                                                \
    "dec %%r9\n"                                                      \
    "jnz 2b\n"                                                        \
    "add %%rdx, %%rdi\n"                                              \
    "dec %%r10\n"                                                     \
    "jnz 1b\n"

#define AVX_DEFINE_R4(name, STORE)                                                      \
    TW_AVX2 static TW_NOINLINE void name(float* data_, const float* w_, size_t s_,      \
                                         size_t blocks_) {                              \
        register float* data __asm__("rdi") = data_;                                    \
        register const float* w __asm__("r8") = w_;                                     \
        register size_t blocks __asm__("r10") = blocks_;                                \
        register size_t s __asm__("r11") = s_;                                          \
        __asm__ volatile(AVX_R4(STORE)                                                  \
                         : "+r"(data), "+r"(blocks)                                     \
                         : "r"(w), "r"(s)                                               \
                         : "rax", "rcx", "rdx", "rsi", "r9", AVX_CLOBBER);              \
    }

AVX_DEFINE_R4(tw_avx2_r4, AVX_STORE_SPLIT)
AVX_DEFINE_R4(tw_avx2_r4_last_fwd, AVX_STORE_INTERLEAVED_FWD)
AVX_DEFINE_R4(tw_avx2_r4_last_bwd, AVX_STORE_INTERLEAVED_BWD)

/* Radix-4 pass with stride 1 and unit weights: rdi data, r10 number of
 * blocks of 4 consecutive lines. */
#define AVX_T4                                                        \
    "1:\n"                                                            \
    AVX_LOAD("0", "(%%rdi)") AVX_LOAD("1", "32(%%rdi)")               \
    AVX_LOAD("2", "128(%%rdi)") AVX_LOAD("3", "160(%%rdi)")           \
    AVX_ADDM("4", "0", "64(%%rdi)") AVX_ADDM("5", "1", "96(%%rdi)")   /* a0 + a1 */ \
    AVX_SUBM("0", "0", "64(%%rdi)") AVX_SUBM("1", "1", "96(%%rdi)")   /* a0 - a1 */ \
    AVX_ADDM("6", "2", "192(%%rdi)") AVX_ADDM("7", "3", "224(%%rdi)") /* a2 + a3 */ \
    AVX_SUBM("2", "2", "192(%%rdi)") AVX_SUBM("3", "3", "224(%%rdi)") /* a2 - a3 */ \
    AVX_ADD("8", "4", "6") AVX_ADD("9", "5", "7")                     /* y0 */ \
    AVX_SUB("4", "4", "6") AVX_SUB("5", "5", "7")                     /* y2 */ \
    AVX_ADD("10", "0", "3") AVX_SUB("11", "1", "2")                   /* y1 */ \
    AVX_SUB("0", "0", "3") AVX_ADD("1", "1", "2")                     /* y3 */ \
    AVX_STORE("(%%rdi)", "8") AVX_STORE("32(%%rdi)", "9")             \
    AVX_STORE("64(%%rdi)", "10") AVX_STORE("96(%%rdi)", "11")         \
    AVX_STORE("128(%%rdi)", "4") AVX_STORE("160(%%rdi)", "5")         \
    AVX_STORE("192(%%rdi)", "0") AVX_STORE("224(%%rdi)", "1")         \
    "add $256, %%rdi\n"                                               \
    "dec %%r10\n"                                                     \
    "jnz 1b\n"

TW_AVX2 static TW_NOINLINE void tw_avx2_t4(float* data_, size_t blocks_) {
    register float* data __asm__("rdi") = data_;
    register size_t blocks __asm__("r10") = blocks_;
    __asm__ volatile(AVX_T4 : "+r"(data), "+r"(blocks) : : AVX_CLOBBER);
}

/* ---- driver ------------------------------------------------------------ */

/* Blocks of at most this many lines (32 KiB) are processed pass by pass.
 * Larger blocks are split into 4 sub-blocks that are finished one after the
 * other, so that all but the last few passes run on data in the L1 cache. */
#ifndef TW_AVX2_LEAF_LINES
#define TW_AVX2_LEAF_LINES 512
#endif

static void tw_avx2_first(unsigned log2n, const float* w, const float* in, float* out, int backward) {
    const size_t n = (size_t)1 << log2n;
    const uint32_t* position = tw_avx2_positions(log2n, w);
    if (tw_avx2_first_variant(log2n) == TW_AVX2_FIRST_A)
        (backward ? tw_avx2_first_a_bwd : tw_avx2_first_a_fwd)(in, n, out, w, position, position + n / 64, n);
    else
        (backward ? tw_avx2_first_b_bwd : tw_avx2_first_b_fwd)(in, n, out, w, position, position + n / 64, 2 * n);
}

/* Performs the radix-4 passes with strides s0 <= s < s_end on `lines` lines,
 * one pass after the other. `w` is the radix-4 weight table of the transform
 * and `s0` the stride of its first radix-4 pass. */
static void tw_avx2_passes(float* data, const float* w, size_t s0, size_t s_end, size_t lines) {
    size_t s = s0;
    if (s == 1 && s < s_end) {
        tw_avx2_t4(data, lines / 4);
        s = 4;
    }
    for (; s < s_end; s *= 4) tw_avx2_r4(data, w + tw_avx2_r4_weights_offset(s0, s), s, lines / (4 * s));
}

/* Performs all radix-4 passes with a stride below `lines` on a block of
 * `lines` lines. */
static void tw_avx2_block(float* data, const float* w, size_t s0, size_t lines) {
    if (lines <= TW_AVX2_LEAF_LINES) {
        tw_avx2_passes(data, w, s0, lines, lines);
        return;
    }
    const size_t s = lines / 4;
    for (unsigned i = 0; i < 4; i++) tw_avx2_block(data + i * s * TW_LINE_FLOATS, w, s0, s);
    tw_avx2_r4(data, w + tw_avx2_r4_weights_offset(s0, s), s, 1);
}

/* Out-of-place transform of size 2^log2n >= 64 with the weights from
 * tw_avx2_weights_fill. */
static void tw_avx2_fft(unsigned log2n, const float* w, const float* in, float* out, int backward) {
    const size_t m = tw_lines(log2n);
    const size_t s0 = tw_avx2_first_stride(log2n);
    tw_avx2_first(log2n, w, in, out, backward);
    w += (m / 8) * TW_AVX2_TILE_FLOATS;
    /* All passes but the last one, which also interleaves the output. */
    const size_t s = m / 4;
    if (m > TW_AVX2_LEAF_LINES)
        for (unsigned i = 0; i < 4; i++) tw_avx2_block(out + i * s * TW_LINE_FLOATS, w, s0, s);
    else
        tw_avx2_passes(out, w, s0, s, m);
    (backward ? tw_avx2_r4_last_bwd : tw_avx2_r4_last_fwd)(out, w + tw_avx2_r4_weights_offset(s0, s), s, 1);
}

#endif /* TW_AVX2_H */
