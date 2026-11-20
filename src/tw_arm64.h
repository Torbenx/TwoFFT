/* TwoFFT - ARM64 (NEON) kernels.
 *
 * Every kernel is a normal C function whose body is a single inline assembly
 * block. This gives full control over the use of the 32 vector registers
 * while leaving calling convention and callee saved registers to the
 * compiler.
 *
 * Register conventions
 * --------------------
 * Vector registers are used in pairs: pair p consists of v(2p) and v(2p+1)
 * and holds 4 complex numbers as separate real and imaginary vectors.
 * Kernels that read the input or write the output exist in a forward and a
 * backward flavour. They only differ in which register of a pair holds the
 * real part: the backward transform is the forward transform with real and
 * imaginary parts exchanged on input and output. Deinterleaving loads (ld2)
 * always fill the low register of a pair first, interleaving stores (st2)
 * always write it first, so exchanging the roles of the two registers
 * exchanges real and imaginary parts in memory at no cost.
 */
#ifndef TW_ARM64_H
#define TW_ARM64_H

#include <string.h>

#include "tw_common.h"
#include "tw_small.h"
#include "tw_weights.h"

/* ---- register pairs ---------------------------------------------------- */

#define A64_LO0 "0"
#define A64_LO1 "2"
#define A64_LO2 "4"
#define A64_LO3 "6"
#define A64_LO4 "8"
#define A64_LO5 "10"
#define A64_LO6 "12"
#define A64_LO7 "14"
#define A64_LO8 "16"
#define A64_LO9 "18"
#define A64_LO10 "20"
#define A64_LO11 "22"
#define A64_LO12 "24"
#define A64_LO13 "26"
#define A64_HI0 "1"
#define A64_HI1 "3"
#define A64_HI2 "5"
#define A64_HI3 "7"
#define A64_HI4 "9"
#define A64_HI5 "11"
#define A64_HI6 "13"
#define A64_HI7 "15"
#define A64_HI8 "17"
#define A64_HI9 "19"
#define A64_HI10 "21"
#define A64_HI11 "23"
#define A64_HI12 "25"
#define A64_HI13 "27"

/* Number of the low / high register of pair p. */
#define A64_LO(p) A64_LO##p
#define A64_HI(p) A64_HI##p

/* ---- instructions ------------------------------------------------------ */
/* Register arguments are register numbers as string literals. */

#define A64_OP3(op, d, a, b) op " v" d ".4s, v" a ".4s, v" b ".4s\n"
#define A64_FADD(d, a, b) A64_OP3("fadd", d, a, b)
#define A64_FSUB(d, a, b) A64_OP3("fsub", d, a, b)
#define A64_FMUL(d, a, b) A64_OP3("fmul", d, a, b)
#define A64_FMLA(d, a, b) A64_OP3("fmla", d, a, b) /* d += a * b */
#define A64_FMLS(d, a, b) A64_OP3("fmls", d, a, b) /* d -= a * b */
#define A64_TRN1(d, a, b) A64_OP3("trn1", d, a, b)
#define A64_TRN2(d, a, b) A64_OP3("trn2", d, a, b)
#define A64_ZIP1(d, a, b) A64_OP3("zip1", d, a, b)
#define A64_ZIP2(d, a, b) A64_OP3("zip2", d, a, b)
#define A64_UZP1(d, a, b) A64_OP3("uzp1", d, a, b)
#define A64_UZP2(d, a, b) A64_OP3("uzp2", d, a, b)
#define A64_ZIP1D(d, a, b) "zip1 v" d ".2d, v" a ".2d, v" b ".2d\n"
#define A64_ZIP2D(d, a, b) "zip2 v" d ".2d, v" a ".2d, v" b ".2d\n"
/* The same with lane `i` of register w as second factor. */
#define A64_FMULE(d, a, w, i) "fmul v" d ".4s, v" a ".4s, v" w ".s[" #i "]\n"
#define A64_FMLAE(d, a, w, i) "fmla v" d ".4s, v" a ".4s, v" w ".s[" #i "]\n"
#define A64_FMLSE(d, a, w, i) "fmls v" d ".4s, v" a ".4s, v" w ".s[" #i "]\n"

#define A64_LDP(a, b, addr) "ldp q" a ", q" b ", " addr "\n"
#define A64_STP(a, b, addr) "stp q" a ", q" b ", " addr "\n"
#define A64_LDRQ(a, addr) "ldr q" a ", " addr "\n"
#define A64_LDRD(a, addr) "ldr d" a ", " addr "\n"
#define A64_LD2(p, addr) "ld2 {v" A64_LO(p) ".4s, v" A64_HI(p) ".4s}, " addr "\n"
/* Stores pair p interleaved (low register first) using scratch pair t. This
 * is never slower than st2, which takes 3 cycles on some cores (Cortex-X4). */
#define A64_ST2(p, t, addr)                           \
    A64_ZIP1(A64_LO(t), A64_LO(p), A64_HI(p))         \
    A64_ZIP2(A64_HI(t), A64_LO(p), A64_HI(p))         \
    A64_STP(A64_LO(t), A64_HI(t), addr)

/* v31 = sqrt(1/2) in all lanes, using w17. */
#define A64_SET_V31_SQRT1_2        \
    "movz w17, #0x04f3\n"          \
    "movk w17, #0x3f35, lsl #16\n" \
    "dup v31.4s, w17\n"

/* ---- complex arithmetic on register pairs ------------------------------ */
/* R and I are macros mapping a pair to its real and imaginary register. */

#define A64_FWD_RE(p) A64_LO(p)
#define A64_FWD_IM(p) A64_HI(p)
#define A64_BWD_RE(p) A64_HI(p)
#define A64_BWD_IM(p) A64_LO(p)

/* s = a + b, d = a - b. s must not be a or b, d may be either. */
#define A64_BF(R, I, s, d, a, b)                          \
    A64_FADD(R(s), R(a), R(b)) A64_FSUB(R(d), R(a), R(b)) \
    A64_FADD(I(s), I(a), I(b)) A64_FSUB(I(d), I(a), I(b))

/* p = a + i*b, m = a - i*b. p must not be a or b, m may be a but not b. */
#define A64_BFJ(R, I, p, m, a, b)                         \
    A64_FSUB(R(p), R(a), I(b)) A64_FADD(I(p), I(a), R(b)) \
    A64_FADD(R(m), R(a), I(b)) A64_FSUB(I(m), I(a), R(b))

/* x1 = h * (re(b) + im(b)), x2 = h * (im(b) - re(b)) with h = sqrt(1/2).
 * W8 * b = x1 + i*x2 and W8^3 * b = x2 - i*x1 for W8 = exp(-2*pi*i/8). */
#define A64_ROT8(R, I, b, x1, x2, HMUL) \
    A64_FADD(x1, R(b), I(b)) A64_FSUB(x2, I(b), R(b)) HMUL(x1) HMUL(x2)

/* p = a + W8*b, m = a - W8*b. m may be b, p may be a or b. x1, x2: scratch. */
#define A64_BFW8(R, I, p, m, a, b, x1, x2, HMUL)        \
    A64_ROT8(R, I, b, x1, x2, HMUL)                     \
    A64_FSUB(R(m), R(a), x1) A64_FSUB(I(m), I(a), x2)   \
    A64_FADD(R(p), R(a), x1) A64_FADD(I(p), I(a), x2)

/* p = a + W8^3*b, m = a - W8^3*b. m may be b, p may be a or b. */
#define A64_BFW83(R, I, p, m, a, b, x1, x2, HMUL)       \
    A64_ROT8(R, I, b, x1, x2, HMUL)                     \
    A64_FSUB(R(m), R(a), x2) A64_FADD(I(m), I(a), x1)   \
    A64_FADD(R(p), R(a), x2) A64_FSUB(I(p), I(a), x1)

/* d = a * w for a weight vector in registers wr, wi. d must not be a. */
#define A64_CMUL(R, I, d, a, wr, wi)                      \
    A64_FMUL(R(d), R(a), wr) A64_FMUL(I(d), R(a), wi)     \
    A64_FMLS(R(d), I(a), wi) A64_FMLA(I(d), I(a), wr)

/* d = a * w for a scalar weight in lanes ir (real), ii (imaginary) of w. */
#define A64_CMULE(R, I, d, a, w, ir, ii)                          \
    A64_FMULE(R(d), R(a), w, ir) A64_FMULE(I(d), R(a), w, ii)     \
    A64_FMLSE(R(d), I(a), w, ii) A64_FMLAE(I(d), I(a), w, ir)

#define A64_HMUL_V31(x) "fmul v" x ".4s, v" x ".4s, v31.4s\n"
#define A64_HMUL_V31E2(x) "fmul v" x ".4s, v" x ".4s, v31.s[2]\n"

/*
 * 8 point DFT of register pairs (see tw_dft8 in tw_scalar.h).
 *
 *   input   position:  0  1  2  3  4  5  6  7   (bit-reversed order)
 *           pair:      8  1  9 11  3  5  7  6
 *   output  Y0..Y7 in pairs 0..7
 *
 * Pairs 0, 2, 4, 10 must be free on entry, pairs 8..11 are free on exit.
 * x1, x2 are two scratch registers, HMUL(x) multiplies x by sqrt(1/2).
 * The comments give the location of the results of each butterfly.
 */
#define A64_DFT8(R, I, x1, x2, HMUL)                              \
    A64_BF(R, I, 2, 1, 8, 1)    /* A0@2   A1@1  */                \
    A64_BF(R, I, 0, 11, 9, 11)  /* A2@0   A3@11 */                \
    A64_BF(R, I, 4, 5, 3, 5)    /* A4@4   A5@5  */                \
    A64_BF(R, I, 10, 6, 7, 6)   /* A6@10  A7@6  */                \
    A64_BF(R, I, 8, 2, 2, 0)    /* B0@8   B2@2  */                \
    A64_BFJ(R, I, 3, 1, 1, 11)  /* B3@3   B1@1  */                \
    A64_BF(R, I, 9, 10, 4, 10)  /* B4@9   B6@10 */                \
    A64_BFJ(R, I, 7, 5, 5, 6)   /* B7@7   B5@5  */                \
    A64_BF(R, I, 0, 4, 8, 9)    /* Y0@0   Y4@4  */                \
    A64_BFJ(R, I, 6, 2, 2, 10)  /* Y6@6   Y2@2  */                \
    A64_BFW8(R, I, 1, 5, 1, 5, x1, x2, HMUL)  /* Y1@1  Y5@5 */    \
    A64_BFW83(R, I, 3, 7, 3, 7, x1, x2, HMUL) /* Y3@3  Y7@7 */

/* Transposes the 4x4 matrix with rows a, b, c, d in place: afterwards
 * register a holds lane 0 of the former a, b, c, d and so on. */
#define A64_TRANSPOSE(a, b, c, d, t0, t1, t2, t3)                 \
    A64_TRN1(t0, a, b) A64_TRN2(t1, a, b)                         \
    A64_TRN1(t2, c, d) A64_TRN2(t3, c, d)                         \
    A64_ZIP1D(a, t0, t2) A64_ZIP1D(b, t1, t3)                     \
    A64_ZIP2D(c, t0, t2) A64_ZIP2D(d, t1, t3)

#define A64_CLOBBER_V                                                                    \
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",     \
    "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",  \
    "v25", "v26", "v27", "v28", "v29", "v30", "v31"

/* ---- first pass -------------------------------------------------------- */

/*
 * All three variants share the following steps for a tile of 32 complex
 * numbers x[k + p*N/8], p = 0..7, with one k per lane:
 *
 *   1. load x_p deinterleaved into the input pairs of A64_DFT8
 *   2. 8 point DFT over p: z_c, c = 0..7
 *   3. y_c = z_c * W_N^(c*k) with weights from the table (x8)
 *   4. transpose, so that each register holds 4 values of c for one k:
 *      line(k) = [y_0..y_3] [y_4..y_7]
 *
 * After step 3 y_0..y_7 are in pairs 0, 8, 9, 10, 11, 1, 2, 3, after step 4
 * the first halves of the lines of lanes 0..3 are in pairs 0, 8, 9, 10 and
 * the second halves in pairs 11, 1, 2, 3.
 *
 * The 8 streams x_p start `stride` bytes apart. This is N bytes when reading
 * the input directly; large transforms are read through a small buffer, see
 * tw_a64_first.
 *
 * Registers: x0..x7 stream pointers, x8 weights, x9 tile, x10 end of the
 * tile range, x11 bit-reversal shift, x12 out, x13 stride, x19 tiles to skip
 * after each run of x20 tiles, v31 sqrt(1/2).
 */
#define A64_FIRST_DFT_WEIGHT_TRANSPOSE(R, I)                          \
    A64_DFT8(R, I, "24", "25", A64_HMUL_V31)                          \
    A64_LDP("26", "27", "[x8], #32") A64_CMUL(R, I, 8, 1, "26", "27") \
    A64_LDP("28", "29", "[x8], #32") A64_CMUL(R, I, 9, 2, "28", "29") \
    A64_LDP("26", "27", "[x8], #32") A64_CMUL(R, I, 10, 3, "26", "27") \
    A64_LDP("28", "29", "[x8], #32") A64_CMUL(R, I, 11, 4, "28", "29") \
    A64_LDP("26", "27", "[x8], #32") A64_CMUL(R, I, 1, 5, "26", "27") \
    A64_LDP("28", "29", "[x8], #32") A64_CMUL(R, I, 2, 6, "28", "29") \
    A64_LDP("26", "27", "[x8], #32") A64_CMUL(R, I, 3, 7, "26", "27") \
    A64_TRANSPOSE(R(0), R(8), R(9), R(10), "24", "25", "26", "27")    \
    A64_TRANSPOSE(I(0), I(8), I(9), I(10), "28", "29", "30", "12")    \
    A64_TRANSPOSE(R(11), R(1), R(2), R(3), "24", "25", "26", "27")    \
    A64_TRANSPOSE(I(11), I(1), I(2), I(3), "28", "29", "30", "12")

/* Common setup: stream pointers, x14 = N bytes (the size of 1/8 of the
 * output), bit-reversal shift 64 - (log2n - 5). */
#define A64_FIRST_SETUP          \
    A64_SET_V31_SQRT1_2          \
    "add x1, x0, x13\n"          \
    "add x2, x1, x13\n"          \
    "add x3, x2, x13\n"          \
    "add x4, x3, x13\n"          \
    "add x5, x4, x13\n"          \
    "add x6, x5, x13\n"          \
    "add x7, x6, x13\n"          \
    "mov x14, #1\n"              \
    "lsl x14, x14, x11\n"        \
    "mov x15, #69\n"             \
    "sub x11, x15, x11\n"

/* x14 = out + (bitrev(tile) << shift) */
#define A64_FIRST_DST(shift)     \
    "rbit x14, x9\n"             \
    "lsr x14, x14, x11\n"        \
    "add x14, x12, x14, lsl #" #shift "\n"

/* The tiles are processed in runs of x20 consecutive tiles, after each of
 * which x19 tiles are skipped (see "Order of the first pass tiles" in
 * tw_weights.h). Label 1 starts a run, label 2 a tile. */
#define A64_FIRST_RUN            \
    "1:\n"                       \
    "mov x21, x20\n"             \
    "2:\n"

#define A64_FIRST_NEXT           \
    "add x9, x9, #1\n"           \
    "subs x21, x21, #1\n"        \
    "b.ne 2b\n"                  \
    "add x9, x9, x19\n"          \
    "cmp x9, x10\n"              \
    "b.lo 1b\n"

/* Variant A: lanes are k = 4*tile + (0, 1, 2, 3). The 4 lines go to the same
 * position in the 4 quarters of the output (x13: size of a quarter). */
#define A64_FIRST_A(R, I)                                             \
    A64_FIRST_SETUP                                                   \
    "lsl x13, x14, #1\n"                                              \
    A64_FIRST_RUN                                                     \
    A64_LD2(8, "[x0], #32") A64_LD2(1, "[x4], #32")                   \
    A64_LD2(9, "[x2], #32") A64_LD2(11, "[x6], #32")                  \
    A64_LD2(3, "[x1], #32") A64_LD2(5, "[x5], #32")                   \
    A64_LD2(7, "[x3], #32") A64_LD2(6, "[x7], #32")                   \
    A64_FIRST_DFT_WEIGHT_TRANSPOSE(R, I)                              \
    A64_FIRST_DST(6)                                                  \
    "add x15, x14, x13\n"                                             \
    "add x16, x14, x13, lsl #1\n"                                     \
    "add x17, x15, x13, lsl #1\n"                                     \
    A64_STP(R(0), I(0), "[x14]") A64_STP(R(11), I(11), "[x14, #32]")  \
    A64_STP(R(8), I(8), "[x16]") A64_STP(R(1), I(1), "[x16, #32]")    \
    A64_STP(R(9), I(9), "[x15]") A64_STP(R(2), I(2), "[x15, #32]")    \
    A64_STP(R(10), I(10), "[x17]") A64_STP(R(3), I(3), "[x17, #32]")  \
    A64_FIRST_NEXT

/* Variant B: lanes are k = 2*tile + (0, 1, N/16, N/16 + 1). Lanes 0, 2 and
 * 1, 3 are partners of the first stage of the sub-FFTs, which is done here.
 * The two pairs of lines go to the same position in the two halves of the
 * output. x13: stride / 2, x16: size of half the output. */
#define A64_FIRST_B_LOAD(p, ptr)                                      \
    A64_LDRQ("24", "[" ptr ", x13]") A64_LDRQ("25", "[" ptr "], #16") \
    A64_UZP1(A64_LO(p), "25", "24") A64_UZP2(A64_HI(p), "25", "24")

#define A64_FIRST_B(R, I)                                             \
    A64_FIRST_SETUP                                                   \
    "lsr x13, x13, #1\n"                                              \
    "lsl x16, x14, #2\n"                                              \
    A64_FIRST_RUN                                                     \
    A64_FIRST_B_LOAD(8, "x0") A64_FIRST_B_LOAD(1, "x4")               \
    A64_FIRST_B_LOAD(9, "x2") A64_FIRST_B_LOAD(11, "x6")              \
    A64_FIRST_B_LOAD(3, "x1") A64_FIRST_B_LOAD(5, "x5")               \
    A64_FIRST_B_LOAD(7, "x3") A64_FIRST_B_LOAD(6, "x7")               \
    A64_FIRST_DFT_WEIGHT_TRANSPOSE(R, I)                              \
    A64_BF(R, I, 4, 9, 0, 9) A64_BF(R, I, 5, 2, 11, 2)                \
    A64_BF(R, I, 6, 10, 8, 10) A64_BF(R, I, 7, 3, 1, 3)               \
    A64_FIRST_DST(7)                                                  \
    "add x15, x14, x16\n"                                             \
    A64_STP(R(4), I(4), "[x14]") A64_STP(R(5), I(5), "[x14, #32]")    \
    A64_STP(R(9), I(9), "[x14, #64]") A64_STP(R(2), I(2), "[x14, #96]") \
    A64_STP(R(6), I(6), "[x15]") A64_STP(R(7), I(7), "[x15, #32]")    \
    A64_STP(R(10), I(10), "[x15, #64]") A64_STP(R(3), I(3), "[x15, #96]") \
    A64_FIRST_NEXT

/* Variant C: lanes are k = tile + (0, N/16, N/32, 3N/32), the inputs of one
 * radix-4 butterfly of the sub-FFTs, which is done here. The 4 lines are
 * adjacent in the output. x13, x15, x16: 1/4, 2/4, 3/4 of the stride. */
#define A64_FIRST_C_LOAD(p, ptr)                                      \
    A64_LDRD("24", "[" ptr ", x15]") A64_LDRD("25", "[" ptr ", x13]") \
    A64_LDRD("26", "[" ptr ", x16]") A64_LDRD("27", "[" ptr "], #8")  \
    A64_ZIP1("27", "27", "24") A64_ZIP1("25", "25", "26")             \
    A64_ZIP1D(A64_LO(p), "27", "25") A64_ZIP2D(A64_HI(p), "27", "25")

/* Radix-4 butterfly of 4 lines l0..l3 for both halves of the lines.
 * Half 0: l0..l3 in pairs a0..a3, result o0..o3 in pairs 0, a1, 4, a2.
 * Half 1: l0..l3 in pairs b0..b3, result o0..o3 in pairs 7, b1, 5, 10.
 * Requires a3 == 10 and pairs 4, 5, 6, 7 free. */
#define A64_FIRST_R4(R, I, a0, a1, a2, b0, b1, b2)                    \
    A64_BF(R, I, 4, a1, a0, a1) A64_BF(R, I, 5, 10, a2, 10)           \
    A64_BF(R, I, 0, 4, 4, 5) A64_BFJ(R, I, a2, a1, a1, 10)            \
    A64_BF(R, I, 5, b1, b0, b1) A64_BF(R, I, 6, 3, b2, 3)             \
    A64_BF(R, I, 7, 5, 5, 6) A64_BFJ(R, I, 10, b1, b1, 3)

#define A64_FIRST_C(R, I)                                             \
    A64_FIRST_SETUP                                                   \
    "lsr x13, x13, #2\n"                                              \
    "lsl x15, x13, #1\n"                                              \
    "add x16, x15, x13\n"                                             \
    A64_FIRST_RUN                                                     \
    A64_FIRST_C_LOAD(8, "x0") A64_FIRST_C_LOAD(1, "x4")               \
    A64_FIRST_C_LOAD(9, "x2") A64_FIRST_C_LOAD(11, "x6")              \
    A64_FIRST_C_LOAD(3, "x1") A64_FIRST_C_LOAD(5, "x5")               \
    A64_FIRST_C_LOAD(7, "x3") A64_FIRST_C_LOAD(6, "x7")               \
    A64_FIRST_DFT_WEIGHT_TRANSPOSE(R, I)                              \
    A64_FIRST_R4(R, I, 0, 8, 9, 11, 1, 2)                             \
    A64_FIRST_DST(8)                                                  \
    A64_STP(R(0), I(0), "[x14]") A64_STP(R(7), I(7), "[x14, #32]")    \
    A64_STP(R(8), I(8), "[x14, #64]") A64_STP(R(1), I(1), "[x14, #96]") \
    A64_STP(R(4), I(4), "[x14, #128]") A64_STP(R(5), I(5), "[x14, #160]") \
    A64_STP(R(9), I(9), "[x14, #192]") A64_STP(R(10), I(10), "[x14, #224]") \
    A64_FIRST_NEXT

#define A64_DEFINE_FIRST(name, BODY, R, I)                                              \
    static TW_NOINLINE void name(const float* in_, size_t stride_, float* out_,         \
                                 const float* w_, size_t log2n_, size_t tile_begin_,    \
                                 size_t tile_end_, size_t run_, size_t skip_) {         \
        register const float* in __asm__("x0") = in_;                                   \
        register const float* w __asm__("x8") = w_;                                     \
        register size_t tile __asm__("x9") = tile_begin_;                               \
        register size_t tile_end __asm__("x10") = tile_end_;                            \
        register size_t log2n __asm__("x11") = log2n_;                                  \
        register float* out __asm__("x12") = out_;                                      \
        register size_t stride __asm__("x13") = stride_;                                \
        register size_t skip __asm__("x19") = skip_;                                    \
        register size_t run __asm__("x20") = run_;                                      \
        __asm__ volatile(BODY(R, I)                                                     \
                         : "+r"(in), "+r"(w), "+r"(tile), "+r"(log2n), "+r"(stride)     \
                         : "r"(out), "r"(tile_end), "r"(skip), "r"(run)                 \
                         : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x14", "x15",      \
                           "x16", "x17", "x21", A64_CLOBBER_V, "memory", "cc");         \
    }

A64_DEFINE_FIRST(tw_a64_first_a_fwd, A64_FIRST_A, A64_FWD_RE, A64_FWD_IM)
A64_DEFINE_FIRST(tw_a64_first_a_bwd, A64_FIRST_A, A64_BWD_RE, A64_BWD_IM)
A64_DEFINE_FIRST(tw_a64_first_b_fwd, A64_FIRST_B, A64_FWD_RE, A64_FWD_IM)
A64_DEFINE_FIRST(tw_a64_first_b_bwd, A64_FIRST_B, A64_BWD_RE, A64_BWD_IM)
A64_DEFINE_FIRST(tw_a64_first_c_fwd, A64_FIRST_C, A64_FWD_RE, A64_FWD_IM)
A64_DEFINE_FIRST(tw_a64_first_c_bwd, A64_FIRST_C, A64_BWD_RE, A64_BWD_IM)

/* ---- radix-8 passes ---------------------------------------------------- */

/*
 * One radix-8 butterfly on one half of 8 lines at x0..x7, which are advanced
 * to the next half. The weights w^4, w^2, w^6, w^1, w^5, w^3, w^7 and
 * sqrt(1/2) are in the lanes of v28..v31, see tw_weights.h.
 * STORE(pair, scratch pair, address) stores an output.
 */
#define A64_R8_HALF(R, I, STORE)                                                      \
    A64_LDP(R(8), I(8), "[x0]")                                                       \
    A64_LDP(R(12), I(12), "[x1]") A64_CMULE(R, I, 1, 12, "28", 0, 1)                  \
    A64_LDP(R(13), I(13), "[x2]") A64_CMULE(R, I, 9, 13, "28", 2, 3)                  \
    A64_LDP(R(12), I(12), "[x3]") A64_CMULE(R, I, 11, 12, "29", 0, 1)                 \
    A64_LDP(R(13), I(13), "[x4]") A64_CMULE(R, I, 3, 13, "29", 2, 3)                  \
    A64_LDP(R(12), I(12), "[x5]") A64_CMULE(R, I, 5, 12, "30", 0, 1)                  \
    A64_LDP(R(13), I(13), "[x6]") A64_CMULE(R, I, 7, 13, "30", 2, 3)                  \
    A64_LDP(R(12), I(12), "[x7]") A64_CMULE(R, I, 6, 12, "31", 0, 1)                  \
    A64_DFT8(R, I, "24", "25", A64_HMUL_V31E2)                                        \
    STORE(0, 8, "[x0], #32") STORE(1, 9, "[x1], #32") STORE(2, 10, "[x2], #32")       \
    STORE(3, 11, "[x3], #32") STORE(4, 12, "[x4], #32") STORE(5, 13, "[x5], #32")     \
    STORE(6, 8, "[x6], #32") STORE(7, 9, "[x7], #32")

/* Radix-8 pass: x0 data, x9 weights, x11 number of blocks, x12 stride s in
 * lines. Each block consists of 8*s lines and s butterflies. */
#define A64_R8(R, I, STORE)                         \
    "lsl x13, x12, #6\n"                            \
    "add x1, x0, x13\n"                             \
    "add x2, x1, x13\n"                             \
    "add x3, x2, x13\n"                             \
    "add x4, x3, x13\n"                             \
    "add x5, x4, x13\n"                             \
    "add x6, x5, x13\n"                             \
    "add x7, x6, x13\n"                             \
    "lsl x14, x13, #3\n"                            \
    "sub x13, x14, x13\n"                           \
    "1:\n"                                          \
    "mov x8, x9\n"                                  \
    "mov x10, x12\n"                                \
    "2:\n"                                          \
    "ldp q28, q29, [x8], #64\n"                     \
    "ldp q30, q31, [x8, #-32]\n"                    \
    A64_R8_HALF(R, I, STORE)                        \
    A64_R8_HALF(R, I, STORE)                        \
    "subs x10, x10, #1\n"                           \
    "b.ne 2b\n"                                     \
    "add x0, x0, x13\n"                             \
    "add x1, x1, x13\n"                             \
    "add x2, x2, x13\n"                             \
    "add x3, x3, x13\n"                             \
    "add x4, x4, x13\n"                             \
    "add x5, x5, x13\n"                             \
    "add x6, x6, x13\n"                             \
    "add x7, x7, x13\n"                             \
    "subs x11, x11, #1\n"                           \
    "b.ne 1b\n"

#define A64_STORE_SPLIT(p, t, addr) A64_STP(A64_FWD_RE(p), A64_FWD_IM(p), addr)
#define A64_STORE_INTERLEAVED(p, t, addr) A64_ST2(p, t, addr)

#define A64_DEFINE_R8(name, R, I, STORE)                                                 \
    static TW_NOINLINE void name(float* data_, const float* w_, size_t s_,               \
                                 size_t blocks_) {                                       \
        register float* data __asm__("x0") = data_;                                      \
        register const float* w __asm__("x9") = w_;                                      \
        register size_t blocks __asm__("x11") = blocks_;                                 \
        register size_t s __asm__("x12") = s_;                                           \
        __asm__ volatile(A64_R8(R, I, STORE)                                             \
                         : "+r"(data), "+r"(blocks)                                      \
                         : "r"(w), "r"(s)                                                \
                         : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x10", "x13", \
                           "x14", A64_CLOBBER_V, "memory", "cc");                        \
    }

A64_DEFINE_R8(tw_a64_r8, A64_FWD_RE, A64_FWD_IM, A64_STORE_SPLIT)
A64_DEFINE_R8(tw_a64_r8_last_fwd, A64_FWD_RE, A64_FWD_IM, A64_STORE_INTERLEAVED)
A64_DEFINE_R8(tw_a64_r8_last_bwd, A64_BWD_RE, A64_BWD_IM, A64_STORE_INTERLEAVED)

/*
 * Radix-8 pass with stride 1 and unit weights: x0 data, x11 number of blocks
 * of 8 consecutive lines. LOAD(pair, offset) / STORE(pair, scratch pair,
 * offset) access a half line at the given offset to the block.
 */
#define A64_T8_HALF(R, I, LOAD, STORE, o)                                             \
    LOAD(8, 0 + o) LOAD(1, 64 + o) LOAD(9, 128 + o) LOAD(11, 192 + o)                 \
    LOAD(3, 256 + o) LOAD(5, 320 + o) LOAD(7, 384 + o) LOAD(6, 448 + o)               \
    A64_DFT8(R, I, "24", "25", A64_HMUL_V31)                                          \
    STORE(0, 8, 0 + o) STORE(1, 9, 64 + o) STORE(2, 10, 128 + o) STORE(3, 11, 192 + o) \
    STORE(4, 12, 256 + o) STORE(5, 13, 320 + o) STORE(6, 8, 384 + o) STORE(7, 9, 448 + o)

#define A64_T8(R, I, LOAD, STORE)                   \
    A64_SET_V31_SQRT1_2                             \
    "1:\n"                                          \
    A64_T8_HALF(R, I, LOAD, STORE, 0)               \
    A64_T8_HALF(R, I, LOAD, STORE, 32)              \
    "add x0, x0, #512\n"                            \
    "subs x11, x11, #1\n"                           \
    "b.ne 1b\n"

#define A64_T8_LOAD(R, I, p, o) "ldp q" R(p) ", q" I(p) ", [x0, #(" #o ")]\n"
#define A64_T8_LOAD_FWD(p, o) A64_T8_LOAD(A64_FWD_RE, A64_FWD_IM, p, o)
#define A64_T8_LOAD_BWD(p, o) A64_T8_LOAD(A64_BWD_RE, A64_BWD_IM, p, o)
#define A64_T8_STORE_SPLIT(p, t, o) "stp q" A64_LO(p) ", q" A64_HI(p) ", [x0, #(" #o ")]\n"
#define A64_T8_STORE_INTERLEAVED(p, t, o) A64_ST2(p, t, "[x0, #(" #o ")]")

#define A64_DEFINE_T8(name, R, I, LOAD, STORE)                                        \
    static TW_NOINLINE void name(float* data_, size_t blocks_) {                      \
        register float* data __asm__("x0") = data_;                                   \
        register size_t blocks __asm__("x11") = blocks_;                              \
        __asm__ volatile(A64_T8(R, I, LOAD, STORE)                                    \
                         : "+r"(data), "+r"(blocks)                                   \
                         :                                                            \
                         : "x17", A64_CLOBBER_V, "memory", "cc");                     \
    }

A64_DEFINE_T8(tw_a64_t8, A64_FWD_RE, A64_FWD_IM, A64_T8_LOAD_FWD, A64_T8_STORE_SPLIT)
A64_DEFINE_T8(tw_a64_t8_last_fwd, A64_FWD_RE, A64_FWD_IM, A64_T8_LOAD_FWD, A64_T8_STORE_INTERLEAVED)
A64_DEFINE_T8(tw_a64_t8_last_bwd, A64_BWD_RE, A64_BWD_IM, A64_T8_LOAD_BWD, A64_T8_STORE_INTERLEAVED)

/* ---- transforms of size 8, 16 and 32 ----------------------------------- */

/* W_8^k for k = 0..3: real parts, imaginary parts. */
static const float tw_a64_w8[8] = {
    1.0f, 0.707106769f,  0.0f, -0.707106769f,
    0.0f, -0.707106769f, -1.0f, -0.707106769f,
};

/* W_16^(c*k) for k = 0..3: real parts, imaginary parts for c = 1, 2, 3. */
static const float tw_a64_w16[24] = {
    1.0f, 0.923879504f,  0.707106769f,  0.382683426f,
    0.0f, -0.382683426f, -0.707106769f, -0.923879504f,
    1.0f, 0.707106769f,  0.0f,          -0.707106769f,
    0.0f, -0.707106769f, -1.0f,         -0.707106769f,
    1.0f, 0.382683426f,  -0.707106769f, -0.923879504f,
    0.0f, -0.923879504f, -0.707106769f, 0.382683426f,
};

/* W_32^(c*k) for k = 0..3: real parts, imaginary parts for c = 1..7. This is
 * the first pass weight table (see tw_weights.h) of a transform of size 32. */
static const float tw_a64_w32[56] = {
    1.0f, 0.980785251f,  0.923879504f,  0.831469595f,
    0.0f, -0.195090324f, -0.382683426f, -0.555570245f,
    1.0f, 0.923879504f,  0.707106769f,  0.382683426f,
    0.0f, -0.382683426f, -0.707106769f, -0.923879504f,
    1.0f, 0.831469595f,  0.382683426f,  -0.195090324f,
    0.0f, -0.555570245f, -0.923879504f, -0.980785251f,
    1.0f, 0.707106769f,  0.0f,          -0.707106769f,
    0.0f, -0.707106769f, -1.0f,         -0.707106769f,
    1.0f, 0.555570245f,  -0.382683426f, -0.980785251f,
    0.0f, -0.831469595f, -0.923879504f, -0.195090324f,
    1.0f, 0.382683426f,  -0.707106769f, -0.923879504f,
    0.0f, -0.923879504f, -0.707106769f, 0.382683426f,
    1.0f, 0.195090324f,  -0.923879504f, -0.555570245f,
    0.0f, -0.980785251f, -0.382683426f, 0.831469595f,
};

/*
 * Size 8: x0 in (advanced), x8 weights, x12 out.
 *
 * Splits into the two sub-FFTs of size 4 over the even and odd outputs
 * (pairs 2 and 3, one input k per lane), then computes both at once with the
 * sub-FFT index in the lanes:
 *     v16/v18 = [s0 d0 s1 d1]   v17/v19 = [s2 d2 s3 d3]   (real/imaginary)
 */
#define A64_FFT8(R, I)                                                \
    A64_LD2(0, "[x0], #32") A64_LD2(1, "[x0]")                        \
    A64_LDP("24", "25", "[x8]")                                       \
    A64_BF(R, I, 2, 1, 0, 1)                                          \
    A64_CMUL(R, I, 3, 1, "24", "25")                                  \
    A64_ZIP1("16", R(2), R(3)) A64_ZIP2("17", R(2), R(3))             \
    A64_ZIP1("18", I(2), I(3)) A64_ZIP2("19", I(2), I(3))             \
    /* e = v20/v21, o = v22/v23, -re(o) = v26 */                      \
    A64_FADD("20", "16", "17") A64_FADD("21", "18", "19")             \
    A64_FSUB("22", "16", "17") A64_FSUB("23", "18", "19")             \
    A64_FSUB("26", "17", "16")                                        \
    /* f = [e0 e1 o0 o1], g = [e2 e3 -i*o2 -i*o3] */                  \
    A64_ZIP1D("27", "20", "22") A64_ZIP1D("28", "21", "23")           \
    A64_ZIP2D("29", "20", "23") A64_ZIP2D("30", "21", "26")           \
    A64_FADD(R(4), "27", "29") A64_FADD(I(4), "28", "30")             \
    A64_FSUB(R(5), "27", "29") A64_FSUB(I(5), "28", "30")             \
    A64_ST2(4, 6, "[x12]") A64_ST2(5, 7, "[x12, #32]")

/*
 * Size 16: x0 in, x8 weights, x12 out.
 *
 * Radix-4 split into 4 sub-FFTs of size 4 (one input k per lane), transpose
 * (one sub-FFT per lane), then the 4 sub-FFTs with a radix-4 butterfly.
 */
#define A64_FFT16(R, I)                                               \
    A64_LD2(4, "[x0], #32") A64_LD2(6, "[x0], #32")                   \
    A64_LD2(5, "[x0], #32") A64_LD2(7, "[x0]")                        \
    A64_LDP("24", "25", "[x8]") A64_LDP("26", "27", "[x8, #32]")      \
    A64_LDP("28", "29", "[x8, #64]")                                  \
    A64_BF(R, I, 0, 5, 4, 5) A64_BF(R, I, 1, 7, 6, 7)                 \
    A64_BF(R, I, 2, 0, 0, 1) A64_BFJ(R, I, 3, 5, 5, 7)                \
    /* z0@2 z1@5 z2@0 z3@3 */                                         \
    A64_CMUL(R, I, 4, 5, "24", "25") A64_CMUL(R, I, 6, 0, "26", "27") \
    A64_CMUL(R, I, 7, 3, "28", "29")                                  \
    A64_TRANSPOSE(R(2), R(4), R(6), R(7), "16", "17", "18", "19")     \
    A64_TRANSPOSE(I(2), I(4), I(6), I(7), "20", "21", "22", "23")     \
    A64_BF(R, I, 0, 6, 2, 6) A64_BF(R, I, 1, 7, 4, 7)                 \
    A64_BF(R, I, 2, 0, 0, 1) A64_BFJ(R, I, 3, 6, 6, 7)                \
    A64_ST2(2, 8, "[x12]") A64_ST2(6, 9, "[x12, #32]")                \
    A64_ST2(0, 10, "[x12, #64]") A64_ST2(3, 11, "[x12, #96]")

/* Size 32: x0 in, x8 weights, x12 out. A single tile of the first pass with
 * lanes k = 0..3 followed by the radix-4 butterfly of the lines. */
#define A64_FFT32(R, I)                                               \
    A64_SET_V31_SQRT1_2                                               \
    A64_LD2(8, "[x0], #32") A64_LD2(3, "[x0], #32")                   \
    A64_LD2(9, "[x0], #32") A64_LD2(7, "[x0], #32")                   \
    A64_LD2(1, "[x0], #32") A64_LD2(5, "[x0], #32")                   \
    A64_LD2(11, "[x0], #32") A64_LD2(6, "[x0]")                       \
    A64_FIRST_DFT_WEIGHT_TRANSPOSE(R, I)                              \
    A64_FIRST_R4(R, I, 0, 9, 8, 11, 2, 1)                             \
    A64_ST2(0, 12, "[x12]") A64_ST2(7, 13, "[x12, #32]")              \
    A64_ST2(9, 12, "[x12, #64]") A64_ST2(2, 13, "[x12, #96]")         \
    A64_ST2(4, 12, "[x12, #128]") A64_ST2(5, 13, "[x12, #160]")       \
    A64_ST2(8, 12, "[x12, #192]") A64_ST2(10, 13, "[x12, #224]")

#define A64_DEFINE_SMALL(name, BODY, R, I, weights)                                     \
    static TW_NOINLINE void name(const float* in_, float* out_) {                       \
        register const float* in __asm__("x0") = in_;                                   \
        register const float* w __asm__("x8") = weights;                                \
        register float* out __asm__("x12") = out_;                                      \
        __asm__ volatile(BODY(R, I)                                                     \
                         : "+r"(in), "+r"(w)                                            \
                         : "r"(out)                                                     \
                         : "x17", A64_CLOBBER_V, "memory", "cc");                       \
    }

A64_DEFINE_SMALL(tw_a64_fft8_fwd, A64_FFT8, A64_FWD_RE, A64_FWD_IM, tw_a64_w8)
A64_DEFINE_SMALL(tw_a64_fft8_bwd, A64_FFT8, A64_BWD_RE, A64_BWD_IM, tw_a64_w8)
A64_DEFINE_SMALL(tw_a64_fft16_fwd, A64_FFT16, A64_FWD_RE, A64_FWD_IM, tw_a64_w16)
A64_DEFINE_SMALL(tw_a64_fft16_bwd, A64_FFT16, A64_BWD_RE, A64_BWD_IM, tw_a64_w16)
A64_DEFINE_SMALL(tw_a64_fft32_fwd, A64_FFT32, A64_FWD_RE, A64_FWD_IM, tw_a64_w32)
A64_DEFINE_SMALL(tw_a64_fft32_bwd, A64_FFT32, A64_BWD_RE, A64_BWD_IM, tw_a64_w32)

/* ---- driver ------------------------------------------------------------ */

/*
 * First pass. Large transforms visit the tiles in the blocked order described
 * in tw_weights.h. The pieces of the input a chunk reads are copied into a
 * contiguous buffer on the stack (16 KiB), arranged like the input of a
 * small transform, on which the kernel runs: in the input the pieces are a
 * power of two apart and would compete for the same few lines of the cache.
 */
static void tw_a64_first(unsigned log2n, const float* w, const float* in, float* out, int backward) {
    typedef void kernel_fn(const float*, size_t, float*, const float*, size_t, size_t, size_t, size_t, size_t);
    const int variant = tw_first_variant(log2n);
    kernel_fn* const kernel = variant == TW_FIRST_A   ? (backward ? tw_a64_first_a_bwd : tw_a64_first_a_fwd)
                              : variant == TW_FIRST_B ? (backward ? tw_a64_first_b_bwd : tw_a64_first_b_fwd)
                                                      : (backward ? tw_a64_first_c_bwd : tw_a64_first_c_fwd);
    const size_t n = (size_t)1 << log2n;
    const size_t tiles = n / 32;
    if (!tw_first_blocked(log2n)) {
        kernel(in, n /* bytes: N / 8 complex numbers */, out, w, log2n, 0, tiles, tiles, 0);
        return;
    }
    /* The input consists of `streams` equally spaced streams. A chunk reads
     * a piece (`run` tiles) at `places` places, which are `distance` tiles
     * apart, of each stream. */
    const size_t streams = (size_t)8 << variant;
    const size_t tile_floats = 64 / streams;
    const size_t places = (size_t)1 << tw_first_log2_places(log2n);
    const size_t run = TW_FIRST_CHUNK_TILES / places;
    const size_t piece_floats = run * tile_floats;
    const size_t distance = tiles / places;
    float buffer[TW_FIRST_CHUNK_TILES * 64];
    for (size_t tile = 0; tile < distance; tile += run) {
        float* dst = buffer;
        for (size_t stream = 0; stream < streams; stream++) {
            const float* src = in + stream * (2 * n / streams) + tile * tile_floats;
            for (size_t place = 0; place < places; place++) {
                /* The piece of the chunk after the next one. With many
                 * places there are too many streams for hardware prefetchers. */
                if (places > 1) __builtin_prefetch(src + 2 * piece_floats);
                memcpy(dst, src, piece_floats * sizeof(float));
                src += distance * tile_floats;
                dst += piece_floats;
            }
        }
        kernel(buffer, TW_FIRST_CHUNK_TILES * 8 * sizeof(float), out, w, log2n, tile, tiles, run, distance - run);
        w += TW_FIRST_CHUNK_TILES * TW_FIRST_TILE_FLOATS;
    }
}

/* Blocks of at most this many lines (32 KiB) are processed pass by pass.
 * Larger blocks are split into 8 sub-blocks that are finished one after the
 * other, so that all but the last few passes run on data in the L1 cache. */
#ifndef TW_A64_LEAF_LINES
#define TW_A64_LEAF_LINES 512
#endif

/* Performs the radix-8 passes with strides s0 <= s < s_end on `lines` lines,
 * one pass after the other. `w` is the radix-8 weight table of the transform
 * and `s0` the stride of its first radix-8 pass (see tw_first_stride). */
static void tw_a64_passes(float* data, const float* w, size_t s0, size_t s_end, size_t lines) {
    size_t s = s0;
    if (s == 1 && s < s_end) {
        tw_a64_t8(data, lines / 8);
        s = 8;
    }
    for (; s < s_end; s *= 8) tw_a64_r8(data, w + tw_r8_weights_offset(s0, s), s, lines / (8 * s));
}

/* Performs all radix-8 passes with a stride below `lines` on a block of
 * `lines` lines. */
static void tw_a64_block(float* data, const float* w, size_t s0, size_t lines) {
    if (lines <= TW_A64_LEAF_LINES) {
        tw_a64_passes(data, w, s0, lines, lines);
        return;
    }
    const size_t s = lines / 8;
    for (unsigned i = 0; i < 8; i++) tw_a64_block(data + i * s * TW_LINE_FLOATS, w, s0, s);
    tw_a64_r8(data, w + tw_r8_weights_offset(s0, s), s, 1);
}

/* Out-of-place transform of size 2^log2n with the weights from tw_weights_fill. */
static void tw_a64_fft(unsigned log2n, const float* w, const float* in, float* out, int backward) {
    switch (log2n) {
        case 0:
        case 1:
        case 2: tw_small_fft(log2n, in, out, backward); return;
        case 3: (backward ? tw_a64_fft8_bwd : tw_a64_fft8_fwd)(in, out); return;
        case 4: (backward ? tw_a64_fft16_bwd : tw_a64_fft16_fwd)(in, out); return;
        case 5: (backward ? tw_a64_fft32_bwd : tw_a64_fft32_fwd)(in, out); return;
        default: break;
    }
    const size_t m = tw_lines(log2n);
    const size_t s0 = tw_first_stride(log2n);
    tw_a64_first(log2n, w, in, out, backward);
    w += (m / 4) * TW_FIRST_TILE_FLOATS;
    if (m == 8) {
        (backward ? tw_a64_t8_last_bwd : tw_a64_t8_last_fwd)(out, 1);
        return;
    }
    /* All passes but the last one, which also interleaves the output. */
    const size_t s = m / 8;
    if (m > TW_A64_LEAF_LINES)
        for (unsigned i = 0; i < 8; i++) tw_a64_block(out + i * s * TW_LINE_FLOATS, w, s0, s);
    else
        tw_a64_passes(out, w, s0, s, m);
    (backward ? tw_a64_r8_last_bwd : tw_a64_r8_last_fwd)(out, w + tw_r8_weights_offset(s0, s), s, 1);
}

#endif /* TW_ARM64_H */
