/* TwoFFT kernels.
 *
 * This file is a template: it is included once per instruction set after the matching
 * twofft_simd_*.h header and defines the functions twofft_<isa>_*.
 *
 * Algorithm overview (decimation in time, natural order in and out):
 *
 *   1. Leaf pass (in -> out). The input is read through L (8 .. 64) sequential streams.
 *      Small DFTs are computed "vertically", every vector lane working on a different
 *      sub-sequence. The result registers are transposed and stored so that each lane's
 *      results form a contiguous run of L points in the output. The runs are placed in bit
 *      reversed order; the bit reversal permutation is thereby merged into the first pass.
 *   2. Middle passes (in place on out). Radix 4 or 8 butterflies that combine 4 or 8
 *      adjacent sub-transforms of S points into one of 4S or 8S points.
 *   3. Final pass (in place on out). Like a middle pass, but stores interleaved complex.
 *
 * Between the leaf and the final pass the data in `out` is kept in "split block" format:
 * for every group of VW consecutive complex numbers, VW real parts are followed by
 * VW imaginary parts. All butterflies therefore work on plain vector loads and stores.
 *
 * Inverse transforms use the identity ifft(x) = swap(fft(swap(x))), where swap exchanges
 * real and imaginary parts. Swapping is free when (de)interleaving, so only the first and
 * the last pass exist in two versions.
 */

#define TWOFFT_CAT3_(a, b, c) a##b##_##c
#define TWOFFT_CAT3(a, b, c) TWOFFT_CAT3_(a, b, c)
#define FN(name) TWOFFT_CAT3(twofft_, TWOFFT_ISA, name)

/* Properties of this instantiation for the instruction set independent code. */
enum { FN(vw) = TWOFFT_VW, FN(ratio_twiddles) = TWOFFT_RATIO_TWIDDLES };

#define KINLINE TWOFFT_TARGET TWOFFT_FORCE_INLINE
#ifndef TWOFFT_KFUNC_ATTR
#define TWOFFT_KFUNC_ATTR
#endif
#define KFUNC TWOFFT_TARGET TWOFFT_KFUNC_ATTR static

/* ---------------------------------------------------------------------------------------
 * Vertical DFT codelets. All operate on complex vectors given as r[] / i[] arrays.
 * ------------------------------------------------------------------------------------ */

/* Multiplies (r, i) by the constant wr + i * wi. */
KINLINE void FN(cmulc)(vf *r, vf *i, const float wr, const float wi)
{
    const vf zr = *r, zi = *i, cr = v_set1(wr), ci = v_set1(wi);
    *r = v_fnmadd(v_mul(cr, zr), ci, zi);
    *i = v_fmadd(v_mul(cr, zi), ci, zr);
}

/* Forward DFT of the 4 values at slots 0, s, 2s, 3s; natural order in and out. */
KINLINE void FN(dft4)(vf *r, vf *i, const int s)
{
    const vf s0r = v_add(r[0], r[2 * s]), s0i = v_add(i[0], i[2 * s]);
    const vf d0r = v_sub(r[0], r[2 * s]), d0i = v_sub(i[0], i[2 * s]);
    const vf s1r = v_add(r[s], r[3 * s]), s1i = v_add(i[s], i[3 * s]);
    const vf d1r = v_sub(r[s], r[3 * s]), d1i = v_sub(i[s], i[3 * s]);
    r[0] = v_add(s0r, s1r);
    i[0] = v_add(s0i, s1i);
    r[s] = v_add(d0r, d1i);
    i[s] = v_sub(d0i, d1r);
    r[2 * s] = v_sub(s0r, s1r);
    i[2 * s] = v_sub(s0i, s1i);
    r[3 * s] = v_sub(d0r, d1i);
    i[3 * s] = v_add(d0i, d1r);
}

/* Forward DFT of 8 values in natural order. Output k is left in slot TWOFFT_SLOT8(k). */
#define TWOFFT_SLOT8(k) ((((k)&3) << 1) + ((k) >> 2))
KINLINE void FN(dft8)(vf *r, vf *i)
{
    const vf c = v_set1(TWOFFT_SQRT1_2);
    vf tr, ti, ur, ui;
    FN(dft4)(r, i, 2);         /* even inputs: E[k] in slot 2k */
    FN(dft4)(r + 1, i + 1, 2); /* odd inputs:  O[k] in slot 2k+1 */
    /* k = 0: E +- O */
    tr = r[1], ti = i[1];
    r[1] = v_sub(r[0], tr), i[1] = v_sub(i[0], ti);
    r[0] = v_add(r[0], tr), i[0] = v_add(i[0], ti);
    /* k = 1: E +- w8 O, w8 = (1 - i) / sqrt(2) */
    ur = v_add(r[3], i[3]), ui = v_sub(i[3], r[3]);
    r[3] = v_fnmadd(r[2], c, ur), i[3] = v_fnmadd(i[2], c, ui);
    r[2] = v_fmadd(r[2], c, ur), i[2] = v_fmadd(i[2], c, ui);
    /* k = 2: E +- (-i) O */
    tr = i[5], ti = r[5];
    r[5] = v_sub(r[4], tr), i[5] = v_add(i[4], ti);
    r[4] = v_add(r[4], tr), i[4] = v_sub(i[4], ti);
    /* k = 3: E +- w8^3 O, w8^3 = (-1 - i) / sqrt(2) */
    ur = v_sub(i[7], r[7]), ui = v_add(r[7], i[7]);
    r[7] = v_fnmadd(r[6], c, ur), i[7] = v_fmadd(i[6], c, ui);
    r[6] = v_fmadd(r[6], c, ur), i[6] = v_fnmadd(i[6], c, ui);
}

/* ---------------------------------------------------------------------------------------
 * Radix 4 and radix 8 butterflies on split block data.
 *
 * A butterfly combines 4 (8) sub-transforms of S points into one of 4S (8S) points.
 * Sub-block p of a block holds the transform of the sub-sequence with offset bitrev(p).
 * mode: 0 = store split blocks, 1 / 2 = store interleaved for a forward / inverse transform.
 * ------------------------------------------------------------------------------------ */

#define TWOFFT_PASS_MIDDLE 0
#define TWOFFT_PASS_FINAL_FWD 1
#define TWOFFT_PASS_FINAL_INV 2

KINLINE void FN(pass_store)(float *x, const vf re, const vf im, const int mode)
{
    if (mode == TWOFFT_PASS_MIDDLE) {
        v_store(x, re);
        v_store(x + TWOFFT_VW, im);
    } else if (mode == TWOFFT_PASS_FINAL_FWD) {
        V_STORE2(x, re, im);
    } else {
        V_STORE2(x, im, re);
    }
}

#if TWOFFT_RATIO_TWIDDLES

/* Number of twiddle vectors per vector position of a radix 4 / radix 8 butterfly. */
#define TWOFFT_NW4 4
#define TWOFFT_NW8 8

/* (a, b) <- (a + w b, a - w b) for w = c (1 + i r): six fused multiply-adds. */
KINLINE void FN(tbf)(vf *ar, vf *ai, vf *br, vf *bi, const vf c, const vf r)
{
    const vf u = v_fnmadd(*br, r, *bi), v = v_fmadd(*bi, r, *br);
    *br = v_fnmadd(*ar, c, u);
    *bi = v_fnmadd(*ai, c, v);
    *ar = v_fmadd(*ar, c, u);
    *ai = v_fmadd(*ai, c, v);
}

/* (a, b) <- (a - i w b, a + i w b) */
KINLINE void FN(tbf_i)(vf *ar, vf *ai, vf *br, vf *bi, const vf c, const vf r)
{
    const vf u = v_fnmadd(*br, r, *bi), v = v_fmadd(*bi, r, *br);
    *br = v_fnmadd(*ar, c, v);
    *bi = v_fmadd(*ai, c, u);
    *ar = v_fmadd(*ar, c, v);
    *ai = v_fnmadd(*ai, c, u);
}

/* w: the twiddles w^2, w (w = exp(-2 pi i (position + lane) / 4S)) as 4 vectors (c, r, c, r). */
KINLINE void FN(bf4)(float *x0, float *x1, float *x2, float *x3, const vf *w, const int mode)
{
    enum { VW = TWOFFT_VW };
    vf ar = v_load(x0), ai = v_load(x0 + VW);
    vf br = v_load(x1), bi = v_load(x1 + VW);
    vf cr = v_load(x2), ci = v_load(x2 + VW);
    vf dr = v_load(x3), di = v_load(x3 + VW);
    /* two radix 2 levels; sub-blocks hold offsets 0, 2, 1, 3 */
    FN(tbf)(&ar, &ai, &br, &bi, w[0], w[1]);
    FN(tbf)(&cr, &ci, &dr, &di, w[0], w[1]);
    FN(tbf)(&ar, &ai, &cr, &ci, w[2], w[3]);
    FN(tbf_i)(&br, &bi, &dr, &di, w[2], w[3]);
    FN(pass_store)(x0, ar, ai, mode);
    FN(pass_store)(x1, br, bi, mode);
    FN(pass_store)(x2, cr, ci, mode);
    FN(pass_store)(x3, dr, di, mode);
}

/* x: position in the first sub-block, dist: distance of the sub-blocks in floats,
 * w: the twiddles w^4, w^2, w, w8 w (w = exp(-2 pi i (position + lane) / 8S),
 * w8 = exp(-2 pi i / 8)) as 8 vectors. */
KINLINE void FN(bf8)(float *x, const size_t dist, const vf *w, const int mode)
{
    enum { VW = TWOFFT_VW };
    vf r[8], i[8];
    unsigned j;
    for (j = 0; j < 8; j++) {
        r[j] = v_load(x + dist * j);
        i[j] = v_load(x + dist * j + VW);
    }
    /* three radix 2 levels; sub-block q holds offset bitrev3(q) */
    FN(tbf)(r + 0, i + 0, r + 1, i + 1, w[0], w[1]);
    FN(tbf)(r + 2, i + 2, r + 3, i + 3, w[0], w[1]);
    FN(tbf)(r + 4, i + 4, r + 5, i + 5, w[0], w[1]);
    FN(tbf)(r + 6, i + 6, r + 7, i + 7, w[0], w[1]);
    FN(tbf)(r + 0, i + 0, r + 2, i + 2, w[2], w[3]);
    FN(tbf_i)(r + 1, i + 1, r + 3, i + 3, w[2], w[3]);
    FN(tbf)(r + 4, i + 4, r + 6, i + 6, w[2], w[3]);
    FN(tbf_i)(r + 5, i + 5, r + 7, i + 7, w[2], w[3]);
    FN(tbf)(r + 0, i + 0, r + 4, i + 4, w[4], w[5]);
    FN(tbf)(r + 1, i + 1, r + 5, i + 5, w[6], w[7]);
    FN(tbf_i)(r + 2, i + 2, r + 6, i + 6, w[4], w[5]);
    FN(tbf_i)(r + 3, i + 3, r + 7, i + 7, w[6], w[7]);
    for (j = 0; j < 8; j++)
        FN(pass_store)(x + dist * j, r[j], i[j], mode);
}

#else /* !TWOFFT_RATIO_TWIDDLES */

#define TWOFFT_NW4 6
#define TWOFFT_NW8 14

/* w: the twiddles w^1, w^2, w^3 (w = exp(-2 pi i (position + lane) / 4S)) as 6 vectors
 * (re, im, re, im, re, im). */
KINLINE void FN(bf4)(float *x0, float *x1, float *x2, float *x3, const vf *w, const int mode)
{
    enum { VW = TWOFFT_VW };
    const vf ar = v_load(x0), ai = v_load(x0 + VW);
    const vf br = v_load(x1), bi = v_load(x1 + VW);
    const vf cr = v_load(x2), ci = v_load(x2 + VW);
    const vf dr = v_load(x3), di = v_load(x3 + VW);
    /* sub-blocks hold offsets 0, 2, 1, 3 */
    const vf t2r = v_fnmadd(v_mul(w[2], br), w[3], bi), t2i = v_fmadd(v_mul(w[2], bi), w[3], br);
    const vf t1r = v_fnmadd(v_mul(w[0], cr), w[1], ci), t1i = v_fmadd(v_mul(w[0], ci), w[1], cr);
    const vf t3r = v_fnmadd(v_mul(w[4], dr), w[5], di), t3i = v_fmadd(v_mul(w[4], di), w[5], dr);
    const vf s02r = v_add(ar, t2r), s02i = v_add(ai, t2i);
    const vf d02r = v_sub(ar, t2r), d02i = v_sub(ai, t2i);
    const vf s13r = v_add(t1r, t3r), s13i = v_add(t1i, t3i);
    const vf d13r = v_sub(t1r, t3r), d13i = v_sub(t1i, t3i);
    FN(pass_store)(x0, v_add(s02r, s13r), v_add(s02i, s13i), mode);
    FN(pass_store)(x1, v_add(d02r, d13i), v_sub(d02i, d13r), mode);
    FN(pass_store)(x2, v_sub(s02r, s13r), v_sub(s02i, s13i), mode);
    FN(pass_store)(x3, v_sub(d02r, d13i), v_add(d02i, d13r), mode);
}

/* x: position in the first sub-block, dist: distance of the sub-blocks in floats,
 * w: the twiddles w^1 .. w^7 (w = exp(-2 pi i (position + lane) / 8S)) as 14 vectors. */
KINLINE void FN(bf8)(float *x, const size_t dist, const vf *w, const int mode)
{
    enum { VW = TWOFFT_VW };
    vf r[8], i[8];
    unsigned j;
    /* sub-block q holds offset bitrev3(q) */
    for (j = 0; j < 8; j++) {
        const float *src = x + dist * twofft_bitrev(j, 3);
        r[j] = v_load(src);
        i[j] = v_load(src + VW);
    }
    for (j = 1; j < 8; j++) {
        const vf wr = w[2 * j - 2], wi = w[2 * j - 1], zr = r[j], zi = i[j];
        r[j] = v_fnmadd(v_mul(wr, zr), wi, zi);
        i[j] = v_fmadd(v_mul(wr, zi), wi, zr);
    }
    FN(dft8)(r, i);
    for (j = 0; j < 8; j++)
        FN(pass_store)(x + dist * j, r[TWOFFT_SLOT8(j)], i[TWOFFT_SLOT8(j)], mode);
}

#endif /* TWOFFT_RATIO_TWIDDLES */

/* ---------------------------------------------------------------------------------------
 * Leaf pass: bit reversal + first levels, interleaved `in` -> split block `out`.
 *
 * One tile reads L = L1 * R input streams and produces VW runs of L output points.
 * Stage 1: for each of the R groups of L1 streams a vertical DFT of size L1 whose results
 *          are transposed and stored into the VW runs (as sub-transforms of L1 points).
 * Stage 2: (R > 1) a radix R butterfly combines the R sub-transforms of every run; its
 *          twiddles are constant and kept in registers where possible.
 * Running stage 2 right after stage 1 keeps a tile in registers and store buffers, and
 * stage 1 alone never needs more registers than a size 8 DFT.
 * ------------------------------------------------------------------------------------ */

KINLINE void FN(leaf_impl)(const float *restrict in, float *restrict out, const unsigned n,
                           const unsigned l1log, const unsigned rlog, const float *restrict tw, const int inv)
{
    enum { VW = TWOFFT_VW, MAXL1 = 8, MAXHOIST = 14 };
    const size_t L1 = (size_t)1 << l1log, R = (size_t)1 << rlog, L = L1 * R;
    const size_t ntiles = ((size_t)1 << n) / (L * VW);
    const size_t gdist = (size_t)2 << (n - l1log - rlog); /* distance of adjacent streams */
    const size_t sdist = gdist << rlog;                   /* distance of the streams of a group */
    const unsigned nw = rlog == 2 ? TWOFFT_NW4 : TWOFFT_NW8; /* twiddle vectors per position */
    const unsigned ntw = (unsigned)(nw * (L1 / VW));         /* twiddle vectors of stage 2 */
    vf hoisted[MAXHOIST];
    float *o[VW];
    size_t rev = 0; /* bit reversed tile index */
    size_t m;
    unsigned a, c, g, j;
    for (c = 0; c < VW; c++)
        o[c] = out + 2 * L * ntiles * twofft_bitrev(TWOFFT_LANE_ELEM(c), TWOFFT_VW_LOG2);
    if (rlog != 0 && ntw <= MAXHOIST)
        for (j = 0; j < ntw; j++)
            hoisted[j] = v_load(tw + VW * j);
    for (m = 0; m < ntiles; m++) {
        const float *p = in + 2 * VW * m;
        const size_t off = 2 * L * rev;
        for (g = 0; g < R; g++) {
            const float *pg = p + g * gdist;
            const size_t sub = off + 2 * L1 * twofft_bitrev(g, rlog);
            vf r[MAXL1], i[MAXL1];
            for (a = 0; a < L1; a++) {
                if (inv)
                    V_LOAD2(pg + a * sdist, i[a], r[a]);
                else
                    V_LOAD2(pg + a * sdist, r[a], i[a]);
            }
            if (l1log == 2)
                FN(dft4)(r, i, 1);
            else
                FN(dft8)(r, i);
            for (j = 0; j < L1; j += VW) {
                vf tr[VW], ti[VW];
                for (c = 0; c < VW; c++) {
                    const unsigned k = j + c;
                    const unsigned slot = l1log == 2 ? k : TWOFFT_SLOT8(k);
                    tr[c] = r[slot];
                    ti[c] = i[slot];
                }
                V_TRANSPOSE(tr);
                V_TRANSPOSE(ti);
                for (c = 0; c < VW; c++) {
                    v_store(o[c] + sub + 2 * j, tr[c]);
                    v_store(o[c] + sub + 2 * j + VW, ti[c]);
                }
            }
        }
        if (rlog != 0) {
            for (c = 0; c < VW; c++) {
                float *x = o[c] + off;
                for (j = 0; j < L1; j += VW) {
                    vf loaded[TWOFFT_NW8];
                    const vf *w = hoisted + nw * (j / VW);
                    unsigned q;
                    if (ntw > MAXHOIST) {
                        for (q = 0; q < nw; q++)
                            loaded[q] = v_load(tw + VW * (nw * (j / VW) + q));
                        w = loaded;
                    }
                    if (rlog == 2)
                        FN(bf4)(x + 2 * j, x + 2 * j + 2 * L1, x + 2 * j + 4 * L1, x + 2 * j + 6 * L1, w,
                                TWOFFT_PASS_MIDDLE);
                    else
                        FN(bf8)(x + 2 * j, 2 * L1, w, TWOFFT_PASS_MIDDLE);
                }
            }
        }
        rev = twofft_bitrev_next(rev, ntiles);
    }
}

/* The leaf variants: twofft_<isa>_leaf_<L1>x<R>_{fwd,inv} */
#define TWOFFT_DEFINE_LEAF(name, l1log, rlog)                                                                \
    KFUNC void FN(name##_fwd)(const float *restrict in, float *restrict out, unsigned n, const float *tw)   \
    {                                                                                                        \
        FN(leaf_impl)(in, out, n, l1log, rlog, tw, 0);                                                       \
    }                                                                                                        \
    KFUNC void FN(name##_inv)(const float *restrict in, float *restrict out, unsigned n, const float *tw)   \
    {                                                                                                        \
        FN(leaf_impl)(in, out, n, l1log, rlog, tw, 1);                                                       \
    }
#if TWOFFT_VW <= 4
TWOFFT_DEFINE_LEAF(leaf_4x4, 2, 2)
TWOFFT_DEFINE_LEAF(leaf_4x8, 2, 3)
#endif
TWOFFT_DEFINE_LEAF(leaf_8x1, 3, 0)
TWOFFT_DEFINE_LEAF(leaf_8x4, 3, 2)
TWOFFT_DEFINE_LEAF(leaf_8x8, 3, 3)
#undef TWOFFT_DEFINE_LEAF

KFUNC void FN(leaf)(const twofft_plan *pl, const float *restrict in, float *restrict out, int inv)
{
    const unsigned n = pl->log2n;
    switch (pl->leaf_l1_log * 4 + pl->leaf_radix_log) {
#if TWOFFT_VW <= 4
    case 2 * 4 + 2:
        if (inv) FN(leaf_4x4_inv)(in, out, n, pl->leaf_tw); else FN(leaf_4x4_fwd)(in, out, n, pl->leaf_tw);
        break;
    case 2 * 4 + 3:
        if (inv) FN(leaf_4x8_inv)(in, out, n, pl->leaf_tw); else FN(leaf_4x8_fwd)(in, out, n, pl->leaf_tw);
        break;
#endif
    case 3 * 4 + 2:
        if (inv) FN(leaf_8x4_inv)(in, out, n, pl->leaf_tw); else FN(leaf_8x4_fwd)(in, out, n, pl->leaf_tw);
        break;
    case 3 * 4 + 3:
        if (inv) FN(leaf_8x8_inv)(in, out, n, pl->leaf_tw); else FN(leaf_8x8_fwd)(in, out, n, pl->leaf_tw);
        break;
    default:
        if (inv) FN(leaf_8x1_inv)(in, out, n, pl->leaf_tw); else FN(leaf_8x1_fwd)(in, out, n, pl->leaf_tw);
        break;
    }
}

/* ---------------------------------------------------------------------------------------
 * Radix 4 and radix 8 passes over nblk blocks of 4S (8S) points.
 * tw: for every vector position the twiddles of the butterfly, see twofft_make_twiddles().
 * ------------------------------------------------------------------------------------ */

KINLINE void FN(pass4_impl)(float *restrict x, const size_t S, const size_t nblk,
                            const float *restrict tw, const int mode)
{
    enum { VW = TWOFFT_VW };
    size_t b, p;
    unsigned j;
    for (b = 0; b < nblk; b++) {
        float *x0 = x + 8 * S * b, *x1 = x0 + 2 * S, *x2 = x0 + 4 * S, *x3 = x0 + 6 * S;
        const float *t = tw;
        for (p = 0; p < S; p += VW) {
            vf w[TWOFFT_NW4];
            for (j = 0; j < TWOFFT_NW4; j++)
                w[j] = v_load(t + VW * j);
            FN(bf4)(x0, x1, x2, x3, w, mode);
            x0 += 2 * VW, x1 += 2 * VW, x2 += 2 * VW, x3 += 2 * VW, t += TWOFFT_NW4 * VW;
        }
    }
}

KINLINE void FN(pass8_impl)(float *restrict x, const size_t S, const size_t nblk,
                            const float *restrict tw, const int mode)
{
    enum { VW = TWOFFT_VW };
    size_t b, p;
    unsigned j;
    for (b = 0; b < nblk; b++) {
        float *xb = x + 16 * S * b;
        const float *t = tw;
        for (p = 0; p < S; p += VW) {
            vf w[TWOFFT_NW8];
            for (j = 0; j < TWOFFT_NW8; j++)
                w[j] = v_load(t + VW * j);
            FN(bf8)(xb, 2 * S, w, mode);
            xb += 2 * VW, t += TWOFFT_NW8 * VW;
        }
    }
}

KFUNC void FN(pass4)(float *restrict x, size_t S, size_t nblk, const float *restrict tw)
{
    FN(pass4_impl)(x, S, nblk, tw, TWOFFT_PASS_MIDDLE);
}
KFUNC void FN(pass8)(float *restrict x, size_t S, size_t nblk, const float *restrict tw)
{
    FN(pass8_impl)(x, S, nblk, tw, TWOFFT_PASS_MIDDLE);
}
KFUNC void FN(final4_fwd)(float *restrict x, size_t S, const float *restrict tw)
{
    FN(pass4_impl)(x, S, 1, tw, TWOFFT_PASS_FINAL_FWD);
}
KFUNC void FN(final4_inv)(float *restrict x, size_t S, const float *restrict tw)
{
    FN(pass4_impl)(x, S, 1, tw, TWOFFT_PASS_FINAL_INV);
}
KFUNC void FN(final8_fwd)(float *restrict x, size_t S, const float *restrict tw)
{
    FN(pass8_impl)(x, S, 1, tw, TWOFFT_PASS_FINAL_FWD);
}
KFUNC void FN(final8_inv)(float *restrict x, size_t S, const float *restrict tw)
{
    FN(pass8_impl)(x, S, 1, tw, TWOFFT_PASS_FINAL_INV);
}

/* ---------------------------------------------------------------------------------------
 * Driver for planned sizes.
 * ------------------------------------------------------------------------------------ */

/* Applies passes 0 .. k-1 to the block of 2^(log2 size after pass k-1) points at x.
 * Large blocks are processed depth first so that the lower passes stay in cache. */
KFUNC void FN(middle)(const twofft_plan *pl, float *x, unsigned k)
{
    unsigned j;
    if (k == 0)
        return;
    if (pl->size_log[k - 1] <= TWOFFT_BLOCK_LOG2) {
        for (j = 0; j < k; j++) {
            const size_t S = (size_t)1 << (pl->size_log[j] - pl->radix_log[j]);
            const size_t nblk = (size_t)1 << (pl->size_log[k - 1] - pl->size_log[j]);
            if (pl->radix_log[j] == 2)
                FN(pass4)(x, S, nblk, pl->tw[j]);
            else
                FN(pass8)(x, S, nblk, pl->tw[j]);
        }
    } else {
        const size_t S = (size_t)1 << (pl->size_log[k - 1] - pl->radix_log[k - 1]);
        const size_t r = (size_t)1 << pl->radix_log[k - 1];
        size_t q;
        for (q = 0; q < r; q++)
            FN(middle)(pl, x + 2 * S * q, k - 1);
        if (pl->radix_log[k - 1] == 2)
            FN(pass4)(x, S, 1, pl->tw[k - 1]);
        else
            FN(pass8)(x, S, 1, pl->tw[k - 1]);
    }
}

KFUNC void FN(transform)(const twofft_plan *pl, const float *restrict in, float *restrict out, int inv)
{
    const unsigned last = pl->npasses - 1u;
    const size_t S = (size_t)1 << (pl->size_log[last] - pl->radix_log[last]);
    size_t q;
    if (TWOFFT_STAGE_ENABLED(1))
        FN(leaf)(pl, in, out, inv);
    if (TWOFFT_STAGE_ENABLED(2))
        for (q = 0; q < ((size_t)1 << pl->radix_log[last]); q++)
            FN(middle)(pl, out + 2 * S * q, last);
    if (!TWOFFT_STAGE_ENABLED(4))
        return;
    if (pl->radix_log[last] == 2) {
        if (inv)
            FN(final4_inv)(out, S, pl->tw[last]);
        else
            FN(final4_fwd)(out, S, pl->tw[last]);
    } else {
        if (inv)
            FN(final8_inv)(out, S, pl->tw[last]);
        else
            FN(final8_fwd)(out, S, pl->tw[last]);
    }
}

/* ---------------------------------------------------------------------------------------
 * Fixed size transforms for N = 8 and N = 16 (4 lane instruction sets only).
 * ------------------------------------------------------------------------------------ */

#if TWOFFT_VW == 4

KINLINE void FN(fft8_impl)(const float *restrict in, float *restrict out, const int inv)
{
    vf ar, ai, br, bi;
    if (inv) {
        V_LOAD2(in, ai, ar);
        V_LOAD2(in + 8, bi, br);
    } else {
        V_LOAD2(in, ar, ai);
        V_LOAD2(in + 8, br, bi);
    }
    {
        /* radix 2 decimation in frequency: s feeds the even, t the odd outputs */
        const vf wr = v_load(twofft_w8[0]), wi = v_load(twofft_w8[1]);
        const vf sr = v_add(ar, br), si = v_add(ai, bi);
        const vf dr = v_sub(ar, br), di = v_sub(ai, bi);
        const vf tr = v_fnmadd(v_mul(dr, wr), di, wi), ti = v_fmadd(v_mul(dr, wi), di, wr);
        /* two 4 point DFTs along the lanes */
        const vf er = v4_lo64(sr, tr), ei = v4_lo64(si, ti);
        const vf fr = v4_hi64(sr, tr), fi = v4_hi64(si, ti);
        const vf gr = v_add(er, fr), gi = v_add(ei, fi);
        const vf hr = v_sub(er, fr), hi = v_sub(ei, fi);
        const vf pr = v4_even(gr, hr), pi = v4_even(gi, hi);
        const vf qr = v4_odd(gr, hr), qi = v4_odd(gi, hi);
        const vf ur = v4_lohi(qr, qi), ui = v4_lohi(qi, v_neg(qr)); /* upper lanes times -i */
        const vf x0r = v_add(pr, ur), x0i = v_add(pi, ui);
        const vf x1r = v_sub(pr, ur), x1i = v_sub(pi, ui);
        if (inv) {
            V_STORE2(out, x0i, x0r);
            V_STORE2(out + 8, x1i, x1r);
        } else {
            V_STORE2(out, x0r, x0i);
            V_STORE2(out + 8, x1r, x1i);
        }
    }
}

KINLINE void FN(fft16_impl)(const float *restrict in, float *restrict out, const int inv)
{
    vf r[4], i[4];
    unsigned j;
    for (j = 0; j < 4; j++) {
        if (inv)
            V_LOAD2(in + 8 * j, i[j], r[j]);
        else
            V_LOAD2(in + 8 * j, r[j], i[j]);
    }
    /* radix 4 decimation in frequency: butterflies across the 4 vectors ... */
    FN(dft4)(r, i, 1);
    V_TRANSPOSE(r);
    V_TRANSPOSE(i);
    /* ... twiddles w16^(n q) (n: vector, q: lane) ... */
    for (j = 1; j < 4; j++) {
        const vf wr = v_load(twofft_w16[j - 1][0]), wi = v_load(twofft_w16[j - 1][1]);
        const vf zr = r[j], zi = i[j];
        r[j] = v_fnmadd(v_mul(zr, wr), zi, wi);
        i[j] = v_fmadd(v_mul(zr, wi), zi, wr);
    }
    /* ... and 4 point DFTs yielding X[4k + q] in lane q of vector k. */
    FN(dft4)(r, i, 1);
    for (j = 0; j < 4; j++) {
        if (inv)
            V_STORE2(out + 8 * j, i[j], r[j]);
        else
            V_STORE2(out + 8 * j, r[j], i[j]);
    }
}

KFUNC void FN(fft8_fwd)(const float *restrict in, float *restrict out) { FN(fft8_impl)(in, out, 0); }
KFUNC void FN(fft8_inv)(const float *restrict in, float *restrict out) { FN(fft8_impl)(in, out, 1); }
KFUNC void FN(fft16_fwd)(const float *restrict in, float *restrict out) { FN(fft16_impl)(in, out, 0); }
KFUNC void FN(fft16_inv)(const float *restrict in, float *restrict out) { FN(fft16_impl)(in, out, 1); }

#endif /* TWOFFT_VW == 4 */

#undef TWOFFT_NW4
#undef TWOFFT_NW8
#undef FN
#undef KINLINE
#undef KFUNC
#undef TWOFFT_CAT3
#undef TWOFFT_CAT3_
