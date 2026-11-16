/* TwoFFT kernels.
 *
 * This file is a template: it is included once per instruction set after the matching
 * twofft_simd_*.h header and defines the functions twofft_<isa>_*.
 *
 * Algorithm overview (decimation in time, natural order in and out):
 *
 *   1. Leaf pass (in -> out). The input is read through L = 8 or 16 sequential streams.
 *      A size L DFT is computed "vertically", every vector lane working on a different
 *      sub-sequence. The result registers are transposed and stored so that each lane's L
 *      results form a contiguous run in the output. The runs are placed in bit reversed
 *      order; the bit reversal permutation is thereby merged into the first pass.
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

#define KINLINE TWOFFT_TARGET TWOFFT_FORCE_INLINE
#define KFUNC TWOFFT_TARGET static

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

/* Forward DFT of 16 values in natural order. Output k is left in slot TWOFFT_SLOT16(k). */
#define TWOFFT_SLOT16(k) ((((k)&3) << 2) + ((k) >> 2))
KINLINE void FN(dft16)(vf *r, vf *i)
{
    vf t;
    /* Z_j = DFT4(x[j], x[j+4], x[j+8], x[j+12]); Z_j[k] in slot j + 4k */
    FN(dft4)(r + 0, i + 0, 4);
    FN(dft4)(r + 1, i + 1, 4);
    FN(dft4)(r + 2, i + 2, 4);
    FN(dft4)(r + 3, i + 3, 4);
    /* Z_j[k] *= w16^(j k) */
    FN(cmulc)(r + 5, i + 5, TWOFFT_COS_PI_8, -TWOFFT_SIN_PI_8);  /* w^1 */
    FN(cmulc)(r + 6, i + 6, TWOFFT_SQRT1_2, -TWOFFT_SQRT1_2);    /* w^2 */
    FN(cmulc)(r + 7, i + 7, TWOFFT_SIN_PI_8, -TWOFFT_COS_PI_8);  /* w^3 */
    FN(cmulc)(r + 9, i + 9, TWOFFT_SQRT1_2, -TWOFFT_SQRT1_2);    /* w^2 */
    t = r[10], r[10] = i[10], i[10] = v_neg(t);                  /* w^4 = -i */
    FN(cmulc)(r + 11, i + 11, -TWOFFT_SQRT1_2, -TWOFFT_SQRT1_2); /* w^6 */
    FN(cmulc)(r + 13, i + 13, TWOFFT_SIN_PI_8, -TWOFFT_COS_PI_8);  /* w^3 */
    FN(cmulc)(r + 14, i + 14, -TWOFFT_SQRT1_2, -TWOFFT_SQRT1_2);   /* w^6 */
    FN(cmulc)(r + 15, i + 15, -TWOFFT_COS_PI_8, TWOFFT_SIN_PI_8);  /* w^9 */
    /* X[k + 4q] = DFT4 over j of Z_j[k]; left in slot 4k + q */
    FN(dft4)(r + 0, i + 0, 1);
    FN(dft4)(r + 4, i + 4, 1);
    FN(dft4)(r + 8, i + 8, 1);
    FN(dft4)(r + 12, i + 12, 1);
}

/* ---------------------------------------------------------------------------------------
 * Leaf pass: bit reversal + first 3 or 4 levels, interleaved `in` -> split block `out`.
 * ------------------------------------------------------------------------------------ */

KINLINE void FN(leaf_impl)(const float *restrict in, float *restrict out, const unsigned n,
                           const unsigned llog, const int inv)
{
    enum { VW = TWOFFT_VW, L = 16 };
    const size_t len = (size_t)1 << llog;                /* L: DFT size of the leaf */
    const size_t ntiles = ((size_t)1 << n) / (len * VW); /* one tile = L streams x VW lanes */
    const size_t stride = (size_t)2 << (n - llog);       /* distance of the input streams */
    float *o[VW];
    size_t rev = 0; /* bit reversed tile index */
    size_t m;
    unsigned a, c, j;
    for (c = 0; c < VW; c++)
        o[c] = out + 2 * len * ntiles * twofft_bitrev(TWOFFT_LANE_ELEM(c), TWOFFT_VW_LOG2);
    for (m = 0; m < ntiles; m++) {
        const float *p = in + 2 * VW * m;
        const size_t off = 2 * len * rev;
        vf r[L], i[L];
        for (a = 0; a < len; a++) {
            if (inv)
                V_LOAD2(p + a * stride, i[a], r[a]);
            else
                V_LOAD2(p + a * stride, r[a], i[a]);
        }
        if (llog == 3)
            FN(dft8)(r, i);
        else
            FN(dft16)(r, i);
        for (j = 0; j < len; j += VW) {
            vf tr[VW], ti[VW];
            for (c = 0; c < VW; c++) {
                const unsigned k = j + c;
                const unsigned slot = ((k & 3) << (llog - 2)) + (k >> 2);
                tr[c] = r[slot];
                ti[c] = i[slot];
            }
            V_TRANSPOSE(tr);
            V_TRANSPOSE(ti);
            for (c = 0; c < VW; c++) {
                v_store(o[c] + off + 2 * j, tr[c]);
                v_store(o[c] + off + 2 * j + VW, ti[c]);
            }
        }
        rev = twofft_bitrev_next(rev, ntiles);
    }
}

KFUNC void FN(leaf8_fwd)(const float *restrict in, float *restrict out, unsigned n)
{
    FN(leaf_impl)(in, out, n, 3, 0);
}
KFUNC void FN(leaf8_inv)(const float *restrict in, float *restrict out, unsigned n)
{
    FN(leaf_impl)(in, out, n, 3, 1);
}
KFUNC void FN(leaf16_fwd)(const float *restrict in, float *restrict out, unsigned n)
{
    FN(leaf_impl)(in, out, n, 4, 0);
}
KFUNC void FN(leaf16_inv)(const float *restrict in, float *restrict out, unsigned n)
{
    FN(leaf_impl)(in, out, n, 4, 1);
}

/* ---------------------------------------------------------------------------------------
 * Radix 4 and radix 8 passes. `x` holds nblk blocks of 4S (8S) complex numbers in split
 * block format. Sub-block p of a block holds the size S transform of the sub-sequence
 * with offset bitrev(p). mode: 0 = middle pass, 1 = final forward, 2 = final inverse.
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

/* tw: per vector position p the twiddles w^1, w^2, w^3 (w = exp(-2 pi i (p + lane) / 4S)),
 * each as VW real parts followed by VW imaginary parts. */
KINLINE void FN(pass4_impl)(float *restrict x, const size_t S, const size_t nblk,
                            const float *restrict tw, const int mode)
{
    enum { VW = TWOFFT_VW };
    size_t b, p;
    for (b = 0; b < nblk; b++) {
        float *x0 = x + 8 * S * b, *x1 = x0 + 2 * S, *x2 = x0 + 4 * S, *x3 = x0 + 6 * S;
        const float *t = tw;
        for (p = 0; p < S; p += VW) {
            const vf ar = v_load(x0), ai = v_load(x0 + VW);
            const vf br = v_load(x1), bi = v_load(x1 + VW);
            const vf cr = v_load(x2), ci = v_load(x2 + VW);
            const vf dr = v_load(x3), di = v_load(x3 + VW);
            const vf w1r = v_load(t), w1i = v_load(t + VW);
            const vf w2r = v_load(t + 2 * VW), w2i = v_load(t + 3 * VW);
            const vf w3r = v_load(t + 4 * VW), w3i = v_load(t + 5 * VW);
            /* sub-blocks hold offsets 0, 2, 1, 3 */
            const vf t2r = v_fnmadd(v_mul(w2r, br), w2i, bi), t2i = v_fmadd(v_mul(w2r, bi), w2i, br);
            const vf t1r = v_fnmadd(v_mul(w1r, cr), w1i, ci), t1i = v_fmadd(v_mul(w1r, ci), w1i, cr);
            const vf t3r = v_fnmadd(v_mul(w3r, dr), w3i, di), t3i = v_fmadd(v_mul(w3r, di), w3i, dr);
            const vf s02r = v_add(ar, t2r), s02i = v_add(ai, t2i);
            const vf d02r = v_sub(ar, t2r), d02i = v_sub(ai, t2i);
            const vf s13r = v_add(t1r, t3r), s13i = v_add(t1i, t3i);
            const vf d13r = v_sub(t1r, t3r), d13i = v_sub(t1i, t3i);
            FN(pass_store)(x0, v_add(s02r, s13r), v_add(s02i, s13i), mode);
            FN(pass_store)(x1, v_add(d02r, d13i), v_sub(d02i, d13r), mode);
            FN(pass_store)(x2, v_sub(s02r, s13r), v_sub(s02i, s13i), mode);
            FN(pass_store)(x3, v_sub(d02r, d13i), v_add(d02i, d13r), mode);
            x0 += 2 * VW, x1 += 2 * VW, x2 += 2 * VW, x3 += 2 * VW, t += 6 * VW;
        }
    }
}

/* tw: per vector position the twiddles w^1 .. w^7 (w = exp(-2 pi i (p + lane) / 8S)). */
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
            vf r[8], i[8];
            /* sub-block q holds offset bitrev3(q) */
            for (j = 0; j < 8; j++) {
                const float *src = xb + 2 * S * twofft_bitrev(j, 3);
                r[j] = v_load(src);
                i[j] = v_load(src + VW);
            }
            for (j = 1; j < 8; j++) {
                const vf wr = v_load(t + 2 * VW * (j - 1)), wi = v_load(t + 2 * VW * (j - 1) + VW);
                const vf zr = r[j], zi = i[j];
                r[j] = v_fnmadd(v_mul(wr, zr), wi, zi);
                i[j] = v_fmadd(v_mul(wr, zi), wi, zr);
            }
            FN(dft8)(r, i);
            for (j = 0; j < 8; j++)
                FN(pass_store)(xb + 2 * S * j, r[TWOFFT_SLOT8(j)], i[TWOFFT_SLOT8(j)], mode);
            xb += 2 * VW, t += 14 * VW;
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
 * Driver for N >= 32.
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
    if (pl->leaf_log == 3) {
        if (inv)
            FN(leaf8_inv)(in, out, pl->log2n);
        else
            FN(leaf8_fwd)(in, out, pl->log2n);
    } else {
        if (inv)
            FN(leaf16_inv)(in, out, pl->log2n);
        else
            FN(leaf16_fwd)(in, out, pl->log2n);
    }
    for (q = 0; q < ((size_t)1 << pl->radix_log[last]); q++)
        FN(middle)(pl, out + 2 * S * q, last);
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

#undef FN
#undef KINLINE
#undef KFUNC
#undef TWOFFT_CAT3
#undef TWOFFT_CAT3_
