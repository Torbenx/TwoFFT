/* TwoFFT correctness tests.
 *
 * Every size is checked against an independent double precision reference:
 *   - random input, forward and inverse (direct O(N^2) DFT for small N, a plain recursive
 *     radix 2 FFT above),
 *   - unit impulses at every position for small N (catches any ordering mistake),
 *   - forward followed by inverse reproduces N times the input.
 *
 * Usage: test_twofft [max_log2_size]
 */

#include "twofft.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const double kTwoPi = 6.283185307179586476925286766559;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static double rng_uniform(void) /* in [-1, 1) */
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static void reference_dft(const double *in, double *out, size_t n, int sign)
{
    size_t k, j;
    for (k = 0; k < n; k++) {
        double sr = 0, si = 0;
        for (j = 0; j < n; j++) {
            const double a = sign * kTwoPi * (double)((k * j) % n) / (double)n;
            const double c = cos(a), s = sin(a);
            sr += in[2 * j] * c - in[2 * j + 1] * s;
            si += in[2 * j] * s + in[2 * j + 1] * c;
        }
        out[2 * k] = sr;
        out[2 * k + 1] = si;
    }
}

/* out[k] = sum_j in[j * stride] exp(sign 2 pi i j k / n) */
static void reference_fft_rec(const double *in, size_t stride, double *out, size_t n, int sign, double *tmp)
{
    size_t k;
    if (n == 1) {
        out[0] = in[0], out[1] = in[1];
        return;
    }
    reference_fft_rec(in, 2 * stride, tmp, n / 2, sign, out);
    reference_fft_rec(in + 2 * stride, 2 * stride, tmp + n, n / 2, sign, out);
    for (k = 0; k < n / 2; k++) {
        const double a = sign * kTwoPi * (double)k / (double)n;
        const double c = cos(a), s = sin(a);
        const double er = tmp[2 * k], ei = tmp[2 * k + 1];
        const double pr = tmp[n + 2 * k], pi = tmp[n + 2 * k + 1];
        const double tr = pr * c - pi * s, ti = pr * s + pi * c;
        out[2 * k] = er + tr, out[2 * k + 1] = ei + ti;
        out[n + 2 * k] = er - tr, out[n + 2 * k + 1] = ei - ti;
    }
}

static void reference(const double *in, double *out, size_t n, int sign, double *tmp)
{
    if (n <= 512)
        reference_dft(in, out, n, sign);
    else
        reference_fft_rec(in, 1, out, n, sign, tmp);
}

/* relative l2 error of got against want */
static double rel_error(const float *got, const double *want, size_t n)
{
    double err = 0, norm = 0;
    size_t j;
    for (j = 0; j < 2 * n; j++) {
        const double d = (double)got[j] - want[j];
        err += d * d;
        norm += want[j] * want[j];
    }
    return sqrt(err / (norm > 0 ? norm : 1));
}

static int failures = 0;

static void check(const char *what, unsigned log2n, double err, double tol)
{
    if (!(err <= tol)) {
        printf("FAIL  n=2^%-2u %-18s error %.3e > %.3e\n", log2n, what, err, tol);
        failures++;
    }
}

int main(int argc, char **argv)
{
    const unsigned max_log2 = argc > 1 ? (unsigned)atoi(argv[1]) : 20;
    twofft_context *ctx = twofft_context_create();
    unsigned log2n;
    if (!ctx) {
        printf("FAIL  context creation\n");
        return 1;
    }
    for (log2n = 0; log2n <= max_log2; log2n++) {
        const size_t n = (size_t)1 << log2n;
        /* float rounding errors grow with sqrt(log2 n); generous bound */
        const double tol = 2e-7 * (double)(log2n + 2);
        float *in = (float *)malloc(2 * n * sizeof(float));
        float *out = (float *)malloc(2 * n * sizeof(float));
        float *back = (float *)malloc(2 * n * sizeof(float));
        double *din = (double *)malloc(2 * n * sizeof(double));
        double *dout = (double *)malloc(2 * n * sizeof(double));
        double *tmp = (double *)malloc(2 * n * sizeof(double));
        double worst = 0, e;
        size_t j, pos;
        if (!in || !out || !back || !din || !dout || !tmp) {
            printf("FAIL  out of memory at n=2^%u\n", log2n);
            return 1;
        }
        for (j = 0; j < 2 * n; j++) {
            in[j] = (float)rng_uniform();
            din[j] = in[j];
        }

        if (twofft_forward(ctx, in, out, log2n) != 0)
            check("forward returned", log2n, 1, 0);
        reference(din, dout, n, -1, tmp);
        e = rel_error(out, dout, n);
        check("forward", log2n, e, tol);
        worst = e > worst ? e : worst;

        if (twofft_inverse(ctx, out, back, log2n) != 0)
            check("inverse returned", log2n, 1, 0);
        for (j = 0; j < 2 * n; j++)
            dout[j] = din[j] * (double)n;
        e = rel_error(back, dout, n);
        check("roundtrip", log2n, e, 2 * tol);

        if (twofft_inverse(ctx, in, out, log2n) != 0)
            check("inverse returned", log2n, 1, 0);
        reference(din, dout, n, +1, tmp);
        e = rel_error(out, dout, n);
        check("inverse", log2n, e, tol);
        worst = e > worst ? e : worst;

        /* impulses: every position for small sizes, a few for large ones */
        for (pos = 0; pos < n; pos += (n <= 1024 ? 1 : n / 61 + 1)) {
            double emax = 0;
            for (j = 0; j < 2 * n; j++)
                in[j] = 0;
            in[2 * pos] = 1;
            twofft_forward(ctx, in, out, log2n);
            for (j = 0; j < n; j++) {
                const double a = -kTwoPi * (double)((pos * j) % n) / (double)n;
                const double dr = out[2 * j] - cos(a), di = out[2 * j + 1] - sin(a);
                const double d = sqrt(dr * dr + di * di);
                emax = d > emax ? d : emax;
            }
            check("impulse", log2n, emax, 2e-6);
            in[2 * pos] = 0;
            in[2 * pos + 1] = 1;
            twofft_inverse(ctx, in, out, log2n);
            emax = 0;
            for (j = 0; j < n; j++) {
                const double a = kTwoPi * (double)((pos * j) % n) / (double)n;
                const double dr = out[2 * j] + sin(a), di = out[2 * j + 1] - cos(a);
                const double d = sqrt(dr * dr + di * di);
                emax = d > emax ? d : emax;
            }
            check("impulse inverse", log2n, emax, 2e-6);
        }
        printf("ok    n=2^%-2u  max rel. error %.2e\n", log2n, worst);
        free(in), free(out), free(back), free(din), free(dout), free(tmp);
    }
    twofft_context_destroy(ctx);
    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
