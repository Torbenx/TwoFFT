/* Double precision reference transforms for the tests. */
#ifndef TW_TEST_REFERENCE_H
#define TW_TEST_REFERENCE_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* O(n^2) DFT, the ground truth. sign = -1 forward, +1 backward. */
static void ref_dft(size_t n, const double* in, double* out, int sign) {
    const double pi2 = 6.283185307179586476925286766559;
    for (size_t k = 0; k < n; k++) {
        long double re = 0, im = 0;
        for (size_t j = 0; j < n; j++) {
            const double a = sign * pi2 * (double)((j * k) % n) / (double)n;
            const double c = cos(a), s = sin(a);
            re += in[2 * j] * c - in[2 * j + 1] * s;
            im += in[2 * j] * s + in[2 * j + 1] * c;
        }
        out[2 * k] = (double)re;
        out[2 * k + 1] = (double)im;
    }
}

/* Recursive radix-2 FFT in double precision. */
static void ref_fft_rec(size_t n, const double* in, size_t stride, double* out, double* tmp, int sign) {
    if (n == 1) {
        out[0] = in[0];
        out[1] = in[1];
        return;
    }
    const size_t h = n / 2;
    ref_fft_rec(h, in, 2 * stride, tmp, out, sign);
    ref_fft_rec(h, in + 2 * stride, 2 * stride, tmp + n, out + n, sign);
    const double pi2 = 6.283185307179586476925286766559;
    for (size_t k = 0; k < h; k++) {
        const double a = sign * pi2 * (double)k / (double)n;
        const double c = cos(a), s = sin(a);
        const double er = tmp[2 * k], ei = tmp[2 * k + 1];
        const double pr = tmp[n + 2 * k], pi = tmp[n + 2 * k + 1];
        const double tr = pr * c - pi * s, ti = pr * s + pi * c;
        out[2 * k] = er + tr;
        out[2 * k + 1] = ei + ti;
        out[n + 2 * k] = er - tr;
        out[n + 2 * k + 1] = ei - ti;
    }
}

static void ref_fft(size_t n, const double* in, double* out, int sign) {
    double* tmp = (double*)malloc(2 * n * sizeof(double));
    ref_fft_rec(n, in, 1, out, tmp, sign);
    free(tmp);
}

/* Deterministic pseudo random numbers in [-1, 1). */
static unsigned long long ref_rng_state = 0x9E3779B97F4A7C15ull;
static double ref_random(void) {
    ref_rng_state ^= ref_rng_state << 13;
    ref_rng_state ^= ref_rng_state >> 7;
    ref_rng_state ^= ref_rng_state << 17;
    return (double)(ref_rng_state >> 11) / (double)(1ull << 52) - 1.0;
}

#endif
