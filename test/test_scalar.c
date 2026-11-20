/* Tests the portable implementation against the double precision reference. */
#include <stdio.h>
#include <string.h>

#include "../src/tw_scalar.h"
#include "reference.h"

int main(void) {
    int failures = 0;
    /* reference self check */
    {
        double in[2 * 64], a[2 * 64], b[2 * 64];
        for (int i = 0; i < 128; i++) in[i] = ref_random();
        ref_dft(64, in, a, -1);
        ref_fft(64, in, b, -1);
        double e = 0;
        for (int i = 0; i < 128; i++) e = fmax(e, fabs(a[i] - b[i]));
        printf("reference self check: %g\n", e);
        if (e > 1e-12) failures++;
    }
    for (unsigned log2n = 0; log2n <= 20; log2n++) {
        const size_t n = (size_t)1 << log2n;
        float* w = (float*)malloc((tw_weights_floats(log2n) + 1) * sizeof(float));
        tw_weights_fill(log2n, w);
        float* in = (float*)malloc(2 * n * sizeof(float));
        float* out = (float*)malloc(2 * n * sizeof(float));
        double* din = (double*)malloc(2 * n * sizeof(double));
        double* dout = (double*)malloc(2 * n * sizeof(double));
        for (int backward = 0; backward < 2; backward++) {
            for (size_t i = 0; i < 2 * n; i++) din[i] = in[i] = (float)ref_random();
            tw_scalar_fft(log2n, w, in, out, backward);
            ref_fft(n, din, dout, backward ? 1 : -1);
            double err = 0, norm = 0;
            for (size_t i = 0; i < 2 * n; i++) {
                err += (out[i] - dout[i]) * (out[i] - dout[i]);
                norm += dout[i] * dout[i];
            }
            const double rel = sqrt(err / norm);
            const int ok = rel < 1e-6;
            if (!ok) failures++;
            printf("log2n %2u %s rel. error %.3g %s\n", log2n, backward ? "backward" : "forward ", rel, ok ? "" : "FAILED");
        }
        free(w), free(in), free(out), free(din), free(dout);
    }
    printf(failures ? "FAILED\n" : "OK\n");
    return failures != 0;
}
