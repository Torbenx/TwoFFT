/* Tests the SIMD backend of the target architecture against the double
 * precision reference and checks that it stays inside the output buffer. */
#include <stdio.h>
#include <string.h>

#include "../src/tw_scalar.h"
#if defined(__aarch64__)
#include "../src/tw_arm64.h"
#define backend_fft tw_a64_fft
#define BACKEND "arm64"
#else
#define backend_fft tw_scalar_fft
#define BACKEND "scalar"
#endif
#include "reference.h"

#define GUARD 64

int main(int argc, char** argv) {
    const unsigned max_log2n = argc > 1 ? (unsigned)atoi(argv[1]) : 20;
    int failures = 0;
    printf("backend: %s\n", BACKEND);
    for (unsigned log2n = 0; log2n <= max_log2n; log2n++) {
        const size_t n = (size_t)1 << log2n;
        float* w = (float*)malloc((tw_weights_floats(log2n) + 1) * sizeof(float));
        tw_weights_fill(log2n, w);
        float* in = (float*)malloc((2 * n + 2 * GUARD + 1) * sizeof(float));
        float* out = (float*)malloc((2 * n + 2 * GUARD + 1) * sizeof(float));
        double* din = (double*)malloc(2 * n * sizeof(double));
        double* dout = (double*)malloc(2 * n * sizeof(double));
        for (int backward = 0; backward < 2; backward++) {
            for (int unaligned = 0; unaligned < 2; unaligned++) {
                float* x = in + GUARD + unaligned;
                float* y = out + GUARD + unaligned;
                for (size_t i = 0; i < 2 * n; i++) din[i] = x[i] = (float)ref_random();
                for (size_t i = 0; i < 2 * n + 2 * GUARD + 1; i++) out[i] = 12345.0f;
                backend_fft(log2n, w, x, y, backward);
                ref_fft(n, din, dout, backward ? 1 : -1);
                double err = 0, norm = 0;
                for (size_t i = 0; i < 2 * n; i++) {
                    err += (y[i] - dout[i]) * (y[i] - dout[i]);
                    norm += dout[i] * dout[i];
                }
                int ok = 1;
                for (size_t i = 0; i < 2 * n; i++) ok &= x[i] == (float)din[i];
                for (size_t i = 0; i < GUARD; i++) ok &= y[-1 - (ptrdiff_t)i] == 12345.0f && y[2 * n + i] == 12345.0f;
                const double rel = sqrt(err / norm);
                ok &= rel < 1e-6;
                if (!ok) failures++;
                if (!ok || unaligned == 0)
                    printf("log2n %2u %s rel. error %.3g %s\n", log2n, backward ? "backward" : "forward ", rel,
                           ok ? "" : "FAILED");
            }
        }
        free(w), free(in), free(out), free(din), free(dout);
    }
    printf(failures ? "FAILED\n" : "OK\n");
    return failures != 0;
}
