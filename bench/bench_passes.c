/* Development tool: cycles per complex number of every pass of the ARM64
 * backend, and of the whole transform.
 *
 *   bench_passes [min log2n] [max log2n]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_util.h"
#include "../src/tw_arm64.h"

static double ghz;

#define TIME(label, call)                                                  \
    do {                                                                   \
        double best = 1e30;                                                \
        for (int b = 0; b < 7; b++) {                                      \
            const double t0 = bench_now();                                 \
            for (size_t i = 0; i < reps; i++) { call; }                    \
            const double t = (bench_now() - t0) / (double)reps;            \
            if (t < best) best = t;                                        \
        }                                                                  \
        printf(" %s %.2f", label, best * ghz * 1e9 / (double)n);           \
    } while (0)

int main(int argc, char** argv) {
    const unsigned lo = argc > 1 ? (unsigned)atoi(argv[1]) : 6;
    const unsigned hi = argc > 2 ? (unsigned)atoi(argv[2]) : 20;
    bench_warm_up(0.5);
    ghz = bench_ghz();
    printf("clock frequency: %.2f GHz, cycles per complex number:\n", ghz);
    for (unsigned log2n = lo < 6 ? 6 : lo; log2n <= hi; log2n++) {
        const size_t n = (size_t)1 << log2n, m = n / 8;
        float* w = (float*)bench_alloc(tw_weights_floats(log2n) * sizeof(float));
        tw_weights_fill(log2n, w);
        float* in = (float*)bench_alloc(2 * n * sizeof(float));
        float* out = (float*)bench_alloc(2 * n * sizeof(float));
        for (size_t i = 0; i < 2 * n; i++) in[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        memset(out, 0, 2 * n * sizeof(float));
        const float* wr = w + (m / 4) * TW_FIRST_TILE_FLOATS;
        const size_t s0 = tw_first_stride(log2n);
        const size_t reps = ((size_t)1 << 22) / n + 2;
        printf("%2u:", log2n);
        TIME("fft", tw_a64_fft(log2n, w, in, out, 0));
        TIME("| first", tw_a64_first(log2n, w, in, out, 0));
        /* the remaining passes run on zeros, so that repetitions cannot overflow */
        memset(out, 0, 2 * n * sizeof(float));
        if (s0 == 1 && m > 8) TIME("t8", tw_a64_t8(out, m / 8));
        for (size_t s = s0 == 1 ? 8 : s0; 8 * s < m; s *= 8) {
            char label[32];
            snprintf(label, sizeof label, "r8/%zu", s);
            TIME(label, tw_a64_r8(out, wr + tw_r8_weights_offset(s0, s), s, m / (8 * s)));
        }
        if (m > 8) TIME("last", tw_a64_r8_last_fwd(out, wr + tw_r8_weights_offset(s0, m / 8), m / 8, 1));
        else TIME("last", tw_a64_t8_last_fwd(out, 1));
        printf("\n");
        free(w), free(in), free(out);
    }
    return 0;
}
