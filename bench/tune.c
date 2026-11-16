/* TwoFFT tuning tool (development only; requires the library built with -DTWOFFT_TUNING).
 *
 * For every size it enumerates all decompositions (leaf shape + pass radices), checks
 * that each one computes the same result and prints them ordered by speed.
 *
 * Usage: tune [-c cpu] [-l min_log2] [-u max_log2] [-t ms] [-r repeats] [-k keep]
 *             [-b block_log2] [-s spec]      (-s: time only this decomposition)
 */

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "twofft.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bench_util.h"

int twofft_tuning_set_decomposition(twofft_context *ctx, unsigned n, const char *spec);
extern unsigned twofft_tuning_block_log2;
extern unsigned twofft_tuning_stages;

typedef struct {
    char spec[24];
    double ns;
} result;

static int by_time(const void *a, const void *b)
{
    const double d = ((const result *)a)->ns - ((const result *)b)->ns;
    return d < 0 ? -1 : d > 0;
}

static double target_ns = 3e6;
static unsigned repeats = 7;

static double time_transform(twofft_context *ctx, const float *in, float *out, unsigned n)
{
    double best = 1e300, t0, t1;
    size_t iters = 1, it;
    unsigned r;
    for (;;) {
        t0 = now_ns();
        for (it = 0; it < iters; it++)
            twofft_forward(ctx, in, out, n);
        t1 = now_ns();
        if (t1 - t0 >= target_ns * 0.5)
            break;
        iters *= 2;
    }
    for (r = 0; r < repeats; r++) {
        t0 = now_ns();
        for (it = 0; it < iters; it++)
            twofft_forward(ctx, in, out, n);
        t1 = now_ns();
        if ((t1 - t0) / (double)iters < best)
            best = (t1 - t0) / (double)iters;
    }
    return best;
}

static result results[4096];
static unsigned nresults;

static void try_spec(twofft_context *ctx, const float *in, float *out, const float *want, unsigned n,
                     const char *spec)
{
    const size_t len = (size_t)2 << n;
    double err = 0, norm = 0;
    size_t j;
    if (twofft_tuning_set_decomposition(ctx, n, spec) != 0)
        return;
    memset(out, 0, len * sizeof(float));
    twofft_forward(ctx, in, out, n);
    for (j = 0; j < len; j++) {
        const double d = (double)out[j] - want[j];
        err += d * d, norm += (double)want[j] * want[j];
    }
    if (!(sqrt(err / norm) < 1e-5)) {
        printf("WRONG RESULT n=%u spec=%s error %g\n", n, spec, sqrt(err / norm));
        exit(1);
    }
    if (nresults < sizeof results / sizeof results[0]) {
        strcpy(results[nresults].spec, spec);
        results[nresults].ns = time_transform(ctx, in, out, n);
        nresults++;
    }
}

static void enumerate(twofft_context *ctx, const float *in, float *out, const float *want, unsigned n,
                      char *spec, unsigned pos, unsigned rest)
{
    if (rest == 0) {
        spec[pos] = 0;
        if (pos > 1)
            try_spec(ctx, in, out, want, n, spec);
        return;
    }
    if (rest >= 2) {
        spec[pos] = '2';
        enumerate(ctx, in, out, want, n, spec, pos + 1, rest - 2);
    }
    if (rest >= 3) {
        spec[pos] = '3';
        enumerate(ctx, in, out, want, n, spec, pos + 1, rest - 3);
    }
}

int main(int argc, char **argv)
{
    static const unsigned leaf_levels[5] = {3, 4, 5, 5, 6};
    unsigned min_log2 = 5, max_log2 = 14, keep = 6, n, i;
    const char *only = NULL;
    int a, cpu = -1;
    for (a = 1; a + 1 < argc; a += 2) {
        if (!strcmp(argv[a], "-c")) cpu = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-l")) min_log2 = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-u")) max_log2 = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-t")) target_ns = atof(argv[a + 1]) * 1e6;
        else if (!strcmp(argv[a], "-r")) repeats = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-k")) keep = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-b")) twofft_tuning_block_log2 = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-s")) only = argv[a + 1];
    }
    bench_prepare_cpu(cpu);
    for (n = min_log2; n <= max_log2; n++) {
        const size_t len = (size_t)2 << n;
        twofft_context *ctx = twofft_context_create();
        float *in = (float *)malloc(len * sizeof(float));
        float *out = (float *)malloc(len * sizeof(float));
        float *want = (float *)malloc(len * sizeof(float));
        uint32_t s = 1;
        size_t j;
        char spec[24];
        for (j = 0; j < len; j++) {
            s = s * 1664525u + 1013904223u;
            in[j] = (float)(s >> 8) * (1.0f / 8388608.0f) - 1.0f;
        }
        twofft_forward(ctx, in, want, n); /* default decomposition, verified by the tests */
        nresults = 0;
        if (only) {
            /* a single decomposition: also time its stages separately */
            double leaf, middle, final;
            try_spec(ctx, in, out, want, n, only);
            twofft_tuning_stages = 1, leaf = time_transform(ctx, in, out, n);
            twofft_tuning_stages = 2, middle = time_transform(ctx, in, out, n);
            twofft_tuning_stages = 4, final = time_transform(ctx, in, out, n);
            twofft_tuning_stages = 7;
            printf("n=%-2u %s: leaf %.1f  middle %.1f  final %.1f  (ns/point: %.3f %.3f %.3f)  ", n, only, leaf,
                   middle, final, leaf / (double)(len / 2), middle / (double)(len / 2), final / (double)(len / 2));
        } else {
            for (i = 0; i < 5; i++) {
                if (n < leaf_levels[i] + 2)
                    continue;
                spec[0] = (char)('0' + i);
                enumerate(ctx, in, out, want, n, spec, 1, n - leaf_levels[i]);
            }
        }
        qsort(results, nresults, sizeof results[0], by_time);
        printf("n=%-2u (%u decompositions)", n, nresults);
        for (i = 0; i < nresults && i < keep; i++)
            printf("  %s %.1f", results[i].spec, results[i].ns);
        printf("\n");
        fflush(stdout);
        free(in), free(out), free(want);
        twofft_context_destroy(ctx);
    }
    return 0;
}
