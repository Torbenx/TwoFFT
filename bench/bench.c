/* TwoFFT benchmark: times out-of-place complex forward transforms of TwoFFT and, when
 * compiled in, of other libraries.
 *
 * Usage: bench [-c cpu] [-l min_log2] [-u max_log2] [-t ms_per_measurement] [-r repeats]
 *              [-o lib1,lib2,...]
 *
 * Prints one line per (library, size): the minimum over `repeats` measurements of the
 * time per transform in nanoseconds. Optional libraries are enabled with
 * -DBENCH_FFTW, -DBENCH_PFFFT, -DBENCH_KISSFFT, -DBENCH_KFR, -DBENCH_VDSP, -DBENCH_POCKETFFT.
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

#ifdef BENCH_FFTW
#include <fftw3.h>
#endif
#ifdef BENCH_PFFFT
#include "pffft.h"
#endif
#ifdef BENCH_KISSFFT
#include "kiss_fft.h"
#endif
#ifdef BENCH_KFR
#include <stdbool.h>
#include <kfr/capi.h>
#endif
#ifdef BENCH_VDSP
#include <Accelerate/Accelerate.h>
#endif

static void *alloc64(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

/* A library under test: setup() prepares a transform of 2^log2n points from `in` to `out`
 * (both 64 byte aligned, interleaved complex), run() executes it once. */
typedef struct {
    const char *name;
    void *(*setup)(unsigned log2n, float *in, float *out);
    void (*run)(void *state);
    void (*teardown)(void *state);
} bench_lib;

/* ---- TwoFFT ---- */
typedef struct {
    twofft_context *ctx;
    const float *in;
    float *out;
    unsigned log2n;
} twofft_state;
static void *twofft_setup(unsigned log2n, float *in, float *out)
{
    twofft_state *s = (twofft_state *)malloc(sizeof *s);
    s->ctx = twofft_context_create();
    s->in = in, s->out = out, s->log2n = log2n;
    return s;
}
static void twofft_run(void *state)
{
    twofft_state *s = (twofft_state *)state;
    twofft_forward(s->ctx, s->in, s->out, s->log2n);
}
static void twofft_teardown(void *state)
{
    twofft_state *s = (twofft_state *)state;
    twofft_context_destroy(s->ctx);
    free(s);
}

#ifdef BENCH_FFTW
static unsigned fftw_flags = FFTW_MEASURE;
static void *fftw_setup(unsigned log2n, float *in, float *out)
{
    /* FFTW_MEASURE overwrites the arrays while planning; the caller refills them */
    return fftwf_plan_dft_1d(1 << log2n, (fftwf_complex *)in, (fftwf_complex *)out, FFTW_FORWARD, fftw_flags);
}
static void fftw_run(void *state) { fftwf_execute((fftwf_plan)state); }
static void fftw_teardown(void *state) { fftwf_destroy_plan((fftwf_plan)state); }
#endif

#ifdef BENCH_PFFFT
typedef struct {
    PFFFT_Setup *setup;
    float *in, *out, *work;
} pffft_state;
static void *pffft_setup(unsigned log2n, float *in, float *out)
{
    pffft_state *s;
    if (log2n < 4) /* pffft needs N >= 16 for complex transforms */
        return NULL;
    s = (pffft_state *)malloc(sizeof *s);
    s->setup = pffft_new_setup(1 << log2n, PFFFT_COMPLEX);
    s->in = in, s->out = out;
    s->work = (float *)pffft_aligned_malloc(((size_t)2 << log2n) * sizeof(float));
    return s;
}
static void pffft_run(void *state)
{
    pffft_state *s = (pffft_state *)state;
    pffft_transform_ordered(s->setup, s->in, s->out, s->work, PFFFT_FORWARD);
}
static void pffft_teardown(void *state)
{
    pffft_state *s = (pffft_state *)state;
    pffft_destroy_setup(s->setup);
    pffft_aligned_free(s->work);
    free(s);
}
#endif

#ifdef BENCH_KISSFFT
typedef struct {
    kiss_fft_cfg cfg;
    float *in, *out;
} kiss_state;
static void *kiss_setup(unsigned log2n, float *in, float *out)
{
    kiss_state *s = (kiss_state *)malloc(sizeof *s);
    s->cfg = kiss_fft_alloc(1 << log2n, 0, NULL, NULL);
    s->in = in, s->out = out;
    return s;
}
static void kiss_run(void *state)
{
    kiss_state *s = (kiss_state *)state;
    kiss_fft(s->cfg, (const kiss_fft_cpx *)s->in, (kiss_fft_cpx *)s->out);
}
static void kiss_teardown(void *state)
{
    kiss_state *s = (kiss_state *)state;
    kiss_fft_free(s->cfg);
    free(s);
}
#endif

#ifdef BENCH_KFR
typedef struct {
    KFR_DFT_PLAN_F32 *plan;
    float *in, *out;
    uint8_t *temp;
} kfr_state;
static void *kfr_setup(unsigned log2n, float *in, float *out)
{
    kfr_state *s = (kfr_state *)malloc(sizeof *s);
    s->plan = kfr_dft_create_plan_f32((size_t)1 << log2n);
    s->in = in, s->out = out;
    s->temp = (uint8_t *)alloc64(kfr_dft_get_temp_size_f32(s->plan) + 64);
    return s;
}
static void kfr_run(void *state)
{
    kfr_state *s = (kfr_state *)state;
    kfr_dft_execute_f32(s->plan, s->out, s->in, s->temp);
}
static void kfr_teardown(void *state)
{
    kfr_state *s = (kfr_state *)state;
    kfr_dft_delete_plan_f32(s->plan);
    free(s->temp);
    free(s);
}
#endif

#ifdef BENCH_VDSP
typedef struct {
    vDSP_DFT_Interleaved_Setup setup;
    float *in, *out;
    size_t n;
} vdsp_state;
static void *vdsp_setup(unsigned log2n, float *in, float *out)
{
    vdsp_state *s;
    if (log2n < 3)
        return NULL;
    s = (vdsp_state *)malloc(sizeof *s);
    s->setup = vDSP_DFT_Interleaved_CreateSetup(NULL, (vDSP_Length)1 << log2n, vDSP_DFT_FORWARD,
                                                vDSP_DFT_Interleaved_ComplextoComplex);
    s->in = in, s->out = out, s->n = (size_t)1 << log2n;
    if (!s->setup) {
        free(s);
        return NULL;
    }
    return s;
}
static void vdsp_run(void *state)
{
    vdsp_state *s = (vdsp_state *)state;
    vDSP_DFT_Interleaved_Execute(s->setup, (const DSPComplex *)s->in, (DSPComplex *)s->out);
}
static void vdsp_teardown(void *state)
{
    vdsp_state *s = (vdsp_state *)state;
    vDSP_DFT_Interleaved_DestroySetup(s->setup);
    free(s);
}
#endif

static const bench_lib libs[] = {
    {"twofft", twofft_setup, twofft_run, twofft_teardown},
#ifdef BENCH_FFTW
    {"fftw", fftw_setup, fftw_run, fftw_teardown},
#endif
#ifdef BENCH_PFFFT
    {"pffft", pffft_setup, pffft_run, pffft_teardown},
#endif
#ifdef BENCH_KFR
    {"kfr", kfr_setup, kfr_run, kfr_teardown},
#endif
#ifdef BENCH_VDSP
    {"vdsp", vdsp_setup, vdsp_run, vdsp_teardown},
#endif
#ifdef BENCH_KISSFFT
    {"kissfft", kiss_setup, kiss_run, kiss_teardown},
#endif
};

static void fill(float *x, size_t count)
{
    uint32_t s = 12345;
    size_t j;
    for (j = 0; j < count; j++) {
        s = s * 1664525u + 1013904223u;
        x[j] = (float)(s >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
}

static int selected(const char *only, const char *name)
{
    const size_t len = strlen(name);
    const char *p = only;
    if (!only)
        return 1;
    while ((p = strstr(p, name)) != NULL) {
        if ((p == only || p[-1] == ',') && (p[len] == 0 || p[len] == ','))
            return 1;
        p += len;
    }
    return 0;
}

int main(int argc, char **argv)
{
    int cpu = -1, a;
    unsigned min_log2 = 1, max_log2 = 20, repeats = 15, log2n;
    double target_ns = 5e6;
    const char *only = NULL;
    size_t li;
    for (a = 1; a + 1 < argc; a += 2) {
        if (!strcmp(argv[a], "-c"))
            cpu = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-l"))
            min_log2 = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-u"))
            max_log2 = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-t"))
            target_ns = atof(argv[a + 1]) * 1e6;
        else if (!strcmp(argv[a], "-r"))
            repeats = (unsigned)atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "-o"))
            only = argv[a + 1];
#ifdef BENCH_FFTW
        else if (!strcmp(argv[a], "-f"))
            fftw_flags = !strcmp(argv[a + 1], "patient") ? FFTW_PATIENT
                         : !strcmp(argv[a + 1], "estimate") ? FFTW_ESTIMATE : FFTW_MEASURE;
#endif
    }
    bench_prepare_cpu(cpu);

    printf("%-10s %5s %14s %12s\n", "lib", "log2n", "ns/transform", "ns/(N log2N)");
    for (log2n = min_log2; log2n <= max_log2; log2n++) {
        const size_t n = (size_t)1 << log2n;
        float *in = (float *)alloc64(2 * n * sizeof(float));
        float *out = (float *)alloc64(2 * n * sizeof(float));
        for (li = 0; li < sizeof libs / sizeof libs[0]; li++) {
            const bench_lib *lib = &libs[li];
            void *state;
            double best = 1e300, t0, t1;
            size_t iters = 1, it;
            unsigned r;
            if (!selected(only, lib->name))
                continue;
            state = lib->setup(log2n, in, out);
            if (!state)
                continue;
            fill(in, 2 * n);
            fill(out, 2 * n);
            lib->run(state); /* warm up, builds lazily created tables */
            for (;;) {       /* calibrate the iteration count */
                t0 = now_ns();
                for (it = 0; it < iters; it++)
                    lib->run(state);
                t1 = now_ns();
                if (t1 - t0 >= target_ns * 0.5 || iters > ((size_t)1 << 40))
                    break;
                iters *= 2;
            }
            for (r = 0; r < repeats; r++) {
                t0 = now_ns();
                for (it = 0; it < iters; it++)
                    lib->run(state);
                t1 = now_ns();
                if ((t1 - t0) / (double)iters < best)
                    best = (t1 - t0) / (double)iters;
            }
            printf("%-10s %5u %14.1f %12.4f\n", lib->name, log2n, best, best / ((double)n * log2n));
            fflush(stdout);
            lib->teardown(state);
        }
        free(in);
        free(out);
    }
    return 0;
}
