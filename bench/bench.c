/* Benchmark of single precision complex power-of-two FFTs.
 *
 *   bench [min log2n] [max log2n] [library ...]
 *
 * Prints the time per forward transform in nanoseconds for each library
 * compiled in (BENCH_PFFFT, BENCH_FFTW, BENCH_VDSP, BENCH_KFR, ...).
 * The reported time is the minimum over several batches. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_util.h"
#include "../src/tw_scalar.h"
#if defined(__aarch64__)
#include "../src/tw_arm64.h"
#endif

typedef struct library {
    const char* name;
    void* (*setup)(unsigned log2n);
    void (*run)(void* plan, const float* in, float* out);
    void (*teardown)(void* plan);
} library;

/* ---- TwoFFT ---- */
typedef struct two_plan {
    unsigned log2n;
    float* w;
} two_plan;
static void* two_setup(unsigned log2n) {
    two_plan* p = (two_plan*)malloc(sizeof(two_plan));
    p->log2n = log2n;
    p->w = (float*)bench_alloc(tw_weights_floats(log2n) * sizeof(float));
    tw_weights_fill(log2n, p->w);
    return p;
}
static void two_teardown(void* plan) {
    free(((two_plan*)plan)->w);
    free(plan);
}
static void two_scalar_run(void* plan, const float* in, float* out) {
    const two_plan* p = (const two_plan*)plan;
    tw_scalar_fft(p->log2n, p->w, in, out, 0);
}
#if defined(__aarch64__)
static void two_run(void* plan, const float* in, float* out) {
    const two_plan* p = (const two_plan*)plan;
    tw_a64_fft(p->log2n, p->w, in, out, 0);
}
#endif

/* ---- PFFFT ---- */
#ifdef BENCH_PFFFT
#include "pffft.h"
typedef struct pf_plan {
    PFFFT_Setup* s;
    float* work;
} pf_plan;
static void* pf_setup(unsigned log2n) {
    if (log2n < 4) return NULL;
    pf_plan* p = (pf_plan*)malloc(sizeof(pf_plan));
    p->s = pffft_new_setup(1 << log2n, PFFFT_COMPLEX);
    p->work = (float*)pffft_aligned_malloc(((size_t)2 << log2n) * sizeof(float));
    return p;
}
static void pf_run(void* plan, const float* in, float* out) {
    pf_plan* p = (pf_plan*)plan;
    pffft_transform_ordered(p->s, in, out, p->work, PFFFT_FORWARD);
}
static void pf_teardown(void* plan) {
    pf_plan* p = (pf_plan*)plan;
    pffft_destroy_setup(p->s);
    pffft_aligned_free(p->work);
    free(p);
}
#endif

/* ---- FFTW ---- */
#ifdef BENCH_FFTW
#include <fftw3.h>
static unsigned fftw_flags = FFTW_MEASURE;
static void* fw_setup(unsigned log2n) {
    const size_t n = (size_t)1 << log2n;
    fftwf_complex* a = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_complex* b = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_plan p = fftwf_plan_dft_1d((int)n, a, b, FFTW_FORWARD, log2n > 18 ? FFTW_ESTIMATE : fftw_flags);
    fftwf_free(a);
    fftwf_free(b);
    return p;
}
static void fw_run(void* plan, const float* in, float* out) {
    fftwf_execute_dft((fftwf_plan)plan, (fftwf_complex*)in, (fftwf_complex*)out);
}
static void fw_teardown(void* plan) { fftwf_destroy_plan((fftwf_plan)plan); }
#endif

/* ---- Apple Accelerate (vDSP) ---- */
#ifdef BENCH_VDSP
#include <Accelerate/Accelerate.h>
static void* vd_setup(unsigned log2n) { return vDSP_DFT_Interleaved_CreateSetup(NULL, (vDSP_Length)1 << log2n, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex); }
static void vd_run(void* plan, const float* in, float* out) {
    vDSP_DFT_Interleaved_Execute((vDSP_DFT_Interleaved_Setup)plan, (const DSPComplex*)in, (DSPComplex*)out);
}
static void vd_teardown(void* plan) { vDSP_DFT_Interleaved_DestroySetup((vDSP_DFT_Interleaved_Setup)plan); }
#endif

static const library libraries[] = {
#if defined(__aarch64__)
    {"twofft", two_setup, two_run, two_teardown},
#endif
    {"scalar", two_setup, two_scalar_run, two_teardown},
#ifdef BENCH_PFFFT
    {"pffft", pf_setup, pf_run, pf_teardown},
#endif
#ifdef BENCH_FFTW
    {"fftw", fw_setup, fw_run, fw_teardown},
#endif
#ifdef BENCH_VDSP
    {"vdsp", vd_setup, vd_run, vd_teardown},
#endif
};
enum { library_count = sizeof(libraries) / sizeof(libraries[0]) };

/* Time of one transform in seconds: minimum over batches of ~`batch_time`. */
static double measure(const library* lib, void* plan, const float* in, float* out, double batch_time, int batches) {
    size_t iterations = 1;
    for (;;) {
        const double t0 = bench_now();
        for (size_t i = 0; i < iterations; i++) lib->run(plan, in, out);
        const double t = bench_now() - t0;
        if (t > batch_time / 4) break;
        iterations *= 2;
    }
    double best = 1e30;
    for (int b = 0; b < batches; b++) {
        const double t0 = bench_now();
        for (size_t i = 0; i < iterations; i++) lib->run(plan, in, out);
        const double t = (bench_now() - t0) / (double)iterations;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char** argv) {
    const unsigned lo = argc > 1 ? (unsigned)atoi(argv[1]) : 6;
    const unsigned hi = argc > 2 ? (unsigned)atoi(argv[2]) : 20;
    int enabled[library_count];
    for (int l = 0; l < library_count; l++) {
        enabled[l] = argc <= 3;
        for (int a = 3; a < argc; a++) enabled[l] |= !strcmp(argv[a], libraries[l].name);
    }
    bench_warm_up(0.5);
    printf("clock frequency: %.2f GHz\n", bench_ghz());
    printf("%6s", "log2n");
    for (int l = 0; l < library_count; l++)
        if (enabled[l]) printf(" %12s", libraries[l].name);
    printf("   (ns per transform)\n");
    for (unsigned log2n = lo; log2n <= hi; log2n++) {
        const size_t n = (size_t)1 << log2n;
        float* in = (float*)bench_alloc(2 * n * sizeof(float));
        float* out = (float*)bench_alloc(2 * n * sizeof(float));
        for (size_t i = 0; i < 2 * n; i++) in[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        memset(out, 0, 2 * n * sizeof(float));
        printf("%6u", log2n);
        for (int l = 0; l < library_count; l++) {
            if (!enabled[l]) continue;
            void* plan = libraries[l].setup(log2n);
            if (!plan) {
                printf(" %12s", "-");
                continue;
            }
            const double t = measure(&libraries[l], plan, in, out, 0.02, 15);
            libraries[l].teardown(plan);
            printf(" %12.1f", t * 1e9);
            fflush(stdout);
        }
        printf("\n");
        free(in);
        free(out);
    }
    printf("clock frequency: %.2f GHz\n", bench_ghz());
    return 0;
}
