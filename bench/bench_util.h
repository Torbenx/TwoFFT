/* Shared helpers of the benchmark programs. */
#ifndef TW_BENCH_UTIL_H
#define TW_BENCH_UTIL_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Wall clock in seconds with nanosecond resolution. */
static double bench_now(void) {
#if defined(__APPLE__)
    return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#endif
}

static void* bench_alloc(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64)) abort();
    return p;
}

/* Runs a dependency chain of 32 additions `n` times: 32 * n cycles. */
static __attribute__((noinline)) size_t bench_spin(size_t n, size_t x) {
#if defined(__aarch64__)
    __asm__ volatile("1:\n"
                     ".rept 32\n add %1, %1, #1\n .endr\n"
                     "subs %0, %0, #1\n b.ne 1b\n"
                     : "+r"(n), "+r"(x) : : "cc");
#elif defined(__x86_64__)
    __asm__ volatile("1:\n"
                     ".rept 32\n add $1, %1\n .endr\n"
                     "sub $1, %0\n jnz 1b\n"
                     : "+r"(n), "+r"(x) : : "cc");
#else
    for (; n; n--) x += 32;
#endif
    return x;
}

/* Current clock frequency of the core in GHz (0 if unknown). */
static double bench_ghz(void) {
#if defined(__aarch64__) || defined(__x86_64__)
    double best = 1e30;
    for (int i = 0; i < 10; i++) {
        const double t0 = bench_now();
        bench_spin(20000, 0);
        const double t = bench_now() - t0;
        if (t < best) best = t;
    }
    return 32 * 20000 / best * 1e-9;
#else
    return 0;
#endif
}

/* Keeps the core busy so that frequency scaling selects its highest step. */
static void bench_warm_up(double seconds) {
    const double t0 = bench_now();
    while (bench_now() - t0 < seconds) bench_spin(100000, 0);
}

#endif
