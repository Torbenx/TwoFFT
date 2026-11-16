/* Shared helpers of the benchmark programs. */
#ifndef TWOFFT_BENCH_UTIL_H
#define TWOFFT_BENCH_UTIL_H

#include <stdio.h>
#include <time.h>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__APPLE__)
#include <pthread.h>
#endif

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Pins the thread to `cpu` (Linux / Android; cpu < 0: no pinning), asks for a performance
 * core on macOS, and spins for a moment so that frequency scaling has settled. */
static void bench_prepare_cpu(int cpu)
{
    volatile double sink = 0;
    const double t0 = now_ns();
#if defined(__linux__)
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0)
            fprintf(stderr, "warning: could not pin to cpu %d\n", cpu);
    }
#elif defined(__APPLE__)
    (void)cpu;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
    (void)cpu;
#endif
    while (now_ns() - t0 < 3e8)
        sink = sink + 1;
}

#endif
