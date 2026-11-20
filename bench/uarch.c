/* Development tool: measures latency and throughput of the NEON instructions
 * used by the kernels on the current core.
 *   throughput: 12 independent instances per loop iteration
 *   latency:    a dependency chain of 12 instances per loop iteration */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

#define CLOBBER "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","x9","x10","memory","cc"

#define LOOP(body) "movi v30.4s, #0\n movi v31.4s, #0\n movi v29.4s, #0\n 1:\n" body "subs %0, %0, #1\n b.ne 1b\n"

#define KERNEL(name, body)                                                    \
    static __attribute__((noinline)) void name(size_t n, float* mem) {        \
        register float* m __asm__("x1") = mem;                                \
        __asm__ volatile(LOOP(body) : "+r"(n) : "r"(m) : CLOBBER);            \
    }

#define REP12(f) f(0) f(1) f(2) f(3) f(4) f(5) f(6) f(7) f(8) f(9) f(10) f(11)

KERNEL(k_spin, ".rept 12\n add x9, x9, #1\n .endr\n")
#define T(i) "fadd v" #i ".4s, v30.4s, v31.4s\n"
KERNEL(k_fadd_t, REP12(T))
#undef T
#define T(i) "fadd v0.4s, v0.4s, v31.4s\n"
KERNEL(k_fadd_l, REP12(T))
#undef T
#define T(i) "fmul v" #i ".4s, v30.4s, v31.4s\n"
KERNEL(k_fmul_t, REP12(T))
#undef T
#define T(i) "fmul v0.4s, v0.4s, v31.4s\n"
KERNEL(k_fmul_l, REP12(T))
#undef T
#define T(i) "fmul v" #i ".4s, v30.4s, v31.s[1]\n"
KERNEL(k_fmule_t, REP12(T))
#undef T
#define T(i) "fmla v" #i ".4s, v30.4s, v31.4s\n"
KERNEL(k_fmla_t, REP12(T))
#undef T
#define T(i) "fmla v0.4s, v30.4s, v31.4s\n"
KERNEL(k_fmla_lacc, REP12(T))
#undef T
#define T(i) "fmla v0.4s, v0.4s, v31.4s\n"
KERNEL(k_fmla_lmul, REP12(T))
#undef T
#define T(i) "fmla v" #i ".4s, v30.4s, v31.s[1]\n"
KERNEL(k_fmlae_t, REP12(T))
#undef T
#define T(i) "trn1 v" #i ".4s, v30.4s, v31.4s\n"
KERNEL(k_trn_t, REP12(T))
#undef T
#define T(i) "zip1 v" #i ".2d, v30.2d, v31.2d\n"
KERNEL(k_zipd_t, REP12(T))
#undef T
#define T(i) "mov v" #i ".16b, v30.16b\n"
KERNEL(k_mov_t, REP12(T))
#undef T
#define T(i) "mov v1.16b, v0.16b\n mov v0.16b, v1.16b\n"
KERNEL(k_mov_l, T(0) T(0) T(0) T(0) T(0) T(0))
#undef T
/* mov + fadd chain: 6 movs and 6 adds in one chain */
#define T(i) "mov v1.16b, v0.16b\n fadd v0.4s, v1.4s, v31.4s\n"
KERNEL(k_movadd_l, T(0) T(0) T(0) T(0) T(0) T(0))
#undef T
#define T(i) "ldr q" #i ", [x1, #(16*" #i ")]\n"
KERNEL(k_ldr_t, REP12(T))
#undef T
#define T(i) "ldp q" #i ", q20, [x1, #(32*" #i ")]\n"
KERNEL(k_ldp_t, REP12(T))
#undef T
#define T(i) "ld2 {v20.4s, v21.4s}, [x1]\n"
KERNEL(k_ld2_t, REP12(T))
#undef T
#define T(i) "str q30, [x1, #(16*" #i ")]\n"
KERNEL(k_str_t, REP12(T))
#undef T
#define T(i) "stp q30, q31, [x1, #(32*" #i ")]\n"
KERNEL(k_stp_t, REP12(T))
#undef T
#define T(i) "st2 {v30.4s, v31.4s}, [x1]\n"
KERNEL(k_st2_t, REP12(T))
#undef T
/* mixed: 6 fadd + 6 fmul independent */
#define T(i) "fadd v" #i ".4s, v30.4s, v31.4s\n fmul v2" #i ".4s, v30.4s, v31.4s\n"
KERNEL(k_mix_t, T(0) T(1) T(2) T(3) T(4) T(5))
#undef T
#define T(i) "ldr d" #i ", [x1, #(8*" #i ")]\n"
KERNEL(k_ldrd_t, REP12(T))
#undef T

static double cycle_time;

static double run(void (*k)(size_t, float*), float* mem) {
    double best = 1e30;
    for (int i = 0; i < 12; i++) {
        const double t0 = now();
        k(200000, mem);
        const double t = now() - t0;
        if (t < best) best = t;
    }
    return best / 200000;
}

int main(void) {
    float* mem = (float*)calloc(4096, 1);
    /* warm up so that the governor selects the highest frequency */
    const double t0 = now();
    while (now() - t0 < 0.5) k_spin(100000, mem);
    cycle_time = run(k_spin, mem) / 12;
    printf("frequency %.2f GHz\n", 1e-9 / cycle_time);
#define REPORT(name, k) printf("%-28s %6.2f cycles per 12\n", name, run(k, mem) / cycle_time)
    REPORT("fadd throughput", k_fadd_t);
    REPORT("fadd latency", k_fadd_l);
    REPORT("fmul throughput", k_fmul_t);
    REPORT("fmul latency", k_fmul_l);
    REPORT("fmul by element throughput", k_fmule_t);
    REPORT("fmla throughput", k_fmla_t);
    REPORT("fmla latency (accumulator)", k_fmla_lacc);
    REPORT("fmla latency (factor)", k_fmla_lmul);
    REPORT("fmla by element throughput", k_fmlae_t);
    REPORT("fadd+fmul mixed throughput", k_mix_t);
    REPORT("trn1 throughput", k_trn_t);
    REPORT("zip1.2d throughput", k_zipd_t);
    REPORT("mov throughput", k_mov_t);
    REPORT("mov latency", k_mov_l);
    REPORT("6 mov + 6 fadd chain", k_movadd_l);
    REPORT("ldr q throughput", k_ldr_t);
    REPORT("ldr d throughput", k_ldrd_t);
    REPORT("ldp q throughput (24 regs)", k_ldp_t);
    REPORT("ld2 throughput (24 regs)", k_ld2_t);
    REPORT("str q throughput", k_str_t);
    REPORT("stp q throughput (24 regs)", k_stp_t);
    REPORT("st2 throughput (24 regs)", k_st2_t);
    printf("frequency %.2f GHz\n", 12e-9 / run(k_spin, mem));
    return 0;
}
