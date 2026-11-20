/* TwoFFT - transforms of size 1, 2 and 4 in portable C. */
#ifndef TW_SMALL_H
#define TW_SMALL_H

#include "tw_common.h"

/* Transform of size 2^log2n for log2n <= 2. */
TW_INLINE void tw_small_fft(unsigned log2n, const float* in, float* out, int backward) {
    if (log2n == 0) {
        out[0] = in[0];
        out[1] = in[1];
    } else if (log2n == 1) {
        const float ar = in[0], ai = in[1], br = in[2], bi = in[3];
        out[0] = ar + br;
        out[1] = ai + bi;
        out[2] = ar - br;
        out[3] = ai - bi;
    } else {
        const float s0r = in[0] + in[4], s0i = in[1] + in[5];
        const float d0r = in[0] - in[4], d0i = in[1] - in[5];
        const float s1r = in[2] + in[6], s1i = in[3] + in[7];
        float d1r = in[2] - in[6], d1i = in[3] - in[7];
        if (backward) d1r = -d1r, d1i = -d1i;
        out[0] = s0r + s1r;
        out[1] = s0i + s1i;
        out[2] = d0r + d1i; /* forward: d0 - i * d1 */
        out[3] = d0i - d1r;
        out[4] = s0r - s1r;
        out[5] = s0i - s1i;
        out[6] = d0r - d1i;
        out[7] = d0i + d1r;
    }
}

#endif /* TW_SMALL_H */
