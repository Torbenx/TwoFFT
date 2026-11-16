/* TwoFFT - fast power-of-two single precision FFT (preliminary development interface). */
#ifndef TWOFFT_H
#define TWOFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TWOFFT_MAX_LOG2_SIZE 30

typedef struct twofft_context twofft_context;

twofft_context *twofft_context_create(void);
void twofft_context_destroy(twofft_context *ctx);

/* Out-of-place complex transforms of 2^log2_size points. `in` and `out` hold interleaved
 * (re, im) pairs and must not overlap. Returns 0 on success. */
int twofft_forward(twofft_context *ctx, const float *in, float *out, unsigned log2_size);
int twofft_inverse(twofft_context *ctx, const float *in, float *out, unsigned log2_size);

#ifdef __cplusplus
}
#endif

#endif /* TWOFFT_H */
