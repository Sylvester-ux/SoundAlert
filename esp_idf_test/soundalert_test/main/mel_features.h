#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#define MEL_BANDS          64
#define MEL_FRAMES_PATCH   96
#define MEL_HOP_SAMPLES    160      /* 10 ms @ 16 kHz */
#define MEL_WIN_SAMPLES    400      /* 25 ms @ 16 kHz, zero-padded to FFT size */
#define MEL_FFT_SIZE       512
#define MEL_FMIN_HZ        125.0f
#define MEL_FMAX_HZ        7500.0f

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t mel_features_init(void);

/* Push hop-many samples (MEL_HOP_SAMPLES). If this completes a new frame
 * that also completes a patch (every 48 new frames after warm-up), the
 * caller gets a 1. Patches are read via mel_features_take_patch(). */
int mel_features_push(const int16_t *samples, size_t n);

/* Copy the latest 96×64 log-mel patch into `dst_int8` (length
 * MEL_FRAMES_PATCH * MEL_BANDS bytes). Quantization parameters (scale,
 * zero-point) come from the YAMNet input tensor and are supplied here by
 * the caller so one patch extractor can serve multiple models. */
void mel_features_take_patch(int8_t *dst_int8, float input_scale, int input_zero_point);

#ifdef __cplusplus
}
#endif
