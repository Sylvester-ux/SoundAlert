#include <stdbool.h>

#include "mel_features.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_dsp.h"

static const char *TAG = "mel";

#define FFT_BINS (MEL_FFT_SIZE / 2 + 1)

typedef struct {
    float weight;
    uint16_t fft_bin;
    uint16_t mel_band;
} mel_entry_t;

static float *s_hann;                    /* length MEL_WIN_SAMPLES */
static float *s_frame;                   /* length MEL_FFT_SIZE */
static float *s_fft_work;                /* length 2 * MEL_FFT_SIZE (complex) */
static mel_entry_t *s_mel_triplets;      /* sparse mel filterbank */
static uint32_t s_mel_entries;

static float *s_ring;                    /* MEL_FRAMES_PATCH x MEL_BANDS log-mel ring */
static uint32_t s_ring_pos;              /* next column to fill */
static uint32_t s_frames_since_patch;
static SemaphoreHandle_t s_ring_lock;

static float *s_audio_accum;             /* rolling sample window */
static uint32_t s_audio_accum_fill;

static float hz_to_mel(float hz)
{
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

static bool check_internal_heap(const char *stage)
{
    bool intact = heap_caps_check_integrity(
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, true);

    ESP_LOGI(TAG, "Internal heap [%s]: %s",
             stage, intact ? "OK" : "CORRUPTION DETECTED");

    return intact;
}

static esp_err_t build_mel_filterbank(void)
{
    float mel_min = hz_to_mel(MEL_FMIN_HZ);
    float mel_max = hz_to_mel(MEL_FMAX_HZ);
    float mel_pts[MEL_BANDS + 2];
    for (int i = 0; i < MEL_BANDS + 2; ++i) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(MEL_BANDS + 1);
    }
    float hz_pts[MEL_BANDS + 2];
    float bin_pts[MEL_BANDS + 2];
    for (int i = 0; i < MEL_BANDS + 2; ++i) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
        bin_pts[i] = hz_pts[i] * (float)MEL_FFT_SIZE / 16000.0f;
    }

    mel_entry_t *tmp = heap_caps_malloc(
        sizeof(mel_entry_t) * MEL_BANDS * 64, MALLOC_CAP_SPIRAM);
    if (!tmp) return ESP_ERR_NO_MEM;
    uint32_t n = 0;

    for (int m = 1; m <= MEL_BANDS; ++m) {
        float left  = bin_pts[m - 1];
        float centr = bin_pts[m];
        float right = bin_pts[m + 1];
        for (int k = 0; k < FFT_BINS; ++k) {
            float w;
            if (k < left || k > right) {
                w = 0.0f;
            } else if (k < centr) {
                w = (k - left) / (centr - left);
            } else {
                w = (right - k) / (right - centr);
            }
            if (w > 0.0f) {
                tmp[n++] = (mel_entry_t){ .weight = w, .fft_bin = (uint16_t)k, .mel_band = (uint16_t)(m - 1) };
            }
        }
    }

    s_mel_triplets = heap_caps_malloc(sizeof(mel_entry_t) * n, MALLOC_CAP_SPIRAM);
    if (!s_mel_triplets) {
        heap_caps_free(tmp);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Filterbank: before copy");

    memcpy(s_mel_triplets, tmp, sizeof(mel_entry_t) * n);
    s_mel_entries = n;

    ESP_LOGI(TAG, "Filterbank: after copy");

    heap_caps_free(tmp);

    ESP_LOGI(TAG, "Filterbank: after free");

    ESP_LOGI(TAG, "mel filterbank: %u nonzero triplets across %d bands",
             (unsigned)n, MEL_BANDS);
    return ESP_OK;
}

esp_err_t mel_features_init(void)
{
    esp_err_t err = dsps_fft2r_init_fc32(NULL, MEL_FFT_SIZE);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "fft init failed: %d", err);
        return err;
    }
    check_internal_heap("after FFT initialization");

    s_hann       = heap_caps_malloc(MEL_WIN_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
    s_frame      = heap_caps_calloc(MEL_FFT_SIZE, sizeof(float), MALLOC_CAP_SPIRAM);
    s_fft_work   = heap_caps_malloc(2 * MEL_FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    s_ring       = heap_caps_calloc(MEL_FRAMES_PATCH * MEL_BANDS, sizeof(float), MALLOC_CAP_SPIRAM);
    s_audio_accum = heap_caps_calloc(MEL_WIN_SAMPLES, sizeof(float), MALLOC_CAP_SPIRAM);
    if (!s_hann || !s_frame || !s_fft_work || !s_ring || !s_audio_accum) {
        ESP_LOGE(TAG, "alloc failed");
        return ESP_ERR_NO_MEM;
    }
check_internal_heap("after mel buffer allocations");
    /* TensorFlow STFT uses a periodic Hann window: denominator is N, not N-1. */
    for (int i = 0; i < MEL_WIN_SAMPLES; ++i) {
        s_hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)MEL_WIN_SAMPLES));
    }

    s_ring_pos = 0;
    s_frames_since_patch = 0;
    s_audio_accum_fill = 0;
    s_ring_lock = xSemaphoreCreateMutex();
    if (s_ring_lock == NULL) {
        ESP_LOGE(TAG, "Failed to create ring mutex");
        return ESP_ERR_NO_MEM;
    }
    check_internal_heap("before filterbank construction");
    return build_mel_filterbank();
}

static esp_err_t compute_one_frame(const float *samples, float *log_mel_out)
{
    for (int i = 0; i < MEL_WIN_SAMPLES; ++i) {
        s_fft_work[2 * i]     = samples[i] * s_hann[i];
        s_fft_work[2 * i + 1] = 0.0f;
    }
    for (int i = MEL_WIN_SAMPLES; i < MEL_FFT_SIZE; ++i) {
        s_fft_work[2 * i]     = 0.0f;
        s_fft_work[2 * i + 1] = 0.0f;
    }

    esp_err_t err = dsps_fft2r_fc32(s_fft_work, MEL_FFT_SIZE);
    if (err != ESP_OK) return err;
    err = dsps_bit_rev_fc32(s_fft_work, MEL_FFT_SIZE);
    if (err != ESP_OK) return err;
    /*
     * Keep the ordinary complex FFT bins. dsps_cplx2reC_fc32() is NOT a
     * generic real-FFT unpacker: it converts an FFT of two real signals
     * packed into the real/imaginary components. Our input is one real
     * microphone signal with an all-zero imaginary component, so applying
     * it here corrupts the spectrum used by the mel filterbank.
     */

    /* YAMNet reference uses magnitude (|X|), not power (|X|²).
     * Feeding power to the mel filterbank produces a wholly different
     * distribution from what the INT8 PTQ was calibrated on — the model
     * then collapses to its near-zero baseline (int8 code ≈ 0.066 post-
     * dequant). See docs/research/deep-analysis-20260423.md §Q1. */
    float magnitude[FFT_BINS];
    for (int k = 0; k < FFT_BINS; ++k) {
        float re = s_fft_work[2 * k];
        float im = s_fft_work[2 * k + 1];
        /* FIXED (false-vibration): FFT-normalise the magnitude.
         * The unnormalised magnitude (peaks ~100 for a loud tone) pushes
         * log-mel peaks to ~+4.5, far above the distribution this INT8
         * model was PTQ-calibrated on (synthetic Gaussian patches,
         * mean -5, std 3 -- see the model card). Internal activations
         * saturate, ~dozens of unrelated classes pin at ~0.65-0.75, and
         * ANY sound (silence included: emergency max was 0.672 on a
         * constant -6.9 patch) crosses the old 0.60 threshold.
         * Host A/B with this exact mel code + the repo model:
         * unnormalised, a synthetic siren's top-3 was Fireworks / Bass
         * guitar / Biting; with 1/FFT_SIZE scaling the same siren's
         * top-3 is Siren / Fire-engine siren / Car alarm, and music /
         * noise / silence stop masquerading as alerts.
         * VALIDATE ON DEVICE with real sounds before trusting it:
         * play a real siren + fire alarm + baby cry and negatives
         * (TV, music, speech, kettle) and compare DECISION logs. */
        magnitude[k] = sqrtf(re * re + im * im) / (float)MEL_FFT_SIZE;
    }

    float mel_energy[MEL_BANDS] = {0};
    const mel_entry_t *t = s_mel_triplets;
    for (uint32_t i = 0; i < s_mel_entries; ++i) {
        mel_energy[t[i].mel_band] += t[i].weight * magnitude[t[i].fft_bin];
    }
    /* YAMNet log_offset=0.001; 1e-10 sent silent bins to ~-23 (out of
     * distribution). 0.001 floors them at ~-6.9, matching the training
     * distribution. */
    for (int m = 0; m < MEL_BANDS; ++m) {
        log_mel_out[m] = logf(mel_energy[m] + 0.001f);
    }
    return ESP_OK;
}

int mel_features_push(const int16_t *samples, size_t n)
{
    int new_patch = 0;
    size_t pos = 0;
    while (pos < n) {
        size_t room = MEL_WIN_SAMPLES - s_audio_accum_fill;
        size_t take = (n - pos) < room ? (n - pos) : room;
        for (size_t i = 0; i < take; ++i) {
            s_audio_accum[s_audio_accum_fill + i] = (float)samples[pos + i] / 32768.0f;
        }
        s_audio_accum_fill += take;
        pos += take;

        if (s_audio_accum_fill < MEL_WIN_SAMPLES) break;

        float log_mel[MEL_BANDS];
        esp_err_t frame_err = compute_one_frame(s_audio_accum, log_mel);
        if (frame_err != ESP_OK) {
            ESP_LOGE(TAG, "FFT feature computation failed: %s",
                     esp_err_to_name(frame_err));
            return -1;
        }

        xSemaphoreTake(s_ring_lock, portMAX_DELAY);
        float *col = &s_ring[s_ring_pos * MEL_BANDS];
        memcpy(col, log_mel, MEL_BANDS * sizeof(float));
        s_ring_pos = (s_ring_pos + 1) % MEL_FRAMES_PATCH;
        s_frames_since_patch++;
        xSemaphoreGive(s_ring_lock);

        /*
         * Wait for 96 real mel frames before the first patch.
         * Thereafter, emit a patch every 96 frames so inference has enough time.
         */
        if (s_frames_since_patch >= MEL_FRAMES_PATCH &&
            (s_frames_since_patch - MEL_FRAMES_PATCH) % 96 == 0) {
            new_patch = 1;
        }

        memmove(s_audio_accum, s_audio_accum + MEL_HOP_SAMPLES,
                (MEL_WIN_SAMPLES - MEL_HOP_SAMPLES) * sizeof(float));
        s_audio_accum_fill = MEL_WIN_SAMPLES - MEL_HOP_SAMPLES;
    }
    return new_patch;
}

void mel_features_take_patch(int8_t *dst_int8, float input_scale, int input_zero_point)
{
    xSemaphoreTake(s_ring_lock, portMAX_DELAY);
    for (int f = 0; f < MEL_FRAMES_PATCH; ++f) {
        uint32_t src_col = (s_ring_pos + f) % MEL_FRAMES_PATCH;
        const float *src = &s_ring[src_col * MEL_BANDS];
        int8_t *dst = &dst_int8[f * MEL_BANDS];
        for (int m = 0; m < MEL_BANDS; ++m) {
            int32_t q = (int32_t)lroundf(src[m] / input_scale) + input_zero_point;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dst[m] = (int8_t)q;
        }
    }
    xSemaphoreGive(s_ring_lock);
}
