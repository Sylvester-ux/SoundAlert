// CUSTOM-MODEL VARIANT: built for soundalert.tflite (SoundAlertNet,
// 4-class softmax: 0 baby_cry, 1 siren, 2 fire_alarm, 3 other)
// instead of the 521-class YAMNet. Flash the new model to the
// same "yamnet" partition at 0x610000. mel_features.c is unchanged.
#include <inttypes.h>
#include <math.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "freertos/queue.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mel_features.h"

static const char *TAG = "SOUNDALERT";
static constexpr int SAMPLE_RATE = 16000;
static constexpr size_t kTensorArenaSize = 2 * 1024 * 1024;
static constexpr size_t kI2sFramesPerRead = 160;
static constexpr size_t kI2sWordsPerRead = kI2sFramesPerRead * 2;
static constexpr int kClassCount = 4;
static constexpr int kTopCount = 5;

/* VIBRATION_INTEGRATION_V1 */
static constexpr gpio_num_t kVibrationPin = GPIO_NUM_7;
// FIXED (false-vibration): see process_alert_scores() for the full rules.
// Only THREE wanted events may vibrate: baby cry, fire alarm, siren.
// Telephone / doorbell / car alarm / generic loud sounds are NOT alerts.
static constexpr float kModelThreshold = 0.80f;  // softmax prob for an alert class;
                                                  // tune from training_report.txt
                                                  // (false-alert table) after training.
static constexpr float kReleaseThreshold = 0.35f;
static constexpr int kRequiredDetections = 3;      // was 2 (~2 patches = ~1.9 s is too easy for a transient)
static constexpr int kRequiredReleasePatches = 2; // was defined but never used
static constexpr float kMinPatchDbfs = -45.0f;      // loudness gate: quiet patches can never alert
static constexpr float kMaxClipFraction = 0.02f;    // clipped/distorted patches can never alert

enum class AlertPattern : uint8_t {
    Siren,
    BabyCry,
    FireAlarm
};

// Latest patch loudness, written by capture_task, read by inference_task.
static volatile float s_last_patch_dbfs = -100.0f;
static volatile float s_last_clip_fraction = 0.0f;

static QueueHandle_t s_alert_queue = nullptr;
static QueueHandle_t s_ready_patch_queue = nullptr;
static QueueHandle_t s_free_patch_queue = nullptr;
static int8_t *s_patch_buffers[2] = {nullptr, nullptr};
static float s_model_input_scale = 0.0f;
static int s_model_input_zero_point = 0;
static i2s_chan_handle_t s_rx_handle = nullptr;
static volatile uint32_t s_dropped_patches = 0;

static float class_score(const TfLiteTensor *output, int index)
{
    return (static_cast<int>(output->data.int8[index]) - output->params.zero_point)
           * output->params.scale;
}

static void vibration_task(void *arg)
{
    (void)arg;
    AlertPattern pattern;

    for (;;) {
        if (xQueueReceive(s_alert_queue, &pattern, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        int pulses = 1;
        TickType_t on_time = pdMS_TO_TICKS(500);
        TickType_t off_time = pdMS_TO_TICKS(200);

        switch (pattern) {
        case AlertPattern::Siren:
            pulses = 3;
            on_time = pdMS_TO_TICKS(250);
            off_time = pdMS_TO_TICKS(150);
            break;
        case AlertPattern::BabyCry:
            pulses = 2;
            on_time = pdMS_TO_TICKS(120);
            off_time = pdMS_TO_TICKS(350);
            break;
        case AlertPattern::FireAlarm:
            pulses = 4;
            on_time = pdMS_TO_TICKS(600);
            off_time = pdMS_TO_TICKS(250);
            break;
        }

        for (int i = 0; i < pulses; ++i) {
            gpio_set_level(kVibrationPin, 1);
            vTaskDelay(on_time);
            gpio_set_level(kVibrationPin, 0);

            if (i + 1 < pulses) {
                vTaskDelay(off_time);
            }
        }

        gpio_set_level(kVibrationPin, 0);
    }
}

static bool initialize_vibration(void)
{
    gpio_config_t cfg = {};
    cfg.pin_bit_mask = 1ULL << kVibrationPin;
    cfg.mode = GPIO_MODE_OUTPUT;
    cfg.pull_up_en = GPIO_PULLUP_DISABLE;
    cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    cfg.intr_type = GPIO_INTR_DISABLE;

    esp_err_t err = gpio_config(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Vibration GPIO setup failed: %s",
                 esp_err_to_name(err));
        return false;
    }

    gpio_set_level(kVibrationPin, 0);

    s_alert_queue = xQueueCreate(1, sizeof(AlertPattern));
    if (s_alert_queue == nullptr) {
        ESP_LOGE(TAG, "Failed to create vibration queue");
        return false;
    }

    if (xTaskCreate(vibration_task, "vibration", 2048, nullptr, 3, nullptr)
        != pdPASS) {
        ESP_LOGE(TAG, "Failed to create vibration task");
        vQueueDelete(s_alert_queue);
        s_alert_queue = nullptr;
        return false;
    }

    ESP_LOGI(TAG, "Vibration controller ready on GPIO7");
    return true;
}

// ---------------------------------------------------------------------------
// CUSTOM-MODEL alert decision. The model is a 4-class softmax trained for
// this device (0 baby_cry, 1 siren, 2 fire_alarm, 3 other):
// probabilities sum to 1, so "other" winning IS the unknown
// rejection — no 521-class scan needed. The loudness/clipping gates,
// 3-patch streak, cooldown and re-arm rules from the YAMNet fix stay.
// ---------------------------------------------------------------------------
static void process_alert_scores(const TfLiteTensor *output)
{
    static int candidate_streak = 0;
    static int release_streak = 0;
    static bool armed = true;
    static int64_t last_alert_us = 0;
    static AlertPattern previous_candidate = AlertPattern::Siren;

    const float p_baby  = class_score(output, 0);
    const float p_siren = class_score(output, 1);
    const float p_fire  = class_score(output, 2);
    const float p_other = class_score(output, 3);

    AlertPattern candidate = AlertPattern::BabyCry;
    float best = p_baby;
    if (p_siren > best) { best = p_siren; candidate = AlertPattern::Siren; }
    if (p_fire  > best) { best = p_fire;  candidate = AlertPattern::FireAlarm; }
    const char *name = candidate == AlertPattern::BabyCry ? "BABY_CRY" :
                       candidate == AlertPattern::Siren ? "SIREN" :
                       "FIRE_ALARM";

    if (s_last_patch_dbfs < kMinPatchDbfs) {
        ESP_LOGI(TAG, "DECISION=REJECT reason=TOO_QUIET dbfs=%.1f baby=%.3f siren=%.3f fire=%.3f other=%.3f",
                 s_last_patch_dbfs, p_baby, p_siren, p_fire, p_other);
        candidate_streak = 0;
        if (++release_streak >= kRequiredReleasePatches) armed = true;
        return;
    }
    if (s_last_clip_fraction > kMaxClipFraction) {
        ESP_LOGI(TAG, "DECISION=REJECT reason=CLIPPED clip=%.3f baby=%.3f siren=%.3f fire=%.3f other=%.3f",
                 s_last_clip_fraction, p_baby, p_siren, p_fire, p_other);
        candidate_streak = 0;
        return;
    }
    if (best < kModelThreshold || p_other >= best) {
        ESP_LOGI(TAG, "DECISION=REJECT reason=%s baby=%.3f siren=%.3f fire=%.3f other=%.3f best=%s/%.3f threshold=%.2f streak=0",
                 p_other >= best ? "OTHER_WINS" : "LOW_SCORE",
                 p_baby, p_siren, p_fire, p_other, name, best, kModelThreshold);
        candidate_streak = 0;
        if (++release_streak >= kRequiredReleasePatches) armed = true;
        return;
    }
    release_streak = 0;
    if (!armed) {
        ESP_LOGI(TAG, "DECISION=REJECT reason=NOT_REARMED candidate=%s score=%.3f", name, best);
        return;
    }
    if (candidate_streak == 0 || candidate != previous_candidate) {
        previous_candidate = candidate;
        candidate_streak = 1;
        ESP_LOGI(TAG, "DECISION=WAIT reason=FIRST_WIN candidate=%s score=%.3f other=%.3f streak=1/%d",
                 name, best, p_other, kRequiredDetections);
        return;
    }
    if (++candidate_streak < kRequiredDetections) {
        ESP_LOGI(TAG, "DECISION=WAIT reason=STREAK candidate=%s score=%.3f streak=%d/%d",
                 name, best, candidate_streak, kRequiredDetections);
        return;
    }
    constexpr int64_t kCooldownUs = 8000000LL;
    const int64_t now_us = esp_timer_get_time();
    if (now_us - last_alert_us < kCooldownUs) {
        ESP_LOGI(TAG, "DECISION=REJECT reason=COOLDOWN candidate=%s remaining_ms=%" PRId64,
                 name, (kCooldownUs - (now_us - last_alert_us)) / 1000);
        candidate_streak = 0;
        return;
    }
    if (xQueueSend(s_alert_queue, &candidate, 0) == pdTRUE) {
        ESP_LOGW(TAG, "DECISION=ALERT pattern=%s baby=%.3f siren=%.3f fire=%.3f other=%.3f dbfs=%.1f",
                 name, p_baby, p_siren, p_fire, p_other, s_last_patch_dbfs);
        last_alert_us = now_us;
        armed = false;
        release_streak = 0;
    } else {
        ESP_LOGW(TAG, "DECISION=REJECT reason=VIBRATION_QUEUE_BUSY candidate=%s", name);
    }
    candidate_streak = 0;
}

static uint8_t *tensor_arena = nullptr;
static void *interpreter_mem = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static tflite::MicroMutableOpResolver<10> resolver;
static const void *model_data = nullptr;
static esp_partition_mmap_handle_t model_handle = 0;

struct TargetClass {
    int index;
    const char *name;
};

// YAMNet class indices from the project's class map.
static constexpr TargetClass kTargetClasses[] = {
    {20, "Baby cry"},
    {304, "Car alarm"},
    {317, "Police siren"},
    {318, "Ambulance siren"},
    {319, "Fire engine siren"},
    {349, "Doorbell"},
    {382, "Alarm"},
    {389, "Alarm clock"},
    {390, "Siren"},
    {391, "Civil defense siren"},
    {393, "Smoke detector"},
    {394, "Fire alarm"},
};

static void print_tensor(const char *name, const TfLiteTensor *tensor)
{
    if (tensor == nullptr || tensor->dims == nullptr) {
        ESP_LOGE(TAG, "%s tensor is null", name);
        return;
    }
    printf("%s: type=%d, dimensions=[", name, tensor->type);
    for (int i = 0; i < tensor->dims->size; ++i) {
        printf("%s%d", i == 0 ? "" : ",", tensor->dims->data[i]);
    }
    printf("], scale=%.10f, zero_point=%" PRId32 "\n",
           tensor->params.scale, tensor->params.zero_point);
}

static bool check_heap_integrity(const char *stage)
{
    ESP_LOGI(TAG, "Checking internal RAM heap: %s", stage);

    const bool internal_ok = heap_caps_check_integrity(
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, true);

    ESP_LOGI(TAG, "Internal RAM heap [%s]: %s",
             stage, internal_ok ? "OK" : "CORRUPTION DETECTED");

    ESP_LOGI(TAG, "Checking PSRAM heap: %s", stage);

    const bool psram_ok = heap_caps_check_integrity(
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, true);

    ESP_LOGI(TAG, "PSRAM heap [%s]: %s",
             stage, psram_ok ? "OK" : "CORRUPTION DETECTED");

    return internal_ok && psram_ok;
}

static bool initialize_model()
{
    ESP_LOGI(TAG, "Locating YAMNet flash partition");
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        static_cast<esp_partition_subtype_t>(0x40),
        "yamnet");
    if (partition == nullptr) {
        ESP_LOGE(TAG, "yamnet partition not found");
        return false;
    }

    ESP_LOGI(TAG, "Model partition offset=0x%" PRIx32 ", size=%" PRIu32,
             partition->address, partition->size);
    esp_err_t err = esp_partition_mmap(
        partition, 0, partition->size, ESP_PARTITION_MMAP_DATA,
        &model_data, &model_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Model mmap failed: %s", esp_err_to_name(err));
        return false;
    }

    const uint8_t *bytes = static_cast<const uint8_t *>(model_data);
    if (partition->size < 8 || bytes[4] != 'T' || bytes[5] != 'F' ||
        bytes[6] != 'L' || bytes[7] != '3') {
        ESP_LOGE(TAG, "Invalid model: TFL3 identifier missing");
        return false;
    }

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch");
        return false;
    }

    if (resolver.AddReshape() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite operator registration failed");
        return false;
    }

    tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Could not allocate PSRAM tensor arena");
        return false;
    }

    interpreter_mem = heap_caps_malloc(
        sizeof(tflite::MicroInterpreter), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (interpreter_mem == nullptr) {
        ESP_LOGE(TAG, "Could not allocate interpreter object");
        return false;
    }

    interpreter = new (interpreter_mem) tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }

    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);
    if (input == nullptr || output == nullptr ||
        input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        !((input->dims->size == 3 && input->dims->data[0] == 1 &&
           input->dims->data[1] == MEL_FRAMES_PATCH &&
           input->dims->data[2] == MEL_BANDS) ||
          (input->dims->size == 4 && input->dims->data[0] == 1 &&
           input->dims->data[1] == MEL_FRAMES_PATCH &&
           input->dims->data[2] == MEL_BANDS && input->dims->data[3] == 1)) ||
        output->dims->size != 2 || output->dims->data[0] != 1 ||
        output->dims->data[1] != kClassCount || input->params.scale <= 0.0f ||
        output->params.scale <= 0.0f) {
        ESP_LOGE(TAG, "Unexpected YAMNet tensor shape, type, or quantization");
        print_tensor("INPUT", input);
        print_tensor("OUTPUT", output);
        return false;
    }

    print_tensor("INPUT", input);
    print_tensor("OUTPUT", output);
    ESP_LOGI(TAG, "Model initialized; free internal=%u, PSRAM=%u bytes",
             static_cast<unsigned>(heap_caps_get_free_size(
                 MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_free_size(
                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
    return true;
}

static bool initialize_microphone(i2s_chan_handle_t *rx_handle)
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    esp_err_t err = i2s_new_channel(&chan_cfg, nullptr, rx_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(err));
        return false;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_4,
            .ws = GPIO_NUM_5,
            .dout = I2S_GPIO_UNUSED,
            .din = GPIO_NUM_6,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    err = i2s_channel_init_std_mode(*rx_handle, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S init failed: %s", esp_err_to_name(err));
        i2s_del_channel(*rx_handle);
        *rx_handle = nullptr;
        return false;
    }
    err = i2s_channel_enable(*rx_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S enable failed: %s", esp_err_to_name(err));
        i2s_del_channel(*rx_handle);
        *rx_handle = nullptr;
        return false;
    }

    ESP_LOGI(TAG, "INMP441 active: BCLK=GPIO4 WS=GPIO5 SD=GPIO6; L/R=GND");
    return true;
}

static void report_scores(const TfLiteTensor *output, int64_t elapsed_us)
{
    ESP_LOGI(TAG,
             "Inference: %.1f ms | baby %.3f | siren %.3f | fire %.3f | other %.3f",
             elapsed_us / 1000.0,
             class_score(output, 0), class_score(output, 1),
             class_score(output, 2), class_score(output, 3));
}

static void inference_task(void *arg)
{
    (void)arg;
    int8_t *patch = nullptr;
    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);
    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Inference task: model tensors unavailable");
        vTaskDelete(nullptr);
        return;
    }

    for (;;) {
        if (xQueueReceive(s_ready_patch_queue, &patch, portMAX_DELAY) != pdTRUE || patch == nullptr) continue;
        memcpy(input->data.int8, patch, MEL_FRAMES_PATCH * MEL_BANDS);
        const int64_t start_us = esp_timer_get_time();
        const TfLiteStatus status = interpreter->Invoke();
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "YAMNet Invoke failed: status=%d", static_cast<int>(status));
        } else {
            ESP_LOGI(TAG, "Inference complete: %.1f ms; dropped patches=%u",
                     elapsed_us / 1000.0, static_cast<unsigned>(s_dropped_patches));
            process_alert_scores(output);
            report_scores(output, elapsed_us);
        }
        if (xQueueSend(s_free_patch_queue, &patch, portMAX_DELAY) != pdTRUE) {
            ESP_LOGE(TAG, "Could not return inference patch buffer to pool");
        }
        patch = nullptr;

        // Yield so other FreeRTOS tasks can run between inferences.
        vTaskDelay(1);
    }
}

static void capture_task(void *arg)
{
    (void)arg;
    int32_t i2s_words[kI2sWordsPerRead];
    int16_t pcm[MEL_HOP_SAMPLES];
    size_t pcm_fill = 0;
    uint64_t total_samples = 0;
    uint64_t patches = 0;

    int16_t pcm_min = INT16_MAX;
    int16_t pcm_max = INT16_MIN;
    uint64_t pcm_sum_sq = 0;
    uint32_t pcm_clipped = 0;
    unsigned pcm_hops = 0;

    // FIXED: per-patch loudness accumulators feeding the alert loudness gate.
    uint64_t patch_sum_sq = 0;
    uint32_t patch_samples = 0;
    uint32_t patch_clipped = 0;

    int64_t last_report_us = esp_timer_get_time();

    ESP_LOGI(TAG, "Capture task running on core %d; inference is a separate task", xPortGetCoreID());
    for (;;) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(s_rx_handle, i2s_words, sizeof(i2s_words), &bytes_read, 1000);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "I2S read failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        const size_t words_read = bytes_read / sizeof(int32_t);
        if (words_read < 2) continue;

        for (size_t i = 0; i + 1 < words_read; i += 2) {
            pcm[pcm_fill++] = static_cast<int16_t>(static_cast<uint32_t>(i2s_words[i]) >> 16);
            ++total_samples;
            if (pcm_fill != MEL_HOP_SAMPLES) continue;
            pcm_fill = 0;

            // Measure the actual int16 PCM delivered to the feature extractor.
            for (size_t j = 0; j < MEL_HOP_SAMPLES; ++j) {
                const int32_t sample = pcm[j];

                if (sample < pcm_min) pcm_min = static_cast<int16_t>(sample);
                if (sample > pcm_max) pcm_max = static_cast<int16_t>(sample);

                pcm_sum_sq += static_cast<uint64_t>(
                    static_cast<int64_t>(sample) * sample);

                patch_sum_sq += static_cast<uint64_t>(
                    static_cast<int64_t>(sample) * sample);
                ++patch_samples;

                if (sample >= 32760 || sample <= -32760) {
                    ++pcm_clipped;
                    ++patch_clipped;
                }
            }

            if (++pcm_hops >= 100) {
                const float rms = sqrtf(
                    static_cast<float>(pcm_sum_sq) /
                    (100.0f * MEL_HOP_SAMPLES));

                ESP_LOGI(TAG,
                         "PCM 1s: min=%d max=%d RMS=%.1f clipped=%u/16000",
                         static_cast<int>(pcm_min),
                         static_cast<int>(pcm_max),
                         static_cast<double>(rms),
                         static_cast<unsigned>(pcm_clipped));

                pcm_min = INT16_MAX;
                pcm_max = INT16_MIN;
                pcm_sum_sq = 0;
                pcm_clipped = 0;
                pcm_hops = 0;
            }

            const int patch_ready = mel_features_push(pcm, MEL_HOP_SAMPLES);
            if (patch_ready < 0) {
                ESP_LOGE(TAG, "Feature extraction failed; capture task stopping");
                vTaskDelete(nullptr);
                return;
            }
            if (patch_ready == 1) {
                // FIXED: publish this patch's loudness for the alert gate.
                if (patch_samples > 0) {
                    const float prms = sqrtf(static_cast<float>(patch_sum_sq) /
                                             static_cast<float>(patch_samples));
                    s_last_patch_dbfs = 20.0f * log10f((prms < 1.0f ? 1.0f : prms) / 32768.0f);
                    s_last_clip_fraction = static_cast<float>(patch_clipped) /
                                           static_cast<float>(patch_samples);
                }
                patch_sum_sq = 0;
                patch_samples = 0;
                patch_clipped = 0;
                int8_t *patch = nullptr;
                if (xQueueReceive(s_free_patch_queue, &patch, 0) != pdTRUE || patch == nullptr) {
                    ++s_dropped_patches;
                    ESP_LOGW(TAG, "PATCH DROPPED: inference backlog; total_dropped=%u", static_cast<unsigned>(s_dropped_patches));
                    continue;
                }
                mel_features_take_patch(patch, s_model_input_scale, s_model_input_zero_point);
                if (xQueueSend(s_ready_patch_queue, &patch, 0) != pdTRUE) {
                    ++s_dropped_patches;
                    ESP_LOGW(TAG, "PATCH DROPPED: ready queue full; total_dropped=%u", static_cast<unsigned>(s_dropped_patches));
                    xQueueSend(s_free_patch_queue, &patch, 0);
                } else {
                    ++patches;
                    ESP_LOGI(TAG, "Audio patch queued #%" PRIu64 " samples=%" PRIu64,
                             patches, total_samples);
                }
            }
        }
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_report_us >= 10000000) {
            ESP_LOGI(TAG, "Capture status: samples=%" PRIu64 ", queued_patches=%" PRIu64 ", dropped_patches=%u",
                     total_samples, patches, static_cast<unsigned>(s_dropped_patches));
            last_report_us = now_us;
        }
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "SoundAlert live-audio YAMNet starting (continuous capture + inference task)");

    // Initialize the expensive model/features before enabling I2S, so DMA does
    // not sit unread while model allocation and filterbank construction run.
    if (!initialize_model()) {
        ESP_LOGE(TAG, "Model initialization failed");
        return;
    }
    TfLiteTensor *input = interpreter->input(0);
    if (input == nullptr) {
        ESP_LOGE(TAG, "Model input tensor unavailable");
        return;
    }
    s_model_input_scale = input->params.scale;
    s_model_input_zero_point = input->params.zero_point;

    const esp_err_t mel_err = mel_features_init();
    if (mel_err != ESP_OK) {
        ESP_LOGE(TAG, "Mel feature initialization failed: %s", esp_err_to_name(mel_err));
        return;
    }
    if (!initialize_vibration()) {
        ESP_LOGE(TAG, "Vibration initialization failed");
        return;
    }

    s_ready_patch_queue = xQueueCreate(2, sizeof(int8_t *));
    s_free_patch_queue = xQueueCreate(2, sizeof(int8_t *));
    if (s_ready_patch_queue == nullptr || s_free_patch_queue == nullptr) {
        ESP_LOGE(TAG, "Failed to create inference queues");
        return;
    }
    for (int i = 0; i < 2; ++i) {
        s_patch_buffers[i] = static_cast<int8_t *>(heap_caps_malloc(
            MEL_FRAMES_PATCH * MEL_BANDS, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (s_patch_buffers[i] == nullptr ||
            xQueueSend(s_free_patch_queue, &s_patch_buffers[i], 0) != pdTRUE) {
            ESP_LOGE(TAG, "Failed to allocate/seed patch buffer %d", i);
            return;
        }
    }

    // Start I2S only after setup is complete; capture begins immediately after
    // the inference task is ready to receive feature patches.
    if (!initialize_microphone(&s_rx_handle)) {
        ESP_LOGE(TAG, "Microphone initialization failed");
        return;
    }

    ESP_LOGI(TAG, "Task plan: capture pinned to Core 0; YAMNet inference pinned to Core 1");
    ESP_LOGI(TAG, "Inference duration previously measured ~821 ms; patch stride set to 96 frames (~960 ms)");
    if (xTaskCreatePinnedToCore(inference_task, "yamnet_infer", 8192, nullptr, 4, nullptr, 1) != pdPASS) {
        ESP_LOGE(TAG, "Failed to create inference task");
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        return;
    }
    if (xTaskCreatePinnedToCore(capture_task, "audio_capture", 6144, nullptr, 5, nullptr, 0) != pdPASS) {
        ESP_LOGE(TAG, "Failed to create capture task");
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        return;
    }
}
