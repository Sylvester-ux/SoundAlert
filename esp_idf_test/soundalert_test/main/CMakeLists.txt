idf_component_register(
    SRCS "soundalert_test.cpp" "mel_features.c"
    INCLUDE_DIRS "."
    REQUIRES
        espressif__esp-tflite-micro
        esp_partition
        esp_psram
        esp_timer
        esp_driver_i2s
        esp_driver_gpio
        espressif__esp-dsp
)
