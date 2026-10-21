
#include <ESP_I2S.h>

I2SClass I2S;

constexpr int I2S_SCK = 4;
constexpr int I2S_WS  = 5;
constexpr int I2S_SD  = 6;

constexpr size_t FRAMES = 128;
int32_t samples[FRAMES * 2];

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\nSoundAlert - INMP441 microphone test");

  // BCLK, WS, DOUT (unused), DIN (microphone data)
  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);

  if (!I2S.begin(
        I2S_MODE_STD,
        16000,
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_STEREO)) {
    Serial.println("ERROR: I2S initialization failed.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("I2S initialized. Listening...");
}

void loop() {
  size_t bytesRead = I2S.readBytes(
      reinterpret_cast<char *>(samples),
      sizeof(samples));

  if (bytesRead == 0) {
    Serial.println("No audio data received.");
    delay(200);
    return;
  }

  size_t count = bytesRead / sizeof(int32_t);
  uint64_t total = 0;
  int32_t peak = 0;
  size_t used = 0;

  // L/R tied to GND selects the left channel.
  for (size_t i = 0; i < count; i += 2) {
    int32_t value = samples[i] >> 8;
    int64_t magnitude = value;

    if (magnitude < 0) {
      magnitude = -magnitude;
    }

    if (magnitude > peak) {
      peak = static_cast<int32_t>(magnitude);
    }

    total += static_cast<uint64_t>(magnitude);
    used++;
  }

  if (used > 0) {
    Serial.printf(
        "Samples: %u | Mean amplitude: %llu | Peak: %ld\n",
        static_cast<unsigned>(used),
        static_cast<unsigned long long>(total / used),
        static_cast<long>(peak));
  }
}

