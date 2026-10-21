// SoundAlert - INMP441 microphone test (v2)
// Wiring: SCK -> GPIO4, WS -> GPIO5, SD -> GPIO6, VDD -> 3V3, GND -> GND, L/R -> GND
// Board package: Arduino-ESP32 core 3.x (uses <ESP_I2S.h>)
// Serial Monitor: 115200 baud

#include <ESP_I2S.h>
#include <math.h>

I2SClass I2S;

constexpr int I2S_SCK = 4;
constexpr int I2S_WS  = 5;
constexpr int I2S_SD  = 6;

constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t FRAMES = 256;          // 16 ms per block at 16 kHz
constexpr int BLOCKS_PER_REPORT = 6;    // about 100 ms per printed line
constexpr bool USE_RIGHT_CHANNEL = false;  // set true if L/R pin is tied to 3V3

int32_t buf[FRAMES * 2];

float dc = 0;
bool dcReady = false;
double sumSq = 0;
float peak = 0;
double rawSum = 0;
int32_t rawMin = INT32_MAX, rawMax = INT32_MIN;
uint32_t total = 0, zeros = 0;
int blocks = 0;

void resetStats() {
  sumSq = 0; peak = 0; rawSum = 0;
  rawMin = INT32_MAX; rawMax = INT32_MIN;
  total = 0; zeros = 0; blocks = 0;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nSoundAlert - INMP441 microphone test (v2)");

  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);

  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("ERROR: I2S initialization failed.");
    while (true) delay(1000);
  }

  Serial.println("Stay quiet for 3 seconds to see the noise floor, then talk and clap.");
  Serial.println("RMS = loudness (DC offset removed). dBFS: 0 is the loudest possible.");
  resetStats();
}

void loop() {
  size_t bytesRead = I2S.readBytes(reinterpret_cast<char *>(buf), sizeof(buf));
  if (bytesRead == 0) {
    Serial.println("No audio data received.");
    delay(200);
    return;
  }

  size_t frames = bytesRead / (2 * sizeof(int32_t));
  for (size_t f = 0; f < frames; f++) {
    int32_t raw = buf[f * 2 + (USE_RIGHT_CHANNEL ? 1 : 0)] >> 8;  // 24-bit signed value

    if (!dcReady) { dc = raw; dcReady = true; }
    dc += (raw - dc) * 0.001f;          // slow average = DC offset (about 2.5 Hz high-pass)
    float x = raw - dc;

    sumSq += (double)x * x;
    float ax = fabsf(x);
    if (ax > peak) peak = ax;

    rawSum += raw;
    if (raw < rawMin) rawMin = raw;
    if (raw > rawMax) rawMax = raw;
    if (raw == 0) zeros++;
  }
  total += frames;
  blocks++;

  if (blocks < BLOCKS_PER_REPORT) return;

  if (zeros == total) {
    Serial.println("All samples are 0: check SD wire (GPIO6), VDD on 3V3, GND, and that the board is soldered.");
  } else if (rawMin == rawMax) {
    Serial.printf("Data is stuck at %ld: check WS/SCK wires and L/R pin.\n", (long)rawMin);
  } else {
    float rms = sqrt(sumSq / total);
    float dbfs = 20.0f * log10f((rms < 1.0f ? 1.0f : rms) / 8388608.0f);
    long dcOffset = (long)(rawSum / total);

    int barLen = (int)((dbfs + 90.0f) * 40.0f / 70.0f);   // -90 dBFS .. -20 dBFS -> 0..40
    if (barLen < 0) barLen = 0;
    if (barLen > 40) barLen = 40;
    char bar[42];
    for (int i = 0; i < barLen; i++) bar[i] = '#';
    bar[barLen] = '\0';

    Serial.printf("RMS %7.0f | Peak %8.0f | %6.1f dBFS | DC %8ld | %s\n",
                  rms, peak, dbfs, dcOffset, bar);
  }
  resetStats();
}
