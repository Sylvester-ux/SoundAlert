// SoundAlert firmware v1
// ESP32-S3-DevKitC-1 (N16R8) + INMP441 I2S mic + DC vibration motor module
//
// Module A: I2S capture, DC-offset removal, RMS / dBFS per 16 ms block
// Module B: adaptive loudness trigger (noise floor + margin) with cooldown,
//           motor blanking and re-baselining
// Module C: non-blocking vibration patterns (millis() state machine, PWM strength)
//
// Wiring: INMP441 SCK->GPIO4, WS->GPIO5, SD->GPIO6, VDD->3V3, GND->GND, L/R->GND
//         Motor module signal -> GPIO7 (module power from the 5V rail or battery, NOT 3V3)
// Board package: Arduino-ESP32 core 3.x. Serial Monitor: 115200 baud.
//
// Serial commands:  1 doorbell  2 cry  3 alarm  4 loud   (play a pattern)
//                   v toggle live status   c recalibrate noise floor

#include <ESP_I2S.h>
#include <esp_system.h>
#include <math.h>

// ------------------------------------------------------------------ settings
constexpr int I2S_SCK = 4; //red
constexpr int I2S_WS  = 5; //green
constexpr int I2S_SD  = 6; //blue or black
constexpr int MOTOR_PIN = 7;

constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t   FRAMES = 256;                 // 16 ms per block
constexpr bool     USE_RIGHT_CHANNEL = false;    // true if L/R pin is tied to 3V3

// Motor
constexpr bool     MOTOR_ACTIVE_LOW = false;     // true if your module switches on with LOW
constexpr uint8_t  MOTOR_STRENGTH = 255;         // 0-255
constexpr uint32_t MOTOR_PWM_HZ = 1000;

// Trigger tuning (calibrate these with MicTest readings)
constexpr float    TRIGGER_MARGIN_DB = 20.0f;    // must be this far above the noise floor
constexpr float    ABS_MIN_DBFS = -50.0f;        // and louder than this (ignores normal speech)
constexpr int      CONSEC_BLOCKS = 2;            // blocks in a row above threshold (32 ms)
constexpr uint32_t COOLDOWN_MS = 1500;           // minimum gap between alerts
constexpr uint32_t BLANK_AFTER_MS = 250;         // ignore the mic while/after the motor buzzes
constexpr uint32_t CALIBRATION_MS = 2000;        // startup noise-floor measurement
constexpr uint32_t REBASELINE_MS = 30000;        // sound above threshold this long = new ambient level

// ------------------------------------------------------------------ patterns
// Steps alternate ON, OFF, ON, ... in milliseconds, starting with ON.
// Use even-length arrays (ending with an OFF gap) for patterns with repeats > 1.
enum SoundType : uint8_t { SOUND_DOORBELL, SOUND_CRY, SOUND_ALARM, SOUND_LOUD, SOUND_COUNT };

struct Pattern {
  const char *name;
  const uint16_t *steps;
  uint8_t nSteps;
  uint8_t repeats;
};

const uint16_t P_DOORBELL[] = {150, 100, 150};
const uint16_t P_CRY[]      = {300, 200, 300, 200, 300};
const uint16_t P_ALARM[]    = {1000, 300, 1000, 600};
const uint16_t P_LOUD[]     = {400};

const Pattern PATTERNS[SOUND_COUNT] = {
  {"DOORBELL", P_DOORBELL, sizeof(P_DOORBELL) / sizeof(uint16_t), 1},
  {"CRY",      P_CRY,      sizeof(P_CRY) / sizeof(uint16_t),      1},
  {"ALARM",    P_ALARM,    sizeof(P_ALARM) / sizeof(uint16_t),    2},
  {"LOUD",     P_LOUD,     sizeof(P_LOUD) / sizeof(uint16_t),     1},
};

// ------------------------------------------------------------------ state
I2SClass I2S;
int32_t buf[FRAMES * 2];

// Module A
float dc = 0;
bool dcReady = false;

// Module B
float floorDb = -80.0f;
bool calibrating = false;
uint32_t calibEnd = 0;
double calibSum = 0;
uint32_t calibN = 0;
int aboveCount = 0;
uint32_t loudSince = 0;
uint32_t lastEventAt = 0;
uint32_t eventCount = 0;
bool verbose = false;

// Module C
struct Player {
  const Pattern *pat = nullptr;
  uint8_t step = 0;
  uint8_t rep = 0;
  uint32_t stepStart = 0;
  bool active = false;
} player;
uint32_t motorEndedAt = 0;

// ------------------------------------------------------------------ Module C: motor
void motorWrite(bool on) {
  uint8_t duty = on ? MOTOR_STRENGTH : 0;
  if (MOTOR_ACTIVE_LOW) duty = 255 - duty;
  ledcWrite(MOTOR_PIN, duty);
}

void startPattern(SoundType t) {
  player.pat = &PATTERNS[t];
  player.step = 0;
  player.rep = 0;
  player.stepStart = millis();
  player.active = true;
  motorWrite(true);  // step 0 is always ON
}

bool motorBusy() { return player.active; }

void motorUpdate() {
  if (!player.active) return;
  uint32_t now = millis();
  if (now - player.stepStart < player.pat->steps[player.step]) return;

  player.step++;
  player.stepStart = now;
  if (player.step >= player.pat->nSteps) {
    player.step = 0;
    player.rep++;
    if (player.rep >= player.pat->repeats) {
      player.active = false;
      motorWrite(false);
      motorEndedAt = now;
      return;
    }
  }
  motorWrite((player.step % 2) == 0);  // even steps ON, odd steps OFF
}

// ------------------------------------------------------------------ Module B: events
// v2 hook: spectral rules (dominant frequency, tonal vs broadband, beep rhythm)
// will choose between DOORBELL / CRY / ALARM here. For now every loud sound is LOUD.
SoundType classify() { return SOUND_LOUD; }

void startCalibration() {
  calibrating = true;
  calibEnd = millis() + CALIBRATION_MS;
  calibSum = 0;
  calibN = 0;
  aboveCount = 0;
  loudSince = 0;
  Serial.println("Calibrating noise floor... stay quiet for 2 seconds.");
}

void processBlock(float db, uint32_t now) {
  if (calibrating) {
    calibSum += db;
    calibN++;
    if (now >= calibEnd && calibN > 0) {
      floorDb = calibSum / calibN;
      calibrating = false;
      Serial.printf("Noise floor: %.1f dBFS | trigger above %.1f dBFS\n", floorDb,
                    fmaxf(floorDb + TRIGGER_MARGIN_DB, ABS_MIN_DBFS));
    }
    return;
  }

  // Ignore the mic while the motor runs (and briefly after) so it can't hear itself.
  if (motorBusy() || (now - motorEndedAt) < BLANK_AFTER_MS) {
    aboveCount = 0;
    return;
  }

  float thr = fmaxf(floorDb + TRIGGER_MARGIN_DB, ABS_MIN_DBFS);

  if (db > thr) {
    if (aboveCount < 255) aboveCount++;
    if (loudSince == 0) loudSince = now;

    if (now - loudSince > REBASELINE_MS) {
      floorDb = db;
      loudSince = 0;
      aboveCount = 0;
      Serial.println("[INFO] Sustained loud level, noise floor re-baselined.");
    } else if (aboveCount >= CONSEC_BLOCKS && (now - lastEventAt) > COOLDOWN_MS) {
      SoundType t = classify();
      lastEventAt = now;
      eventCount++;
      Serial.printf("[EVENT] %s | %.1f dBFS | floor %.1f | +%.1f dB\n",
                    PATTERNS[t].name, db, floorDb, db - floorDb);
      startPattern(t);
    }
  } else {
    aboveCount = 0;
    loudSince = 0;
    float alpha = (db < floorDb) ? 0.10f : 0.01f;  // falls fast, rises slowly
    floorDb += (db - floorDb) * alpha;
  }
}

// ------------------------------------------------------------------ helpers
const char *resetReasonName(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:   return "power-on";
    case ESP_RST_SW:        return "software reset";
    case ESP_RST_PANIC:     return "PANIC (crash)";
    case ESP_RST_INT_WDT:   return "interrupt watchdog";
    case ESP_RST_TASK_WDT:  return "task watchdog";
    case ESP_RST_WDT:       return "other watchdog";
    case ESP_RST_BROWNOUT:  return "BROWNOUT (power dipped)";
    default:                return "other";
  }
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    switch (c) {
      case '1': Serial.println("Test: DOORBELL"); startPattern(SOUND_DOORBELL); break;
      case '2': Serial.println("Test: CRY");      startPattern(SOUND_CRY);      break;
      case '3': Serial.println("Test: ALARM");    startPattern(SOUND_ALARM);    break;
      case '4': Serial.println("Test: LOUD");     startPattern(SOUND_LOUD);     break;
      case 'v': verbose = !verbose; Serial.println(verbose ? "Status: on" : "Status: off"); break;
      case 'c': startCalibration(); break;
      default: break;
    }
  }
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  Serial.begin(115200);
  delay(1500);
  setCpuFrequencyMhz(160);  // plenty for this job, lower power than 240 MHz

  Serial.println("\nSoundAlert firmware v1");
  Serial.printf("Last reset: %s\n", resetReasonName(esp_reset_reason()));

  ledcAttach(MOTOR_PIN, MOTOR_PWM_HZ, 8);
  motorWrite(false);

  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("ERROR: I2S initialization failed.");
    while (true) delay(1000);
  }

  Serial.println("Commands: 1 doorbell, 2 cry, 3 alarm, 4 loud, v status, c recalibrate");
  startCalibration();
}

void loop() {
  handleSerial();
  motorUpdate();

  // ---- Module A: capture one block, remove DC offset, compute RMS / dBFS
  size_t bytesRead = I2S.readBytes(reinterpret_cast<char *>(buf), sizeof(buf));
  if (bytesRead == 0) {
    Serial.println("No audio data received.");
    delay(200);
    return;
  }

  size_t frames = bytesRead / (2 * sizeof(int32_t));
  double sumSq = 0;
  uint32_t zeros = 0;
  for (size_t f = 0; f < frames; f++) {
    int32_t raw = buf[f * 2 + (USE_RIGHT_CHANNEL ? 1 : 0)] >> 8;  // 24-bit signed
    if (!dcReady) { dc = raw; dcReady = true; }
    dc += (raw - dc) * 0.001f;
    float x = raw - dc;
    sumSq += (double)x * x;
    if (raw == 0) zeros++;
  }

  uint32_t now = millis();

  if (zeros == frames) {
    static uint32_t lastWarn = 0;
    if (now - lastWarn > 2000) {
      lastWarn = now;
      Serial.println("All mic samples are 0: check SD wire (GPIO6), 3V3, GND and soldering.");
    }
    return;
  }

  float rms = sqrtf((float)(sumSq / frames));
  float db = 20.0f * log10f((rms < 1.0f ? 1.0f : rms) / 8388608.0f);

  // ---- Module B + C
  processBlock(db, now);
  motorUpdate();

  // ---- Logging
  static uint32_t lastStatus = 0, lastHeartbeat = 0;
  if (verbose && now - lastStatus >= 200) {
    lastStatus = now;
    Serial.printf("%6.1f dBFS | floor %6.1f | trigger %6.1f\n", db, floorDb,
                  fmaxf(floorDb + TRIGGER_MARGIN_DB, ABS_MIN_DBFS));
  }
  if (now - lastHeartbeat >= 60000) {  // one line a minute for the 24-hour stability test
    lastHeartbeat = now;
    Serial.printf("[HB] up %lu s | events %lu | floor %.1f dBFS | free heap %lu\n",
                  (unsigned long)(now / 1000), (unsigned long)eventCount, floorDb,
                  (unsigned long)ESP.getFreeHeap());
  }
}
