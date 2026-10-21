
#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n=== SoundAlert Memory Check ===");
  Serial.printf("PSRAM: %u bytes\n", ESP.getPsramSize());
  Serial.printf("Free PSRAM: %u bytes\n", ESP.getFreePsram());
  Serial.printf("Free heap: %u bytes\n", ESP.getFreeHeap());
}

void loop() {
  delay(1000);
}
