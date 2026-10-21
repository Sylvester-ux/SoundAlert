# SoundAlert — False Vibration Diagnosis & Fix
Date: 2026-10-21 · Repo: github.com/Sylvester-ux/SoundAlert (commit dd2a2b8) · Model: huggingface.co/chayuto/yamnet-mel-int8-tflm (INT8 YAMNet, 521 AudioSet classes, in repo at `models/yamnet/yamnet.tflite`)

## Symptom
Hardware vibrates for unknown sounds instead of only baby cry, fire alarms, and sirens.

## First check — which firmware is on the board?
The repo contains two different firmwares:
- **Arduino `MicTest/BoardTest/BoardTest.ino` ("firmware v1")** — loudness only. `classify()` literally returns `SOUND_LOUD` for everything; any sound 20 dB above the noise floor and louder than −50 dBFS for 32 ms vibrates. If this is flashed, false vibration is by design — it has no YAMNet at all.
- **ESP-IDF `esp_idf_test/soundalert_test/`** — the YAMNet firmware analysed below. Serial banner: `SoundAlert live-audio YAMNet starting`.

## Root causes (ESP-IDF firmware, all verified in code)

1. **No "unknown" rejection — the core bug.** `process_alert_scores()` only compares three hand-picked groups *against each other*: baby (20) vs an emergency max over 7 classes (317, 318, 319, 390, 391, 393, 394) vs telephone (384). It never looks at the other ~512 classes. YAMNet is a scene classifier — its own model card warns: *"expect non-zero probabilities for many classes in noisy audio."* So when speech/music/TV is really Speech at 0.90, a spurious 0.61 on some siren class still vibrates, because nothing checks that the target is the model's actual best explanation.
2. **Emergency = max over 7 classes** gives every patch 7 chances to cross the threshold instead of 1.
3. **Thresholds sit below this model's noise floor.** Host test with the repo's own model file: a constant "silence" log-mel patch (−6.9) gives emergency max **0.672** (threshold 0.60); pure random Gaussian patches give 0.625–0.75. Dozens of unrelated classes pin at ~0.65–0.75 on nonsense input.
4. **No loudness gate in the ESP-IDF firmware.** Capture computes RMS every second and only logs it. Quiet room noise is classified and can alert.
5. **Baby-cry special case skips the margin check** and lets a competitor beat baby by 0.12 and still alert as BABY_CRY (`kBabyCryThreshold 0.50` + `kBabyCompetitorTolerance`).
6. **Telephone bell (384) is an active trigger**, although it is not a wanted event. Doorbell/car-alarm classes are defined in a debug list too (dead code, but a trap).
7. **Only 2 consecutive patches (~1.9 s) required**, and the release/re-arm constants (`kReleaseThreshold`, `kRequiredReleasePatches`) are defined but never used — one long sound re-alerts every 5 s cooldown.

## Root cause in the model/mel chain (host-verified)

The model's INT8 PTQ was calibrated on **200 synthetic Gaussian patches (N(−5, 3))** — stated on its own model card — not real audio. The firmware's mel extractor feeds **unnormalised FFT magnitude** (loud-tone peaks ≈ 100, log-mel peaks ≈ +4.5), which pushes internal activations into saturation: with that exact mel, a synthetic siren's top-3 classes are *Fireworks / Bass guitar / Biting*.

A/B with the same model and same sounds, only dividing magnitude by FFT size (512):
- Siren → top-3 becomes **Siren 0.71 / Fire-engine siren 0.70 / Car alarm 0.70** ✓
- Music → *Harmonic / Background music / Cello*, siren drops ✓
- Tonal log-mel statistics stay in the calibrated range; broadband noise (which fills the normally-empty high mel bands) is the worst out-of-distribution case, which is why everyday sounds false-fire hardest.

## Fix (attached files — drop-in replacements)

- `soundalert_test_FIXED.cpp` → replaces `main/soundalert_test.cpp`
  - Only 3 wanted events: **BabyCry (20), Siren (317/318/319/390/391), FireAlarm (393/394)**. Telephone removed.
  - **Unknown check:** target must beat *every* non-target class (family relatives like Alarm 382 / Crying 19 excluded from competitors, since real events light them up too), else `REJECT reason=UNKNOWN`.
  - **Loudness + clipping gate:** per-patch RMS now computed in `capture_task` and shared (`s_last_patch_dbfs`, floor −45 dBFS) — quiet patches can never alert; clipped/distorted patches rejected.
  - Thresholds: baby 0.55, siren 0.65, fire 0.60; **3 consecutive patches** of the same event; 8 s cooldown; **re-arm requires 2 quiet patches** (the previously dead release logic, implemented).
  - Every rejection logs its reason + the top unknown class, so field tuning is data-driven.
- `mel_features_FIXED.c` → replaces `main/mel_features.c`
  - Magnitude divided by FFT size (the A/B-verified scaling above), with the evidence in the comment.
- `run_model_tests.py` → the host test harness used here (mel mirrors the firmware; needs `ai-edge-litert` + the repo model).

## Host test results (synthetic sounds, per-patch pass = would count toward alert)

| Sound | Old logic + old mel | Fixed logic + fixed mel |
|---|---|---|
| Siren sweep (positive) | 5/5 alert | 4/5 pass |
| Smoke alarm beeps (positive) | 5/5 alert | 4/5 pass (lands in Siren family — still vibrates; pattern label may say SIREN) |
| White noise, loud | 4/5 alert | **0/5** |
| White noise, quiet | 5/5 alert | **0/5** |
| Music melody | 5/5 alert | **0/5** |
| Claps/transients | 2/5 alert | **0/5** |
| Speech-like babble | 5/5 alert | **0/5** |
| Silence | alert on raw-model test (emergency 0.672) | **0/5** (UNKNOWN + on-device RMS gate) |
| Baby-cry-like (synthetic, rough approx.) | 4/5 alert | mostly rejected — see caveat |

Negatives: **~70% false-alert patches → 0/30.**

## Caveats — read before trusting it
- **Not compiled or run on hardware here** (no ESP-IDF toolchain/device in this environment). The fixed files are drop-ins with balanced/sanity-checked code, but the build + on-device test is still to do.
- Positives were **synthetic** (siren sweep, 3.1 kHz beeps). The baby-cry synth is a rough harmonic wail, not a real cry — **baby cry is the one event not proven**. Field-test with a real cry recording before judging the 0.55 threshold.
- This model will never be great: synthetic-PTQ INT8 YAMNet scores cluster at 0.6–0.75 for everything. The fixes make the *decision logic* honest, but the long-term fix is recalibrating the model with real audio (`convert_yamnet.py --audio-dir ... --calib-count 500` per the model card) or training a small dedicated 3-class model (baby / alarm / siren vs other).

## Field test protocol (10 minutes)
1. Flash the fixed firmware. Play, at realistic distance/volume: baby cry recording, smoke-alarm test beep / alarm video, a real siren video — each ≥10 s. Expect exactly one alert burst each, correct pattern in the log.
2. Negatives, 1 min each: TV/conversation, music, kettle/microwave beeps, clapping, traffic through a window, silence. Expect **no vibration**; logs should show `REJECT reason=UNKNOWN/LOW_SCORE/TOO_QUIET` — send those log lines back if anything vibrates, they name the exact class and scores to tune.
