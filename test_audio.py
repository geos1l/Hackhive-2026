"""
Diagnostic script to test audio components step by step.
Run: python3 test_audio.py
"""
import sys
from pathlib import Path

print("=" * 60)
print("AUDIO DIAGNOSTIC TEST")
print("=" * 60)

# Step 1: Check dependencies
print("\n[1/5] Checking dependencies...")
try:
    import numpy as np
    print(f"  - numpy: OK (version {np.__version__})")
except ImportError as e:
    print(f"  - numpy: MISSING - run: pip3 install numpy")
    sys.exit(1)

try:
    import scipy
    print(f"  - scipy: OK (version {scipy.__version__})")
except ImportError as e:
    print(f"  - scipy: MISSING - run: pip3 install scipy")
    sys.exit(1)

try:
    import sounddevice as sd
    print(f"  - sounddevice: OK (version {sd.__version__})")
except ImportError as e:
    print(f"  - sounddevice: MISSING - run: pip3 install sounddevice")
    sys.exit(1)

try:
    import faster_whisper
    print(f"  - faster_whisper: OK")
except ImportError as e:
    print(f"  - faster_whisper: MISSING - run: pip3 install faster-whisper")
    sys.exit(1)

# Step 2: List audio devices
print("\n[2/5] Listing audio devices...")
try:
    devices = sd.query_devices()
    print(f"  Found {len(devices)} audio device(s):")

    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append((i, d))
            marker = " <-- DEFAULT" if i == sd.default.device[0] else ""
            print(f"    [{i}] {d['name']} (inputs: {d['max_input_channels']}){marker}")

    if not input_devices:
        print("  ERROR: No input devices (microphones) found!")
        sys.exit(1)

    default_input = sd.default.device[0]
    print(f"\n  Default input device: [{default_input}] {devices[default_input]['name']}")
except Exception as e:
    print(f"  ERROR listing devices: {e}")
    sys.exit(1)

# Step 3: Test microphone recording
print("\n[3/5] Testing microphone recording (2 seconds)...")
try:
    SAMPLE_RATE = 16000
    DURATION = 2

    print(f"  Recording {DURATION}s at {SAMPLE_RATE}Hz...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()  # Wait for recording to finish

    audio = audio.flatten()

    # Check if we got audio
    max_amplitude = np.max(np.abs(audio))
    mean_amplitude = np.mean(np.abs(audio))

    print(f"  Recorded {len(audio)} samples")
    print(f"  Max amplitude: {max_amplitude}")
    print(f"  Mean amplitude: {mean_amplitude:.1f}")

    if max_amplitude < 100:
        print("  WARNING: Audio is very quiet - microphone may not be working")
    else:
        print("  Audio levels look OK")

    # Save test recording
    from scipy.io.wavfile import write as wav_write
    test_wav = Path("output/test_recording.wav")
    wav_write(str(test_wav), SAMPLE_RATE, audio)
    print(f"  Saved test recording to: {test_wav}")

except Exception as e:
    print(f"  ERROR recording: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test Whisper transcription
print("\n[4/5] Testing Whisper transcription...")
try:
    from faster_whisper import WhisperModel

    print("  Loading Whisper model 'tiny.en' (smallest, for testing)...")
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("  Model loaded!")

    print(f"  Transcribing {test_wav}...")
    segments, info = model.transcribe(str(test_wav), language="en")

    text = " ".join(segment.text.strip() for segment in segments)

    print(f"  Detected language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"  Transcription: '{text}'")

    # Save transcription
    result_file = Path("output/test_transcription.txt")
    with open(result_file, "w") as f:
        f.write(f"Audio file: {test_wav}\n")
        f.write(f"Duration: {DURATION}s\n")
        f.write(f"Max amplitude: {max_amplitude}\n")
        f.write(f"Mean amplitude: {mean_amplitude:.1f}\n")
        f.write(f"Detected language: {info.language}\n")
        f.write(f"Language probability: {info.language_probability:.2f}\n")
        f.write(f"\nTranscription:\n{text}\n")
    print(f"  Results saved to: {result_file}")

except Exception as e:
    print(f"  ERROR with Whisper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Summary
print("\n[5/5] Summary")
print("=" * 60)
print("All tests passed!")
print(f"  - Test recording: output/test_recording.wav")
print(f"  - Transcription:  output/test_transcription.txt")
print("=" * 60)

# ITS WORKINGGGGGG!!! 