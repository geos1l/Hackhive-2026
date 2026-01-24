"""
Simple microphone test - no Whisper needed.
Run: python3 test_mic_only.py
"""
import sys
from pathlib import Path

print("=" * 60)
print("MICROPHONE TEST")
print("=" * 60)

# Check dependencies
try:
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
    print("Dependencies OK")
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# List audio devices
print("\nAudio devices:")
try:
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " <-- DEFAULT INPUT" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']} (inputs: {d['max_input_channels']}){marker}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Record
print("\n" + "-" * 60)
print("Recording 3 seconds... SPEAK NOW!")
print("-" * 60)

try:
    SAMPLE_RATE = 16000
    DURATION = 3

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()

    audio = audio.flatten()

    # Stats
    max_amp = np.max(np.abs(audio))
    mean_amp = np.mean(np.abs(audio))

    print(f"\nRecorded {len(audio)} samples ({DURATION}s)")
    print(f"Max amplitude:  {max_amp}")
    print(f"Mean amplitude: {mean_amp:.1f}")

    if max_amp < 100:
        print("\nWARNING: Audio is very quiet!")
        print("  - Check if microphone is connected")
        print("  - Check system audio permissions")
        print("  - Try speaking louder")
    elif max_amp < 1000:
        print("\nAudio detected but quiet - try speaking louder")
    else:
        print("\nMicrophone is working!")

    # Save
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    wav_path = output_dir / "mic_test.wav"
    wav_write(str(wav_path), SAMPLE_RATE, audio)
    print(f"\nSaved recording to: {wav_path}")

    # Also save a text report
    report_path = output_dir / "mic_test_report.txt"
    with open(report_path, "w") as f:
        f.write("MICROPHONE TEST REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Duration: {DURATION} seconds\n")
        f.write(f"Samples recorded: {len(audio)}\n")
        f.write(f"Max amplitude: {max_amp}\n")
        f.write(f"Mean amplitude: {mean_amp:.1f}\n\n")
        if max_amp < 100:
            f.write("STATUS: FAILED - No audio detected\n")
        elif max_amp < 1000:
            f.write("STATUS: WEAK - Audio very quiet\n")
        else:
            f.write("STATUS: OK - Microphone working\n")
    print(f"Saved report to: {report_path}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
