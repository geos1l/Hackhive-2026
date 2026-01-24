#!/usr/bin/env python3
"""Simple ElevenLabs TTS test - no Whisper dependency."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from src.audio.speaker import TTSSpeaker


def main():
    print("=" * 50)
    print("ElevenLabs TTS Test")
    print("=" * 50)

    # Check API key
    print("\n[1/3] Checking API key...")
    if not Settings.ELEVENLABS_API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not set in .env")
        return False
    print(f"  API key found: {Settings.ELEVENLABS_API_KEY[:8]}...")

    # Initialize speaker
    print("\n[2/3] Initializing TTS speaker...")
    try:
        speaker = TTSSpeaker(
            api_key=Settings.ELEVENLABS_API_KEY,
            voice="Rex Thunder",
            model="fast"
        )
        print("  TTSSpeaker initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize speaker: {e}")
        return False

    # Generate audio
    print("\n[3/3] Generating speech...")
    test_text = "Hello! This is a test of the ElevenLabs text to speech system. If you can hear this, the test was successful."

    output_dir = Settings.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tts_test.wav"

    try:
        speaker.generate_wav(test_text, output_path)
        print(f"  Audio saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate audio: {e}")
        return False

    # Verify file exists
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  File size: {size_kb:.1f} KB")

    print("\n" + "=" * 50)
    print("SUCCESS! TTS test completed.")
    print(f"Listen to the audio: {output_path}")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
