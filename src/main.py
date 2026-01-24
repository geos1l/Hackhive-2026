"""
Main entry point for Phase 1 + Phase 6 testing.

Audio I/O Pipeline Test:
1. Record from mic OR load .wav file
2. Transcribe with local Whisper -> get text
3. Echo the text back (placeholder for AI)
4. Convert to speech with ElevenLabs
5. Play through speakers OR save to file
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.audio import AudioInput
from src.output import OutputHandler, OutputMode


def main():
    parser = argparse.ArgumentParser(
        description="AI Voice Assistant - Audio I/O Pipeline Test"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["speak", "stream", "terminal", "save", "both"],
        default="both",
        help="Output mode (default: both)",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Process audio file instead of recording from mic",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Process text directly (skip recording/transcription)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Record for fixed duration (seconds) instead of push-to-talk",
    )
    parser.add_argument(
        "--whisper-model",
        "-w",
        default=None,
        help="Whisper model size (tiny.en, base.en, small.en, medium.en, large-v3)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and exit",
    )
    args = parser.parse_args()

    # Check configuration
    if args.check:
        Settings.print_status()
        missing = Settings.validate()
        if missing:
            print(f"\nMissing required keys: {missing}")
            print("Please set these in your .env file")
            sys.exit(1)
        print("\nConfiguration OK!")
        sys.exit(0)

    # Validate settings
    missing = Settings.validate()
    if missing:
        print(f"ERROR: Missing required API keys: {missing}")
        print("Please copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    # Map mode string to enum
    mode_map = {
        "speak": OutputMode.SPEAK,
        "stream": OutputMode.STREAM,
        "terminal": OutputMode.TERMINAL,
        "save": OutputMode.SAVE,
        "both": OutputMode.BOTH,
    }

    # Determine whisper model
    whisper_model = args.whisper_model or Settings.WHISPER_MODEL

    # Initialize components
    print("Initializing audio components...")
    audio_input = AudioInput(whisper_model=whisper_model)
    output_handler = OutputHandler(
        Settings.ELEVENLABS_API_KEY, mode=mode_map[args.mode]
    )

    # Get input text
    if args.text:
        # Direct text input
        text = args.text
        print(f"Input text: {text}")
    elif args.file:
        # Load and transcribe audio file
        print(f"Loading audio from: {args.file}")
        text = audio_input.load_and_transcribe(args.file)
        print(f"Transcribed: {text}")
    elif args.duration:
        # Record for fixed duration
        text = audio_input.record_duration_and_transcribe(args.duration)
        print(f"Transcribed: {text}")
    else:
        # Push-to-talk recording
        print("\n" + "=" * 50)
        print("AUDIO I/O PIPELINE TEST")
        print("=" * 50)
        print("Press Enter to start recording...")
        input()
        text = audio_input.record_and_transcribe()
        print(f"Transcribed: {text}")

    # Process and output
    # For now, just echo the text back (placeholder for AI response)
    response = f"You said: {text}"

    print("\nGenerating speech output...")
    output_handler.output(response)

    print("\nDone!")


def interactive_loop():
    """Run interactive voice assistant loop."""
    # Validate settings
    missing = Settings.validate()
    if missing:
        print(f"ERROR: Missing required API keys: {missing}")
        print("Please copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    # Initialize components
    print("Initializing audio components...")
    audio_input = AudioInput(whisper_model=Settings.WHISPER_MODEL)
    output_handler = OutputHandler(
        Settings.ELEVENLABS_API_KEY, mode=OutputMode.BOTH
    )

    print("\n" + "=" * 60)
    print("AI VOICE ASSISTANT - Phase 1 + 6 Test")
    print("=" * 60)
    print("Commands:")
    print("  [Enter]  - Record from microphone")
    print("  file:X   - Load audio from file X")
    print("  text:X   - Process text X directly")
    print("  mode:X   - Change output mode (speak/stream/terminal/save/both)")
    print("  quit     - Exit")
    print("=" * 60)

    mode_map = {
        "speak": OutputMode.SPEAK,
        "stream": OutputMode.STREAM,
        "terminal": OutputMode.TERMINAL,
        "save": OutputMode.SAVE,
        "both": OutputMode.BOTH,
    }

    while True:
        try:
            cmd = input("\n> ").strip()

            if cmd.lower() == "quit":
                print("Goodbye!")
                break
            elif cmd.startswith("file:"):
                path = cmd[5:].strip()
                text = audio_input.load_and_transcribe(path)
                print(f"Transcribed: {text}")
            elif cmd.startswith("text:"):
                text = cmd[5:].strip()
            elif cmd.startswith("mode:"):
                mode_name = cmd[5:].strip().lower()
                if mode_name in mode_map:
                    output_handler.set_mode(mode_map[mode_name])
                    print(f"Output mode changed to: {mode_name}")
                else:
                    print(f"Unknown mode: {mode_name}")
                continue
            elif cmd == "":
                # Push-to-talk recording
                text = audio_input.record_and_transcribe()
                print(f"Transcribed: {text}")
            else:
                # Treat as text input
                text = cmd

            # Generate response (echo for now)
            response = f"You said: {text}"
            output_handler.output(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) == 1:
        interactive_loop()
    else:
        main()
