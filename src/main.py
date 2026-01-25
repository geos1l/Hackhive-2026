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
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import keyboard

from config.settings import Settings
from src.audio import AudioInput
from src.audio.recorder import MicrophoneRecorder
from src.audio.transcriber import WhisperTranscriber
from src.output import OutputHandler, OutputMode
from src.services.router_ai import RouterAI


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


def push_to_talk_loop():
    """
    Main push-to-talk voice assistant loop.
    
    Flow:
    1. Wait for spacebar press
    2. Record while spacebar is held (up to 30s max)
    3. Transcribe with real-time segment display
    4. Show final transcription
    5. Convert to speech and play immediately
    6. Loop back to step 1
    """
    # Validate settings
    missing = Settings.validate()
    if missing:
        print(f"ERROR: Missing required API keys: {missing}")
        print("Please copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    # Determine whisper model
    whisper_model = Settings.WHISPER_MODEL

    # Initialize components
    print("Initializing audio components...")
    recorder = MicrophoneRecorder()
    transcriber = WhisperTranscriber(model_size=whisper_model)
    
    # Initialize Router AI with Supabase integration
    if not Settings.validate_supabase():
        print("Warning: Supabase credentials not set. Router AI will use Gemini directly.")
    router_ai = RouterAI(
        router_api_key=Settings.GEMINI_API_KEY,
        supabase_url=Settings.SUPABASE_URL,
        supabase_key=Settings.SUPABASE_KEY
    )
    output_handler = OutputHandler(
        Settings.ELEVENLABS_API_KEY, mode=OutputMode.BOTH
    )

    print("\n" + "=" * 60)
    print("PUSH-TO-TALK VOICE ASSISTANT")
    print("=" * 60)
    print("Hold SPACEBAR to record (max 30s)")
    print("Press 'S' during audio playback to stop")
    print("Press ESC to quit")
    print("=" * 60 + "\n")

    try:
        while True:
            # Check for quit (ESC key)
            if keyboard.is_pressed('esc'):
                print("\nExiting...")
                break
            
            # Wait for spacebar press
            if keyboard.is_pressed('space'):
                # Record while held
                audio = recorder.record_while_held(max_duration=30.0)
                
                if len(audio) == 0:
                    print("No audio recorded. Try again.")
                    time.sleep(0.5)  # Brief pause to avoid rapid re-triggering
                    continue
                
                # Transcribe with real-time segment display
                text = transcriber.transcribe_array_streaming(audio)
                
                if not text:
                    print("No transcription available. Try again.")
                    time.sleep(0.5)
                    continue
                
                print(f"\nFinal transcription: {text}\n")
                
                # Router AI processing
                print("Processing with Gemini...")
                response = router_ai.process(text)
                print(f"Gemini response: {response}\n")
                
                # TTS and play immediately
                print("Generating speech...")
                print("Press 'S' to stop audio playback (program will continue)")
                output_handler.output(response, mode=OutputMode.BOTH)
                
                # Monitor for stop key while audio is playing
                while output_handler.speaker.is_playing():
                    if keyboard.is_pressed('s'):
                        print("\n✓ Audio playback stopped.")
                        output_handler.speaker.stop()
                        break
                    time.sleep(0.1)  # Check every 100ms
                
                print("\nReady for next recording. Hold SPACEBAR to record again.\n")
                
                # Brief pause to avoid rapid re-triggering
                time.sleep(0.5)
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) == 1:
        push_to_talk_loop()
    else:
        main()
