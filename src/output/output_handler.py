"""Dual output mode handler (TTS or terminal)."""
from pathlib import Path
from typing import Optional
from enum import Enum
import time

from src.audio.speaker import TTSSpeaker


class OutputMode(Enum):
    """Output modes for the voice assistant."""

    SPEAK = "speak"  # Play through speakers
    STREAM = "stream"  # Stream for low latency
    TERMINAL = "terminal"  # Print to terminal only
    SAVE = "save"  # Save to file only
    BOTH = "both"  # Speak and print


class OutputHandler:
    """Handles output in multiple modes (simulating ESP32 speaker/display)."""

    def __init__(
        self,
        elevenlabs_api_key: str,
        voice: str = "rachel",
        mode: OutputMode = OutputMode.BOTH,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize output handler.

        Args:
            elevenlabs_api_key: ElevenLabs API key
            voice: Voice name for TTS
            mode: Default output mode
            output_dir: Directory for saved audio files
        """
        self.speaker = TTSSpeaker(elevenlabs_api_key, voice)
        self.mode = mode
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def output(
        self,
        text: str,
        mode: Optional[OutputMode] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Output text according to mode.

        Args:
            text: Response text from LLM
            mode: Override default mode
            save_path: Path to save audio (for SAVE mode)

        Returns:
            Path to saved audio file (if saved)
        """
        current_mode = mode or self.mode
        saved_path = None

        # Terminal output
        if current_mode in [OutputMode.TERMINAL, OutputMode.BOTH]:
            self._print_response(text)

        # Audio output
        if current_mode == OutputMode.SPEAK:
            self.speaker.speak(text)
        elif current_mode == OutputMode.STREAM:
            self.speaker.speak_stream(text)
        elif current_mode == OutputMode.BOTH:
            self.speaker.speak_stream(text)
        elif current_mode == OutputMode.SAVE:
            saved_path = save_path or self._generate_path()
            self.speaker.generate_wav(text, saved_path)
            print(f"Audio saved to: {saved_path}")

        return saved_path

    def _print_response(self, text: str) -> None:
        """Pretty-print response to terminal."""
        print("\n" + "=" * 60)
        print("ASSISTANT RESPONSE:")
        print("-" * 60)
        print(text)
        print("=" * 60 + "\n")

    def _generate_path(self) -> Path:
        """Generate unique output path."""
        timestamp = int(time.time())
        return self.output_dir / f"response_{timestamp}.wav"

    def get_esp32_audio(self, text: str) -> tuple[bytes, int]:
        """
        Get audio bytes ready for ESP32 transmission.

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        return self.speaker.get_audio_bytes(text)

    def set_mode(self, mode: OutputMode) -> None:
        """
        Change output mode.

        Args:
            mode: New output mode
        """
        self.mode = mode

    def set_voice(self, voice: str) -> bool:
        """
        Change TTS voice.

        Args:
            voice: Voice name or ID

        Returns:
            True if voice was changed
        """
        return self.speaker.set_voice(voice)
