"""Centralized configuration from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # ElevenLabs - For TTS
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

    # Whisper model (local, no API key needed)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "output"

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of missing required keys."""
        required = [
            "ELEVENLABS_API_KEY",
        ]
        return [k for k in required if not getattr(cls, k)]

    @classmethod
    def print_status(cls):
        """Print configuration status."""
        print("Configuration Status:")
        print(f"  [+] WHISPER_MODEL: {cls.WHISPER_MODEL} (local, no API key needed)")

        value = cls.ELEVENLABS_API_KEY
        status = "+" if value else "x"
        masked = value[:8] + "..." if value else "NOT SET"
        print(f"  [{status}] ELEVENLABS_API_KEY: {masked}")
