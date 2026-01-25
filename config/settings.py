"""Centralized configuration from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # ElevenLabs - For TTS
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    # OpenRouter - For Router AI (Gemini via OpenRouter)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Actually OpenRouter API key
    # Whisper model (local, no API key needed)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://klszeubsjtmbbyjcnpcd.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtsc3pldWJzanRtYmJ5amNucGNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkyMDA4MTQsImV4cCI6MjA4NDc3NjgxNH0.8OoUmXC3yzzUDXOxqk13B9Ppx4TONYHtKaJeVK0i1Ow")

    # LLM API Keys (for routing - read dynamically from .env)
    # These will be accessed via os.getenv() when needed
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    # Add more as needed: COHERE_API_KEY, etc.

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
    @classmethod
    def get_api_key(cls, env_var_name: str) -> str:
        """
        Get API key from environment variable.
        
        Args:
            env_var_name: Name of environment variable
            
        Returns:
            API key value or empty string
        """
        return os.getenv(env_var_name, "")

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of missing required keys."""
        required = [
            "ELEVENLABS_API_KEY",
            "GEMINI_API_KEY",
        ]
        return [k for k in required if not getattr(cls, k)]
    
    @classmethod
    def validate_supabase(cls) -> bool:
        """Check if Supabase credentials are set."""
        return bool(cls.SUPABASE_URL and cls.SUPABASE_KEY)

    @classmethod
    def print_status(cls):
        """Print configuration status."""
        print("Configuration Status:")
        print(f"  [+] WHISPER_MODEL: {cls.WHISPER_MODEL} (local, no API key needed)")

        value = cls.ELEVENLABS_API_KEY
        status = "+" if value else "x"
        masked = value[:8] + "..." if value else "NOT SET"
        print(f"  [{status}] ELEVENLABS_API_KEY: {masked}")

        value = Settings.GEMINI_API_KEY
        status = "+" if value else "x"
        masked = value[:8] + "..." if value else "NOT SET"
        print(f"  [{status}] GEMINI_API_KEY: {masked}")
