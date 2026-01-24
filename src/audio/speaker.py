"""ElevenLabs TTS integration with ESP32-compatible output."""
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream
import numpy as np
from scipy.io.wavfile import write as wav_write
from pathlib import Path
from typing import Optional


class TTSSpeaker:
    """Text-to-speech using ElevenLabs API."""

    # Voice IDs
    VOICES = {
        "Rex Thunder": "mtrellq69YZsNwzUSyXh",  # Deep, powerful
        "rachel": "EXAVITQu4vr4xnSDxMaL",  # Calm, professional (default)
        "adam": "pNInz6obpgDQGcFmaJgB",  # Deep, authoritative
        "bella": "EXAVITQu4vr4xnSDxMaL",  # Warm, friendly
        "josh": "TxGEqnHWrfWFTfGW9XjX",  # Energetic
    }

    # Models
    MODELS = {
        "fast": "eleven_flash_v2_5",  # Low latency (~150ms)
        "quality": "eleven_multilingual_v2",  # High quality
    }

    # ESP32-compatible format
    ESP32_SAMPLE_RATE = 16000
    ESP32_DTYPE = np.int16

    def __init__(
        self, api_key: str, voice: str = "rachel", model: str = "fast"
    ):
        """
        Initialize TTS speaker.

        Args:
            api_key: ElevenLabs API key
            voice: Voice name or ID
            model: Model name ('fast' or 'quality')
        """
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = self.VOICES.get(voice, voice)  # Allow custom voice ID
        self.model_id = self.MODELS.get(model, model)

    def speak(self, text: str) -> None:
        """
        Generate speech and play through speakers.

        Args:
            text: Text to convert to speech
        """
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        play(audio)

    def speak_stream(self, text: str) -> None:
        """
        Stream speech for lower latency playback.

        Args:
            text: Text to convert to speech
        """
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_flash_v2_5",  # Streaming requires flash model
        )
        stream(audio_stream)

    def generate_pcm(self, text: str) -> bytes:
        """
        Generate raw PCM audio at 16kHz (ESP32 format).

        Args:
            text: Text to convert to speech

        Returns:
            Raw PCM audio bytes
        """
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="pcm_16000",  # 16kHz PCM
        )
        return b"".join(audio)

    def generate_wav(self, text: str, output_path: Path) -> Path:
        """
        Generate WAV file in ESP32-compatible format.

        Args:
            text: Text to convert to speech
            output_path: Path to save WAV file

        Returns:
            Path to saved WAV file
        """
        pcm_data = self.generate_pcm(text)

        # Convert bytes to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Save as WAV
        wav_write(str(output_path), self.ESP32_SAMPLE_RATE, audio_array)
        return output_path

    def get_audio_bytes(self, text: str) -> tuple[bytes, int]:
        """
        Get audio as bytes with sample rate (for ESP32 transmission).

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        pcm_data = self.generate_pcm(text)
        return pcm_data, self.ESP32_SAMPLE_RATE

    def set_voice(self, voice: str) -> bool:
        """
        Change TTS voice.

        Args:
            voice: Voice name or ID

        Returns:
            True if voice was changed successfully
        """
        if voice in self.VOICES:
            self.voice_id = self.VOICES[voice]
            return True
        # Assume it's a custom voice ID
        self.voice_id = voice
        return True
