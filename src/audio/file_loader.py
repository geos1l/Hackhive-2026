"""WAV file loader with ESP32-compatible format conversion."""
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample
import numpy as np
from pathlib import Path
from typing import Union


class WavFileLoader:
    """Loads and normalizes WAV files to ESP32-compatible format."""

    TARGET_RATE = 16000  # 16 kHz

    def load(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """
        Load WAV file, resample to 16kHz if needed.

        Args:
            path: Path to WAV file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        sample_rate, audio = wav_read(str(path))

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        # Resample if not 16kHz
        if sample_rate != self.TARGET_RATE:
            num_samples = int(len(audio) * self.TARGET_RATE / sample_rate)
            audio = resample(audio, num_samples).astype(np.int16)
            sample_rate = self.TARGET_RATE

        return audio, sample_rate

    def validate_format(self, path: Union[str, Path]) -> dict:
        """
        Check if file meets ESP32 requirements.

        Args:
            path: Path to WAV file

        Returns:
            Dict with format information and validity
        """
        sample_rate, audio = wav_read(str(path))
        is_mono = len(audio.shape) == 1
        channels = 1 if is_mono else audio.shape[1]

        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": len(audio) / sample_rate,
            "dtype": str(audio.dtype),
            "is_valid": sample_rate == 16000 and is_mono,
        }
