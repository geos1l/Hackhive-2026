"""Microphone recording module for ESP32-compatible audio capture."""
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import tempfile


class MicrophoneRecorder:
    """Records audio from laptop microphone (simulating ESP32 mic)."""

    SAMPLE_RATE = 16000  # 16 kHz for ESP32 compatibility
    CHANNELS = 1  # Mono
    DTYPE = np.int16  # PCM16

    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.stream = None

    def start_recording(self) -> None:
        """Start recording (push-to-talk mode)."""
        self.is_recording = True
        self.audio_chunks = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            if self.is_recording:
                self.audio_chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            callback=callback,
        )
        self.stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_chunks:
            return np.concatenate(self.audio_chunks, axis=0).flatten()
        return np.array([], dtype=self.DTYPE)

    def record_for_duration(self, seconds: float) -> np.ndarray:
        """Record for fixed duration (alternative mode)."""
        frames = int(seconds * self.SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
        )
        sd.wait()
        return audio.flatten()

    def save_wav(self, audio: np.ndarray, path: Path) -> Path:
        """Save audio to WAV file."""
        write(str(path), self.SAMPLE_RATE, audio)
        return path

    def save_to_temp(self, audio: np.ndarray) -> Path:
        """Save audio to temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, self.SAMPLE_RATE, audio)
        return Path(temp_file.name)
