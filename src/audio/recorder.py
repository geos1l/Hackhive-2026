"""Microphone recording module for ESP32-compatible audio capture."""
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import tempfile
import time
import keyboard


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

    def record_while_held(self, max_duration: float = 30.0) -> np.ndarray:
        """
        Record audio while a key (spacebar) is held down.
        
        Args:
            max_duration: Maximum recording duration in seconds (default: 30.0)
            
        Returns:
            Audio data as numpy array
        """
        print("Hold SPACEBAR to record (max 30s)... Release to stop.")
        
        # Wait for spacebar press
        keyboard.wait('space')
        
        # Start recording
        self.start_recording()
        start_time = time.time()
        
        # Monitor while spacebar is held
        try:
            while keyboard.is_pressed('space'):
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    print(f"\nMax duration ({max_duration}s) reached!")
                    break
                # Update display with elapsed time
                print(f"\rRecording... {elapsed:.1f}s / {max_duration:.0f}s", end='', flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        
        # Stop recording
        print()  # New line after progress display
        audio = self.stop_recording()
        
        duration = len(audio) / self.SAMPLE_RATE
        print(f"Recorded {len(audio)} samples ({duration:.2f}s)")
        
        return audio
