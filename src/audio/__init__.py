"""Audio input/output module for voice assistant."""
from .recorder import MicrophoneRecorder
from .file_loader import WavFileLoader
from .transcriber import WhisperTranscriber


class AudioInput:
    """Unified interface for audio input (mic or file) with local Whisper."""

    def __init__(self, whisper_model: str = "base.en"):
        """
        Initialize audio input with local Whisper.

        Args:
            whisper_model: Whisper model size. Options:
                - 'tiny.en' / 'tiny'     (~75 MB)  - Fastest
                - 'base.en' / 'base'     (~140 MB) - Good balance (default)
                - 'small.en' / 'small'   (~460 MB) - Better accuracy
                - 'medium.en' / 'medium' (~1.5 GB) - High accuracy
                - 'large-v2' / 'large-v3' (~3 GB)  - Best accuracy
        """
        self.recorder = MicrophoneRecorder()
        self.loader = WavFileLoader()
        self.transcriber = WhisperTranscriber(model_size=whisper_model)

    def record_and_transcribe(self) -> str:
        """
        Record from mic and transcribe.

        Returns:
            Transcribed text
        """
        print("Recording... Press Enter to stop.")
        self.recorder.start_recording()
        input()
        audio = self.recorder.stop_recording()
        print(f"Recorded {len(audio)} samples ({len(audio)/16000:.1f}s)")
        print("Transcribing...")
        return self.transcriber.transcribe_array(audio)

    def record_duration_and_transcribe(self, seconds: float) -> str:
        """
        Record for fixed duration and transcribe.

        Args:
            seconds: Duration to record

        Returns:
            Transcribed text
        """
        print(f"Recording for {seconds} seconds...")
        audio = self.recorder.record_for_duration(seconds)
        print(f"Recorded {len(audio)} samples")
        print("Transcribing...")
        return self.transcriber.transcribe_array(audio)

    def load_and_transcribe(self, path: str) -> str:
        """
        Load WAV file and transcribe.

        Args:
            path: Path to WAV file

        Returns:
            Transcribed text
        """
        audio, sample_rate = self.loader.load(path)
        print(f"Loaded {len(audio)} samples from {path}")
        print("Transcribing...")
        return self.transcriber.transcribe_array(audio, sample_rate)


__all__ = [
    "MicrophoneRecorder",
    "WavFileLoader",
    "WhisperTranscriber",
    "AudioInput",
]
