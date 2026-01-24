"""Speech-to-text transcription using local Whisper (faster-whisper)."""
from faster_whisper import WhisperModel
from pathlib import Path
import tempfile
from scipy.io.wavfile import write as wav_write
import numpy as np
from typing import Optional
import os


class WhisperTranscriber:
    """
    Transcribes audio using local Whisper model (faster-whisper).

    No API key required - runs entirely locally.

    Models available (smallest to largest):
    - tiny, tiny.en     (~75 MB)  - Fastest, least accurate
    - base, base.en     (~140 MB) - Good balance for short phrases
    - small, small.en   (~460 MB) - Better accuracy
    - medium, medium.en (~1.5 GB) - High accuracy
    - large-v2, large-v3 (~3 GB)  - Best accuracy, slowest

    The '.en' variants are English-only and slightly faster/better for English.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        download_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """
        Initialize local Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
                       Add '.en' suffix for English-only models
            device: Device to use ('cpu', 'cuda', 'auto')
            compute_type: Computation type ('float16', 'int8', 'auto')
            download_dir: Explicit directory to cache models (prevents re-downloads)
            local_files_only: If True, only use cached models (no downloads)
        """
        print(f"Loading Whisper model '{model_size}'...")

        # Set explicit cache directory to prevent re-downloads
        if download_dir is None:
            # Use project-local cache or system cache
            cache_base = Path.home() / ".cache" / "whisper"
            download_dir = str(cache_base)
            cache_base.mkdir(parents=True, exist_ok=True)
            print(f"Using cache directory: {download_dir}")
        else:
            Path(download_dir).mkdir(parents=True, exist_ok=True)

        # Auto-detect best settings
        if device == "auto":
            device = "cpu"  # Safe default, faster-whisper will use GPU if available

        if compute_type == "auto":
            compute_type = "int8"  # Good balance of speed and quality on CPU

        # Initialize model with explicit cache directory
        # This prevents re-downloading if model is already cached
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_dir,  # Explicit cache location prevents re-downloads
                local_files_only=local_files_only,  # Prevent downloads if True
            )
            print(f"Whisper model loaded on {device}")
        except Exception as e:
            if local_files_only:
                raise RuntimeError(
                    f"Model '{model_size}' not found in cache ({download_dir}). "
                    f"Set local_files_only=False to download it."
                ) from e
            raise

    def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = "en",
    ) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect

        Returns:
            Transcribed text string
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,  # Filter out silence
        )

        # Combine all segments
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> str:
        """
        Transcribe numpy audio array.

        Args:
            audio: Audio data as numpy array (int16 or float32)
            sample_rate: Sample rate of audio
            language: Language code or None for auto-detect

        Returns:
            Transcribed text string
        """
        # Convert to float32 if needed (faster-whisper expects float32)
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Save to temporary file (faster-whisper works best with files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Save as int16 WAV
            audio_int16 = (audio_float * 32768).astype(np.int16)
            wav_write(f.name, sample_rate, audio_int16)
            temp_path = Path(f.name)

        try:
            return self.transcribe_file(temp_path, language)
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    def transcribe_array_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> str:
        """
        Transcribe numpy audio array with real-time segment display.
        
        Args:
            audio: Audio data as numpy array (int16 or float32)
            sample_rate: Sample rate of audio
            language: Language code or None for auto-detect
            
        Returns:
            Transcribed text string
        """
        # Convert to float32 if needed (faster-whisper expects float32)
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Save to temporary file (faster-whisper works best with files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Save as int16 WAV
            audio_int16 = (audio_float * 32768).astype(np.int16)
            wav_write(f.name, sample_rate, audio_int16)
            temp_path = Path(f.name)

        try:
            print("Transcribing...")
            segments, info = self.model.transcribe(
                str(temp_path),
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence
            )
            
            # Display segments as they're processed
            full_text = []
            for i, segment in enumerate(segments, 1):
                text = segment.text.strip()
                if text:  # Only display non-empty segments
                    full_text.append(text)
                    print(f"  [{i}] {text}")
            
            return " ".join(full_text).strip()
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
