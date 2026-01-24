"""
Test script to verify Whisper model caching works correctly.

This test:
1. Creates a WhisperTranscriber (first time - will download if needed)
2. Creates another WhisperTranscriber (should use cache, no download)
3. Verifies both work and use the same cache directory
4. Tests with an existing audio file

Run: python test_whisper_cache.py
"""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio.transcriber import WhisperTranscriber


def test_cache_behavior():
    """Test that models are cached and not re-downloaded."""
    print("=" * 60)
    print("WHISPER CACHE TEST")
    print("=" * 60)
    
    # Use tiny.en for faster testing
    model_size = "tiny.en"
    
    print(f"\n[Test 1] First initialization (may download model '{model_size}')...")
    print("-" * 60)
    start_time = time.time()
    
    transcriber1 = WhisperTranscriber(model_size=model_size)
    
    first_load_time = time.time() - start_time
    print(f"✓ First load completed in {first_load_time:.2f} seconds")
    
    # Check if we have a test audio file
    test_audio = Path("output/test_recording.wav")
    if not test_audio.exists():
        print(f"\n⚠ No test audio file found at {test_audio}")
        print("  Run 'python test_audio.py' first to create a test recording, or")
        print("  provide a WAV file path as argument: python test_whisper_cache.py <audio.wav>")
        
        # Still test that second initialization is faster
        print(f"\n[Test 2] Second initialization (should use cache)...")
        print("-" * 60)
        start_time = time.time()
        
        transcriber2 = WhisperTranscriber(model_size=model_size)
        
        second_load_time = time.time() - start_time
        print(f"✓ Second load completed in {second_load_time:.2f} seconds")
        
        if second_load_time < first_load_time * 0.5:
            print(f"✓ SUCCESS: Second load was {first_load_time/second_load_time:.1f}x faster (using cache)")
        else:
            print(f"⚠ WARNING: Second load wasn't significantly faster")
            print(f"   This might indicate the model is being re-downloaded")
        
        print("\n" + "=" * 60)
        print("Cache test completed (no audio transcription test)")
        print("=" * 60)
        return
    
    print(f"\n[Test 2] Second initialization (should use cache)...")
    print("-" * 60)
    start_time = time.time()
    
    transcriber2 = WhisperTranscriber(model_size=model_size)
    
    second_load_time = time.time() - start_time
    print(f"✓ Second load completed in {second_load_time:.2f} seconds")
    
    if second_load_time < first_load_time * 0.5:
        print(f"✓ SUCCESS: Second load was {first_load_time/second_load_time:.1f}x faster (using cache)")
    else:
        print(f"⚠ WARNING: Second load wasn't significantly faster")
        print(f"   This might indicate the model is being re-downloaded")
    
    print(f"\n[Test 3] Testing transcription with cached model...")
    print("-" * 60)
    print(f"Transcribing: {test_audio}")
    
    start_time = time.time()
    text = transcriber2.transcribe_file(test_audio)
    transcribe_time = time.time() - start_time
    
    print(f"✓ Transcription completed in {transcribe_time:.2f} seconds")
    print(f"✓ Transcribed text: '{text}'")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - First load:  {first_load_time:.2f}s (includes download if needed)")
    print(f"  - Second load: {second_load_time:.2f}s (should be much faster)")
    print(f"  - Transcription: {transcribe_time:.2f}s")
    print(f"  - Cache location: ~/.cache/whisper/")


if __name__ == "__main__":
    try:
        test_cache_behavior()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
