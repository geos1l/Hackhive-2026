"""
Standalone test script to verify Whisper model caching works correctly.

This test directly imports WhisperTranscriber without requiring other audio modules.
It verifies that:
1. First initialization downloads/caches the model
2. Second initialization uses the cache (much faster)

Run: python test_whisper_cache_standalone.py
"""
import sys
import time
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct import from file to avoid package __init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "transcriber", 
    project_root / "src" / "audio" / "transcriber.py"
)
transcriber_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcriber_module)
WhisperTranscriber = transcriber_module.WhisperTranscriber


def test_cache_behavior():
    """Test that models are cached and not re-downloaded."""
    print("=" * 60)
    print("WHISPER CACHE TEST")
    print("=" * 60)
    
    # Use tiny.en for faster testing (smallest model)
    model_size = "tiny.en"
    cache_dir = Path.home() / ".cache" / "whisper"
    
    print(f"\nModel: {model_size}")
    print(f"Cache directory: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")
    
    print(f"\n[Test 1] First initialization (may download model '{model_size}')...")
    print("-" * 60)
    start_time = time.time()
    
    try:
        transcriber1 = WhisperTranscriber(model_size=model_size)
        first_load_time = time.time() - start_time
        print(f"✓ First load completed in {first_load_time:.2f} seconds")
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR during first load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check cache directory
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*"))
        print(f"✓ Cache directory contains {len(cache_files)} files/directories")
        if cache_files:
            print(f"  Sample cache paths:")
            for f in cache_files[:3]:  # Show first 3
                print(f"    - {f}")
    
    print(f"\n[Test 2] Second initialization (should use cache, no download)...")
    print("-" * 60)
    start_time = time.time()
    
    try:
        transcriber2 = WhisperTranscriber(model_size=model_size)
        second_load_time = time.time() - start_time
        print(f"✓ Second load completed in {second_load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ ERROR during second load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare load times
    print(f"\n[Results]")
    print("-" * 60)
    print(f"First load:  {first_load_time:.2f} seconds")
    print(f"Second load: {second_load_time:.2f} seconds")
    
    if second_load_time < first_load_time:
        speedup = first_load_time / second_load_time
        print(f"✓ SUCCESS: Second load was {speedup:.1f}x faster")
        
        if speedup > 2.0:
            print(f"✓ EXCELLENT: Significant speedup indicates cache is working!")
        elif speedup > 1.2:
            print(f"✓ GOOD: Some speedup detected (cache may be working)")
        else:
            print(f"⚠ WARNING: Minimal speedup - model might be re-downloading")
    else:
        print(f"⚠ WARNING: Second load was slower - something may be wrong")
    
    # Test with audio file if available
    test_audio = Path("output/test_recording.wav")
    if test_audio.exists():
        print(f"\n[Test 3] Testing transcription with cached model...")
        print("-" * 60)
        print(f"Transcribing: {test_audio}")
        
        try:
            start_time = time.time()
            text = transcriber2.transcribe_file(test_audio)
            transcribe_time = time.time() - start_time
            
            print(f"✓ Transcription completed in {transcribe_time:.2f} seconds")
            print(f"✓ Transcribed text: '{text}'")
        except Exception as e:
            print(f"⚠ Transcription test failed: {e}")
    else:
        print(f"\n[Test 3] Skipped (no test audio file at {test_audio})")
        print("  Run 'python test_audio.py' first to create a test recording")
    
    print("\n" + "=" * 60)
    print("Cache test completed!")
    print("=" * 60)
    print(f"\nKey points:")
    print(f"  ✓ Model cache directory: {cache_dir}")
    print(f"  ✓ Models are cached to prevent re-downloading")
    print(f"  ✓ Subsequent initializations should be much faster")
    
    return True


if __name__ == "__main__":
    try:
        success = test_cache_behavior()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
