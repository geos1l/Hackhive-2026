# Testing Whisper Cache Fix

## What Was Fixed

The `WhisperTranscriber` class now uses an explicit cache directory to prevent models from being re-downloaded on every initialization. This fixes the slow download issue you were experiencing.

### Changes Made

1. **Added `download_root` parameter** - Explicitly sets cache directory to `~/.cache/whisper`
2. **Added `local_files_only` parameter** - Option to prevent downloads if model isn't cached
3. **Cache directory creation** - Automatically creates cache directory if it doesn't exist

## How to Test

### Option 1: Using Virtual Environment (Recommended)

If you have a virtual environment set up:

```bash
# Activate your venv
# Windows:
.\venv\Scripts\activate
# or
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Run the test
python test_whisper_cache_standalone.py
```

### Option 2: Install Dependencies First

If dependencies aren't installed:

```bash
# Install required packages
pip install faster-whisper numpy scipy

# Then run the test
python test_whisper_cache_standalone.py
```

### Option 3: Quick Manual Test

You can also test it manually in Python:

```python
import time
from src.audio.transcriber import WhisperTranscriber

# First initialization (will download if needed)
print("First load...")
start = time.time()
t1 = WhisperTranscriber(model_size="tiny.en")
print(f"Took {time.time() - start:.2f} seconds")

# Second initialization (should use cache)
print("\nSecond load (should be faster)...")
start = time.time()
t2 = WhisperTranscriber(model_size="tiny.en")
print(f"Took {time.time() - start:.2f} seconds")

# If second is much faster, cache is working!
```

## What to Look For

### ✅ Success Indicators

1. **First run**: May take 10-30 seconds (downloading model)
2. **Second run**: Should take 1-5 seconds (using cache)
3. **Cache directory**: Should see files in `~/.cache/whisper/` or `C:\Users\<user>\.cache\whisper\`
4. **No re-downloads**: Subsequent initializations should be fast

### ❌ Problem Indicators

1. **Both runs are slow**: Model is being re-downloaded each time
2. **Cache directory is empty**: Models aren't being cached
3. **Different cache locations**: Multiple cache directories being used

## Expected Behavior

### Before Fix
- Model downloaded on every `WhisperTranscriber()` call
- Slow initialization every time
- Network usage on every run

### After Fix
- Model downloaded once on first use
- Cached to `~/.cache/whisper/`
- Subsequent initializations are fast (just loading from disk)
- No network usage after first download

## Cache Location

- **Windows**: `C:\Users\<username>\.cache\whisper\`
- **Linux/Mac**: `~/.cache/whisper/`

You can verify the cache is working by:
1. Running the test twice
2. Checking that the cache directory contains model files
3. Comparing load times (second should be much faster)

## Troubleshooting

### If test fails with "ModuleNotFoundError"

Install dependencies:
```bash
pip install faster-whisper
```

### If second load is still slow

1. Check cache directory exists: `~/.cache/whisper/`
2. Check cache has files: `ls ~/.cache/whisper/` (Linux/Mac) or `dir C:\Users\<user>\.cache\whisper\` (Windows)
3. Verify `download_root` is being set correctly in the code

### If you want to force re-download

Delete the cache directory:
```bash
# Windows
rmdir /s C:\Users\<username>\.cache\whisper

# Linux/Mac
rm -rf ~/.cache/whisper
```

## Code Verification

The fix is in `src/audio/transcriber.py`:

```python
# Line ~43-48: Cache directory setup
if download_dir is None:
    cache_base = Path.home() / ".cache" / "whisper"
    download_dir = str(cache_base)
    cache_base.mkdir(parents=True, exist_ok=True)

# Line ~58-60: Using download_root parameter
self.model = WhisperModel(
    model_size,
    device=device,
    compute_type=compute_type,
    download_root=download_dir,  # <-- This prevents re-downloads
    local_files_only=local_files_only,
)
```
