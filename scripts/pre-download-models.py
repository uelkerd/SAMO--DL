"""
Pre-download AI models to speed up Docker builds
This script downloads models to a local cache that can be used by Docker builds
"""

import os
import time
import shutil

def download_emotion_model(cache_dir: str):
    """Download the emotion detection model"""
    try:
        print("📥 Downloading emotion model: j-hartmann/emotion-english-distilroberta-base")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = 'j-hartmann/emotion-english-distilroberta-base'
        start_time = time.time()

        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

        duration = time.time() - start_time
        print(f"✅ Downloaded emotion model in {duration:.1f}s")
    except (OSError, RuntimeError, ValueError, HfHubHTTPError) as e:
        print(f"❌ Failed to download emotion model: {e}")
        return False
    return True

def download_t5_model(cache_dir: str):
    """Download the T5 summarization model"""
    try:
        print("📥 Downloading T5 model: t5-small")
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        model_name = 't5-small'
        start_time = time.time()

        T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

        duration = time.time() - start_time
        print(f"✅ Downloaded T5 model in {duration:.1f}s")
    except (OSError, RuntimeError, ValueError, HfHubHTTPError) as e:
        print(f"❌ Failed to download T5 model: {e}")
        return False
    return True

def download_whisper_model(cache_dir: str):
    """Download the Whisper transcription model"""
    print("📥 Downloading Whisper model: base")
    try:
        import whisper
    except ImportError:
        print("❌ Whisper not installed. Run: pip install -U openai-whisper")
        return False
    try:
        model_size = 'base'
        start_time = time.time()
        whisper.load_model(model_size, download_root=cache_dir)
        duration = time.time() - start_time
        print(f"✅ Downloaded Whisper model in {duration:.1f}s")
    except (OSError, RuntimeError, ValueError, HfHubHTTPError) as e:
        print(f"❌ Failed to download Whisper model: {e}")
        return False
    return True

def main():
    """Main function to download all models"""
    print("🚀 SAMO-DL Model Pre-Downloader")
    print("=" * 40)

    # Honor HF_HOME and TRANSFORMERS_CACHE environment variables
    os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ.get("HF_HOME", ""), "transformers"))
    cache_dir = os.getenv("HF_HOME", os.path.join(os.getcwd(), "models_cache"))

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Cache directory: {cache_dir}")
    usage = shutil.disk_usage(cache_dir)
    free_gb = usage.free / (1024**3)
    min_free_gb = 1.5
    if free_gb < min_free_gb:
        print(f"❌ Insufficient disk space: {free_gb:.2f}GB available, {min_free_gb}GB required")
        sys.exit(1)
    print(f"Available disk space: {free_gb:.2f} GB (sufficient)")
    print()

    # Download models
    models = [
        ("Emotion Detection", download_emotion_model),
        ("T5 Summarization", download_t5_model),
        ("Whisper Transcription", download_whisper_model),
    ]

    success_count = 0
    total_start_time = time.time()

    for model_name, download_func in models:
        print(f"🔄 Starting download of {model_name} model...")
        if download_func(cache_dir):
            success_count += 1
        print()

    total_duration = time.time() - total_start_time

    # Summary
    print("=" * 40)
    if success_count == len(models):
        print("✅ All models downloaded successfully!")
        print("💡 You can now copy models_cache to your Docker build context")
        print("   or mount it as a volume during build")
        sys.exit(0)
    else:
        print(f"⚠️  {success_count}/{len(models)} models downloaded successfully")
        print("❌ Partial failure - exiting with error code")
        sys.exit(1)

    print(f"⏱️  Total download time: {total_duration:.1f}s")
    # Show cache size
    try:
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        print(f"📁 Cache size: {cache_size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"ℹ️  Skipped cache size computation: {e}")

if __name__ == "__main__":
    main()
