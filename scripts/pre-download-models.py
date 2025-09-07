#!/usr/bin/env python3
"""
Pre-download AI models to speed up Docker builds
This script downloads models to a local cache that can be used by Docker builds
"""

import os
import sys
import time
import shutil
from pathlib import Path

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
    except Exception as e:
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
    except Exception as e:
        print(f"❌ Failed to download T5 model: {e}")
        return False
    return True

def download_whisper_model(cache_dir: str):
    """Download the Whisper transcription model"""
    try:
        print("📥 Downloading Whisper model: base")
        import whisper

        model_size = 'base'
        start_time = time.time()

        whisper.load_model(model_size, download_root=cache_dir)

        duration = time.time() - start_time
        print(f"✅ Downloaded Whisper model in {duration:.1f}s")
    except Exception as e:
        print(f"❌ Failed to download Whisper model: {e}")
        return False
    return True

def main():
    """Main function to download all models"""
    print("🚀 SAMO-DL Model Pre-Downloader")
    print("=" * 40)

    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), "models_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Cache directory: {cache_dir}")
    usage = shutil.disk_usage(cache_dir)
    print(f"Available disk space: {usage.free // (1024 * 1024)} MB")
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
        print(f"✅ All models downloaded successfully!")
        print("💡 You can now copy models_cache to your Docker build context")
        print("   or mount it as a volume during build")
    else:
        print(f"⚠️  {success_count}/{len(models)} models downloaded successfully")

    print(f"⏱️  Total download time: {total_duration:.1f}s")
    # Show cache size
    try:
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        print(f"📁 Cache size: {cache_size / (1024**3):.2f} GB")
    except:
        print("📁 Cache directory created")

if __name__ == "__main__":
    main()
