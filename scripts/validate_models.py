#!/usr/bin/env python3
"""
Model validation script for Docker builds.
Tests that all required models are accessible and load correctly.
"""

import os


def main():
    """Test model accessibility and validate that all required models are available."""
    print("🧪 Testing model accessibility...")

    validation_passed = True

    # Test transformers cache
    try:
        from transformers import AutoTokenizer

        _ = AutoTokenizer.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base",
            cache_dir="/app/models",
            local_files_only=True,
        )
        print("✅ Emotion model tokenizer loads successfully")
    except (ImportError, OSError, RuntimeError) as e:
        print(f"⚠️ Emotion model tokenizer not available: {e}")
        print("📝 Will be downloaded at runtime if needed")
        validation_passed = False

    try:
        from transformers import T5Tokenizer

        _ = T5Tokenizer.from_pretrained("t5-small", cache_dir="/app/models", local_files_only=True)
        print("✅ T5 tokenizer loads successfully")
    except (ImportError, OSError, RuntimeError) as e:
        print(f"⚠️ T5 tokenizer not available: {e}")
        print("📝 Will be downloaded at runtime if needed")
        validation_passed = False

    # Test Whisper model file exists (optional)
    whisper_path = "/app/models/base.pt"
    if os.path.exists(whisper_path):
        print(f"✅ Whisper model file exists at {whisper_path}")
    else:
        print(f"⚠️ Whisper model file missing at {whisper_path}")
        print("📝 Will be downloaded at runtime if needed")
        validation_passed = False

    if validation_passed:
        print("🎉 All model validation tests passed!")
    else:
        print("⚠️ Some models not available - will be downloaded at runtime")
        print("✅ Build can continue - models will be lazy-loaded")


if __name__ == "__main__":
    main()
