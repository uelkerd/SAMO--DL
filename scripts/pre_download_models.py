#!/usr/bin/env python3
"""Pre-download models for Docker build optimization."""
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Pre-download all required models."""
    # Get model directory from environment variable, fallback to /app/models
    model_dir = os.environ.get("MODEL_DIR", "/app/models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set cache environment variables to use the same directory
    os.environ["HF_HOME"] = model_dir
    
    print(f"📁 Using model directory: {model_dir}")

    print("🚀 Pre-downloading SAMO emotion model...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Try the real SAMO model first
        model_name = "duelker/samo-goemotions-deberta-v3-large"
        print(f"Downloading {model_name}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
            _model = AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir=model_dir
            )
            print("✅ SAMO model downloaded successfully")
        except Exception as e:
            print(f"⚠️ SAMO model not available ({e}), trying fallback...")
            # Fallback to a compatible emotion model
            fallback_model = "j-hartmann/emotion-english-distilroberta-base"
            print(f"Downloading fallback model {fallback_model}...")
            _tokenizer = AutoTokenizer.from_pretrained(fallback_model, cache_dir=model_dir)
            _model = AutoModelForSequenceClassification.from_pretrained(
                fallback_model, cache_dir=model_dir
            )
            print("✅ Fallback emotion model downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading emotion model: {e}")
        raise

    print("🚀 Pre-downloading T5 summarization model...")
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        t5_model = "t5-small"
        print(f"Downloading {t5_model}...")
        _t5_tokenizer = T5Tokenizer.from_pretrained(t5_model, cache_dir=model_dir)
        _t5_model_obj = T5ForConditionalGeneration.from_pretrained(
            t5_model, cache_dir=model_dir
        )
        print("✅ T5 model downloaded successfully")
    except Exception as e:
        print(f"⚠️ Error downloading T5 model: {e}")
        print("📝 T5 summarization will be downloaded at runtime if needed")
        # Don't fail the build for T5 - continue without it

    print("🚀 Pre-downloading Whisper model...")
    try:
        # Check if whisper is available
        try:
            import whisper
            whisper_model = "base"
            print(f"Downloading Whisper {whisper_model}...")
            whisper.load_model(whisper_model, download_root=model_dir)
            print("✅ Whisper model downloaded successfully")
        except ImportError:
            print("⚠️ Whisper not available - skipping Whisper model download")
            print("📝 Whisper will be downloaded at runtime if needed")
    except Exception as e:
        print(f"❌ Error downloading Whisper model: {e}")
        # Don't fail the entire build for Whisper - continue without it
        print(
            "⚠️ Continuing without Whisper model - will be downloaded at "
            "runtime if needed"
        )

    print("🎉 Core models pre-downloaded successfully!")


if __name__ == "__main__":
    main()
