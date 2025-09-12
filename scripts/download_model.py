#!/usr/bin/env python3
"""
Model pre-downloading script for DeBERTa emotion detection model.
This script downloads the model during Docker build to avoid OOM during startup.
"""

import os
import sys

def download_deberta_model():
    """Download the DeBERTa model and tokenizer."""
    try:
        # Set environment variables for DeBERTa
        os.environ['USE_DEBERTA'] = 'true'
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

        # Model configuration
        model_name = 'duelker/samo-goemotions-deberta-v3-large'
        cache_dir = '/app/models'

        print(f"🚀 Pre-downloading DeBERTa model: {model_name}")
        print(f"📁 Cache directory: {cache_dir}")

        # Import transformers
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False, 
            cache_dir=cache_dir
        )
        print("✅ Tokenizer downloaded successfully")

        # Download model
        print("📥 Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        print("✅ Model downloaded successfully")

        # Verify download
        print(f"📊 Model config: {model.config.num_labels} emotion classes")
        print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")

        print("🎉 DeBERTa model pre-download completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error downloading DeBERTa model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_deberta_model()
    sys.exit(0 if success else 1)
