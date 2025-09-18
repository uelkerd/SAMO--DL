#!/usr/bin/env python3
"""
Model validation script for Docker builds.
Tests that all required models are accessible and load correctly.
"""
import os
import sys

def main():
    print("🧪 Testing model accessibility...")
    
    # Test transformers cache
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "duelker/samo-goemotions-deberta-v3-large", 
            cache_dir="/app/models", 
            local_files_only=True
        )
        print("✅ DeBERTa tokenizer loads successfully")
    except Exception as e:
        print(f"❌ DeBERTa tokenizer failed: {e}")
        sys.exit(1)
    
    try:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(
            "t5-small", 
            cache_dir="/app/models", 
            local_files_only=True
        )
        print("✅ T5 tokenizer loads successfully")
    except Exception as e:
        print(f"❌ T5 tokenizer failed: {e}")
        sys.exit(1)
    
    # Test Whisper model file exists
    whisper_path = "/app/models/base.pt"
    if os.path.exists(whisper_path):
        print(f"✅ Whisper model file exists at {whisper_path}")
    else:
        print(f"❌ Whisper model file missing at {whisper_path}")
        sys.exit(1)
    
    print("🎉 All model validation tests passed!")

if __name__ == "__main__":
    main()
