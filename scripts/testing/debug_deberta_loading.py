#!/usr/bin/env python3
"""
Debug DeBERTa Model Loading Issues

This script isolates and debugs the DeBERTa model loading problems:
- Protobuf compatibility issues
- Network/download problems
- Model configuration issues

Target: duelker/samo-goemotions-deberta-v3-large
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up environment variables to handle protobuf issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deberta_loading_method_1():
    """Method 1: Standard pipeline loading with error handling."""
    print("🔧 Method 1: Standard Pipeline Loading")
    print("-" * 40)

    try:
        from transformers import pipeline
        import torch

        model_name = "duelker/samo-goemotions-deberta-v3-large"
        print(f"📦 Loading {model_name}...")

        start_time = time.time()
        clf = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU
            top_k=None,
            truncation=True,
            max_length=256,
            model_kwargs={"torch_dtype": torch.float32}
        )
        load_time = time.time() - start_time

        print(".2f")

        # Test inference
        test_text = "I am feeling happy today!"
        start_time = time.time()
        result = clf(test_text)
        inference_time = time.time() - start_time

        print(".3f")
        print(f"✅ Result: {result[0][0] if result and result[0] else 'No result'}")

        return clf

    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        return None

def test_deberta_loading_method_2():
    """Method 2: Manual loading with fallbacks."""
    print("\n🔧 Method 2: Manual Loading with Fallbacks")
    print("-" * 40)

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        import torch

        model_name = "duelker/samo-goemotions-deberta-v3-large"
        print(f"📦 Manual loading {model_name}...")

        start_time = time.time()

        # Try different tokenizer configurations
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            print("✅ Fast tokenizer loaded")
        except Exception as e1:
            print(f"⚠️ Fast tokenizer failed: {e1}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                print("✅ Slow tokenizer loaded")
            except Exception as e2:
                print(f"❌ Both tokenizers failed: {e2}")
                return None

        # Try different model configurations
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded with low_cpu_mem_usage=True")
        except Exception as e1:
            print(f"⚠️ Low memory loading failed: {e1}")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )
                print("✅ Model loaded with low_cpu_mem_usage=False")
            except Exception as e2:
                print(f"❌ Model loading failed: {e2}")
                return None

        load_time = time.time() - start_time
        print(".2f")

        # Create pipeline from loaded components
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            top_k=None,
            truncation=True,
            max_length=256
        )

        # Test inference
        test_text = "I am feeling happy today!"
        start_time = time.time()
        result = clf(test_text)
        inference_time = time.time() - start_time

        print(".3f")
        print(f"✅ Result: {result[0][0] if result and result[0] else 'No result'}")

        return clf

    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        return None

def test_deberta_loading_method_3():
    """Method 3: Force download and cache first."""
    print("\n🔧 Method 3: Pre-download to Cache")
    print("-" * 40)

    try:
        from huggingface_hub import snapshot_download
        from transformers import pipeline
        import torch

        model_name = "duelker/samo-goemotions-deberta-v3-large"
        cache_dir = "/tmp/deberta_cache"

        print(f"📥 Pre-downloading {model_name} to {cache_dir}...")

        start_time = time.time()
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.safetensors"]  # Skip large files first
        )
        download_time = time.time() - start_time

        print(".2f")

        # Now load from local cache
        print("📦 Loading from local cache...")
        start_time = time.time()
        clf = pipeline(
            "text-classification",
            model=cache_dir,
            tokenizer=cache_dir,
            device=-1,
            top_k=None,
            truncation=True,
            max_length=256,
            model_kwargs={"torch_dtype": torch.float32}
        )
        load_time = time.time() - start_time

        print(".2f")

        # Test inference
        test_text = "I am feeling happy today!"
        start_time = time.time()
        result = clf(test_text)
        inference_time = time.time() - start_time

        print(".3f")
        print(f"✅ Result: {result[0][0] if result and result[0] else 'No result'}")

        return clf

    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        return None

def test_model_comparison():
    """Compare DeBERTa with current production model."""
    print("\n🔬 Model Comparison Test")
    print("-" * 30)

    # Test production model first
    print("📊 Testing Production Model (j-hartmann/emotion-english-distilroberta-base)")
    try:
        from transformers import pipeline
        import torch

        prod_clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1,
            top_k=None,
            truncation=True,
            max_length=512
        )

        test_text = "I am feeling happy today!"
        prod_result = prod_clf(test_text)
        print(f"🎯 Production result: {prod_result[0][0] if prod_result and prod_result[0] else 'No result'}")

    except Exception as e:
        print(f"❌ Production model failed: {e}")
        return

    # Test DeBERTa if available
    print("\\n📊 Testing DeBERTa Model")
    deberta_result = None

    methods = [test_deberta_loading_method_1, test_deberta_loading_method_2, test_deberta_loading_method_3]

    for i, method in enumerate(methods, 1):
        print(f"\\n🔄 Trying Method {i}...")
        deberta_clf = method()
        if deberta_clf:
            test_text = "I am feeling happy today!"
            deberta_result = deberta_clf(test_text)
            print(f"🎯 DeBERTa result: {deberta_result[0][0] if deberta_result and deberta_result[0] else 'No result'}")
            break

    if deberta_result:
        print("\\n✅ SUCCESS: DeBERTa model working!")
        print("📋 Comparison:")
        print(f"   Production: {prod_result[0][0]['label']} ({prod_result[0][0]['score']:.3f})")
        print(f"   DeBERTa: {deberta_result[0][0]['label']} ({deberta_result[0][0]['score']:.3f})")
    else:
        print("\\n❌ All DeBERTa loading methods failed")

def main():
    """Main debugging function."""
    print("🐛 DeBERTa Model Loading Debugger")
    print("=" * 50)
    print("Target: duelker/samo-goemotions-deberta-v3-large")
    print("Goal: Fix loading issues and compare with production model")
    print()

    # Check environment
    print("🔍 Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   Protobuf implementation: {os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'Not set')}")
    print()

    # Run tests
    test_model_comparison()

    print("\\n" + "=" * 50)
    print("🐛 DEBUGGING COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
