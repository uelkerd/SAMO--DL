#!/usr/bin/env python3
"""
Quick Model Test - Test available models individually

This script tests models one by one to avoid loading issues and provide
immediate feedback on what works.
"""

import time
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bert_model():
    """Test the current BERT model."""
    print("🧪 Testing Current BERT Model")
    print("-" * 40)

    try:
        from src.models.emotion_detection.samo_bert_emotion_classifier import create_samo_bert_emotion_classifier

        print("📥 Loading BERT model...")
        model, loss_fn = create_samo_bert_emotion_classifier()
        print("✅ BERT model loaded successfully!")

        # Test inference
        test_texts = [
            "I am so happy today! This is amazing!",
            "I'm feeling really sad and disappointed about this situation.",
            "I'm frustrated but hopeful about the future.",
            "Thank you so much for your help!",
            "I feel anxious and worried about what might happen next."
        ]

        print("\n🔬 Running inference tests...")

        start_time = time.time()
        results = model.predict_emotions(test_texts[:3], threshold=0.5)  # Test with smaller batch first
        inference_time = time.time() - start_time

        print("✅ Inference completed successfully!")
        print(".2f")
        print(f"📊 Results structure: {list(results.keys())}")

        # Show sample results
        print("\n📋 Sample Results:")
        for i, text in enumerate(test_texts[:3]):
            emotions = results.get('emotions', [[]])[i] if results.get('emotions') else []
            print(f"Text: {text[:50]}...")
            print(f"Emotions: {emotions}")
            print()

        return True

    except Exception as e:
        print(f"❌ BERT model test failed: {e}")
        return False

def test_production_model():
    """Test the production DistilRoBERTa model."""
    print("🧪 Testing Production Model (DistilRoBERTa)")
    print("-" * 40)

    try:
        from transformers import pipeline

        print("📥 Loading production model...")
        model_name = "j-hartmann/emotion-english-distilroberta-base"

        clf = pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU
            top_k=None,
            truncation=True,
            max_length=512
        )

        print("✅ Production model loaded successfully!")

        # Test inference
        test_texts = [
            "I am so happy today!",
            "I'm feeling really sad.",
            "I'm frustrated and angry."
        ]

        print("\n🔬 Running inference tests...")

        start_time = time.time()
        results = clf(test_texts)
        inference_time = time.time() - start_time

        print("✅ Inference completed successfully!")
        print(".2f")

        # Show sample results
        print("\n📋 Sample Results:")
        for i, text in enumerate(test_texts):
            result = results[i][0] if results and len(results) > i and results[i] else {}
            emotion = result.get('label', 'unknown')
            confidence = result.get('score', 0.0)
            print(f"Text: {text}")
            print(".3f")
            print()

        return True

    except Exception as e:
        print(f"❌ Production model test failed: {e}")
        return False

def test_deberta_model():
    """Test the DeBERTa model with error handling."""
    print("🧪 Testing DeBERTa Model")
    print("-" * 40)

    # Set environment variable to handle protobuf issues
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        print("📥 Loading DeBERTa model (this may take a while)...")
        model_name = "duelker/samo-goemotions-deberta-v3-large"

        # Try to load with various fallbacks
        try:
            clf = pipeline(
                "text-classification",
                model=model_name,
                device=-1,  # CPU
                top_k=None,
                truncation=True,
                max_length=256,  # Shorter for DeBERTa
                model_kwargs={"torch_dtype": "float32"}
            )
        except Exception as e1:
            print(f"⚠️ Pipeline loading failed, trying manual loading: {e1}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype="float32"
                )
                clf = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                    top_k=None,
                    truncation=True,
                    max_length=256
                )
            except Exception as e2:
                print(f"❌ DeBERTa model loading failed: {e2}")
                print("💡 This might be due to:")
                print("   - Network connectivity issues")
                print("   - Model download timeout")
                print("   - Protobuf compatibility issues")
                print("   - Insufficient memory")
                return False

        print("✅ DeBERTa model loaded successfully!")

        # Test with shorter text due to model size
        test_texts = [
            "I am happy!",
            "I feel sad.",
            "I'm angry."
        ]

        print("\n🔬 Running inference tests...")

        start_time = time.time()
        results = clf(test_texts)
        inference_time = time.time() - start_time

        print("✅ Inference completed successfully!")
        print(".2f")

        # Show sample results
        print("\n📋 Sample Results:")
        for i, text in enumerate(test_texts):
            result = results[i][0] if results and len(results) > i and results[i] else {}
            emotion = result.get('label', 'unknown')
            confidence = result.get('score', 0.0)
            print(f"Text: {text}")
            print(".3f")
            print()

        return True

    except Exception as e:
        print(f"❌ DeBERTa model test failed: {e}")
        return False

def main():
    """Run all model tests."""
    print("🚀 SAMO Model Comparison Test Suite")
    print("=" * 50)
    print("Testing emotion detection models individually...")
    print()

    results = {}

    # Test BERT model
    print("1️⃣ Testing BERT Model")
    results['bert'] = test_bert_model()
    print()

    # Test Production model
    print("2️⃣ Testing Production Model")
    results['production'] = test_production_model()
    print()

    # Test DeBERTa model
    print("3️⃣ Testing DeBERTa Model")
    results['deberta'] = test_deberta_model()
    print()

    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 30)
    successful = sum(results.values())
    total = len(results)

    print(f"✅ Models working: {successful}/{total}")

    working_models = [name for name, status in results.items() if status]
    if working_models:
        print(f"🤖 Working models: {', '.join(working_models)}")

    failed_models = [name for name, status in results.items() if not status]
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")

    print("\n💡 Next Steps:")
    if results.get('bert'):
        print("   - BERT model is ready for integration")
    if results.get('production'):
        print("   - Production model is working as baseline")
    if not results.get('deberta'):
        print("   - DeBERTa needs troubleshooting (network/protobuf issues)")

    print("\n🎯 Recommendation:")
    if results.get('bert'):
        print("   Use BERT model as primary, keep production as fallback")
    elif results.get('production'):
        print("   Stick with production model until BERT issues resolved")

if __name__ == "__main__":
    main()
