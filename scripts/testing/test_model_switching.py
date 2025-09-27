#!/usr/bin/env python3
"""
Test Model Switching - Verify DeBERTa integration works

This script tests that the updated model utilities can switch between
production and DeBERTa models correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set protobuf compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def test_production_model():
    """Test loading production model."""
    print("🧪 Testing Production Model (6 emotions)")
    print("-" * 40)

    # Set environment variables
    os.environ['USE_DEBERTA'] = 'false'

    # Import and test
    sys.path.insert(0, str(project_root / 'deployment' / 'cloud-run'))
    import model_utils

    # Reset global state
    model_utils.emotion_pipeline = None
    model_utils.model_loaded = False
    model_utils.model_loading = False

    success = model_utils.ensure_model_loaded()
    print(f"✅ Production model loaded: {success}")

    if success:
        # Test prediction
        test_text = "I am so happy today!"
        try:
            result = model_utils.predict_emotions(test_text)
            top_emotion = result[0]['emotion'] if result else 'unknown'
            confidence = result[0]['confidence'] if result else 0.0
            print(f"🎯 Prediction: {top_emotion} ({confidence:.3f})")
            print(f"📊 Emotions detected: {len(result) if result else 0}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

    return success

def test_deberta_model():
    """Test loading DeBERTa model."""
    print("\\n🧪 Testing DeBERTa Model (28 emotions)")
    print("-" * 40)

    # Set environment variables
    os.environ['USE_DEBERTA'] = 'true'

    # Need to reload the module to pick up environment changes
    import importlib
    import model_utils
    importlib.reload(model_utils)

    # Reset global state
    model_utils.emotion_pipeline = None
    model_utils.model_loaded = False
    model_utils.model_loading = False

    success = model_utils.ensure_model_loaded()
    print(f"✅ DeBERTa model loaded: {success}")

    if success:
        # Test prediction
        test_text = "I am so happy today!"
        try:
            result = model_utils.predict_emotions(test_text)
            top_emotion = result[0]['emotion'] if result else 'unknown'
            confidence = result[0]['confidence'] if result else 0.0
            print(f"🎯 Prediction: {top_emotion} ({confidence:.3f})")
            print(f"📊 Emotions detected: {len(result) if result else 0}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

    return success

def main():
    """Main test function."""
    print("🔄 Model Switching Test")
    print("=" * 25)
    print("Testing both production and DeBERTa models")
    print()

    # Test production model
    prod_success = test_production_model()

    # Test DeBERTa model
    deberta_success = test_deberta_model()

    print("\\n" + "=" * 40)
    print("📊 TEST RESULTS")
    print("=" * 40)
    print(f"✅ Production Model: {'WORKING' if prod_success else 'FAILED'}")
    print(f"✅ DeBERTa Model: {'WORKING' if deberta_success else 'FAILED'}")

    if prod_success and deberta_success:
        print("\\n🎉 SUCCESS! Both models working!")
        print("🚀 Ready for deployment with model switching")
        print("\\n💡 To use DeBERTa in production:")
        print("   Set environment variable: USE_DEBERTA=true")
        print("   The model will automatically switch to 28 emotions")
    else:
        print("\\n❌ Some tests failed - check error messages above")

if __name__ == "__main__":
    main()
