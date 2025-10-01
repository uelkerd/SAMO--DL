#!/usr/bin/env python3
"""
Test DeBERTa model loading and inference in isolation.

This script tests the updated model_utils.py with direct DeBERTa loading
to ensure it works before integrating with the full API.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for DeBERTa testing
os.environ['USE_DEBERTA'] = 'true'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def test_deberta_loading():
    """Test DeBERTa model loading and inference in isolation."""
    try:
        logger.info("🔧 Testing DeBERTa isolated loading...")

        # Add the deployment directory to path to import model_utils
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'cloud-run'))

        # Import the updated model_utils
        from model_utils import ensure_model_loaded, predict_emotions, get_model_status

        logger.info("✅ Successfully imported model_utils")

        # Test model loading
        logger.info("🔄 Loading DeBERTa model...")
        success = ensure_model_loaded()

        if not success:
            logger.error("❌ Failed to load DeBERTa model")
            return False

        logger.info("✅ DeBERTa model loaded successfully")

        # Get model status
        status = get_model_status()
        logger.info(f"📊 Model status: {status}")

        # Test emotion prediction
        test_texts = [
            "I am so happy today!",
            "This is absolutely terrible",
            "I'm feeling a bit nervous about the presentation",
            "That was an amazing achievement!"
        ]

        logger.info("🧪 Testing emotion predictions...")
        for text in test_texts:
            result = predict_emotions(text)
            if 'error' in result:
                logger.error(f"❌ Prediction failed for '{text}': {result['error']}")
                return False

            # Log top 3 emotions
            top_emotions = result['emotions'][:3]
            logger.info(f"📝 '{text}' -> {', '.join([f'{e['emotion']}:{e['confidence']:.3f}' for e in top_emotions])}")

        logger.info("✅ All DeBERTa tests passed!")
        return True

    except Exception as e:
        logger.exception(f"❌ DeBERTa isolated test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_deberta_loading()
    sys.exit(0 if success else 1)
