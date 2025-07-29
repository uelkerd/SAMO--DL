#!/usr/bin/env python3
"""
API Health Check for CI/CD Pipeline.

This script validates that all API components are working correctly
and can be imported without errors.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_imports():
    """Test that all API modules can be imported successfully."""
    try:
        logger.info("🔍 Testing API imports...")

        # Test core API imports
        from models.emotion_detection.api_demo import EmotionRequest, EmotionResponse
        from models.summarization.api_demo import SummarizeRequest, SummarizationResponse
        from api_rate_limiter import add_rate_limiting

        logger.info("✅ All API imports successful")
        return True

    except Exception as e:
        logger.error(f"❌ API import test failed: {e}")
        return False


def test_api_models():
    """Test that API models can be instantiated."""
    try:
        logger.info("🤖 Testing API model instantiation...")

        # Test emotion analysis request model
        from models.emotion_detection.api_demo import EmotionRequest

        emotion_request = EmotionRequest(text="I feel happy and excited today!", threshold=0.2)

        logger.info(f"✅ Emotion request created: {emotion_request.text[:30]}...")

        # Test summarization request model
        from models.summarization.api_demo import SummarizeRequest

        summary_request = SummarizeRequest(
            text="This is a test text for summarization.", max_length=50, min_length=10
        )

        logger.info(f"✅ Summary request created: {summary_request.text[:30]}...")

        return True

    except Exception as e:
        logger.error(f"❌ API model test failed: {e}")
        return False


def test_api_validation():
    """Test API request validation."""
    try:
        logger.info("🔒 Testing API validation...")

        from models.emotion_detection.api_demo import EmotionRequest
        from pydantic import ValidationError

        # Test invalid emotion request
        try:
            EmotionRequest(text="", threshold=2.0)  # Invalid: empty text, threshold > 1
            logger.error("❌ Validation should have failed for invalid request")
            return False
        except ValidationError:
            logger.info("✅ Validation correctly rejected invalid emotion request")

        # Test valid emotion request
        EmotionRequest(text="This is a valid test text.", threshold=0.3)
        logger.info("✅ Valid emotion request accepted")

        return True

    except Exception as e:
        logger.error(f"❌ API validation test failed: {e}")
        return False


def main():
    """Run all API health checks."""
    logger.info("🚀 Starting API Health Check...")

    tests = [
        ("API Imports", test_api_imports),
        ("API Models", test_api_models),
        ("API Validation", test_api_validation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")

    logger.info(f"\n{'='*50}")
    logger.info(f"API Health Check Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info("🎉 All API health checks passed!")
        return True
    else:
        logger.error("💥 Some API health checks failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
