#!/usr/bin/env python3
"""
T5 Summarization Model Test for CI/CD Pipeline.

This script validates that the T5 text summarization model
can be loaded and initialized correctly.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.summarization.t5_summarizer import create_t5_summarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_t5_model_loading():
    """Test T5 model initialization."""
    try:
        logger.info("🤖 Testing T5 summarization model loading...")

        # Initialize model with CPU device for CI
        model = create_t5_summarizer(
            model_name="t5-small",  # Use small model for CI
            device="cpu",
        )

        logger.info("✅ T5 model initialized successfully")

        # Test basic model properties
        assert hasattr(model, "model"), "Model should have 'model' attribute"
        assert hasattr(model, "tokenizer"), "Model should have 'tokenizer' attribute"
        assert hasattr(model, "device"), "Model should have 'device' attribute"

        logger.info("✅ Model attributes validation passed")

        return True

    except Exception as e:
        logger.error(f"❌ T5 model loading failed: {e}")
        return False


def test_t5_summarization():
    """Test T5 summarization functionality."""
    try:
        logger.info("📝 Testing T5 summarization functionality...")

        # Initialize model
        model = create_t5_summarizer(model_name="t5-small", device="cpu")

        # Test text for summarization
        test_text = """
        The T5 (Text-To-Text Transfer Transformer) model is a transformer-based
        architecture that treats every NLP problem as a text-to-text problem.
        It was introduced by Google Research and has shown excellent performance
        across various natural language processing tasks including summarization,
        translation, and question answering.
        """

        # Perform summarization
        summary = model.generate_summary(
            text=test_text.strip(),
            max_length=50,
            min_length=10
        )

        logger.info(f"✅ Summarization successful: {summary[:50]}...")

        # Validate summary
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 0, "Summary should not be empty"
        assert len(summary) < len(test_text), "Summary should be shorter than input"

        logger.info("✅ Summary validation passed")

        return True

    except Exception as e:
        logger.error(f"❌ T5 summarization test failed: {e}")
        return False


def main():
    """Run T5 model tests."""
    logger.info("🚀 Starting T5 Model Tests...")

    tests = [
        ("T5 Model Loading", test_t5_model_loading),
        ("T5 Summarization", test_t5_summarization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*40}")

        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")

    logger.info(f"\n{'='*40}")
    logger.info(f"T5 Tests Results: {passed}/{total} tests passed")
    logger.info(f"{'='*40}")

    if passed == total:
        logger.info("🎉 All T5 model tests passed!")
        return True
    else:
        logger.error("💥 Some T5 model tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
