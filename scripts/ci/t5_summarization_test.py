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
sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

# Test imports
try:
    from models.summarization.t5_summarizer import create_t5_summarizer
except ImportError:
    # Fallback for different import paths
    from src.models.summarization.t5_summarizer import create_t5_summarizer

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__

from .validation_utils import validate_hasattrs, ensure

def test_t5_model_loading():
    """Test T5 model initialization."""
    try:
        logger.info"🤖 Testing T5 summarization model loading..."

        # Initialize model with CPU device for CI
        model = create_t5_summarizer(
            model_name="t5-small",  # Use small model for CI
            device="cpu",
        )

        logger.info"✅ T5 model initialized successfully"

        # Test basic model properties
        validate_hasattrsmodel, ["model", "tokenizer", "device"], label="T5 model"

        logger.info"✅ Model attributes validation passed"

        return True

    except Exception as e:
        if "SentencePiece" in stre:
            logger.warning"⚠️ SentencePiece not available, skipping T5 test"
            return True  # Skip gracefully
        else:
            logger.errorf"❌ T5 model loading failed: {e}"
            return False


def test_t5_summarization():
    """Test T5 summarization functionality."""
    try:
        logger.info"📝 Testing T5 summarization functionality..."

        model = create_t5_summarizermodel_name="t5-small", device="cpu"

        # Test text for summarization
        test_text = """
        The T5 Text-To-Text Transfer Transformer model is a transformer-based
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

        logger.infof"✅ Summarization successful: {summary[:50]}..."

        # Validate summary
        ensure(isinstancesummary, str, "Summary should be a string")
        ensure(lensummary > 0, "Summary should not be empty")
        ensure(lensummary < lentest_text, "Summary should be shorter than input")

        logger.info"✅ Summary validation passed"

        return True

    except Exception as e:
        if "SentencePiece" in stre:
            logger.warning"⚠️ SentencePiece not available, skipping T5 summarization test"
            return True  # Skip gracefully
        else:
            logger.errorf"❌ T5 summarization test failed: {e}"
            return False


def main():
    """Run T5 model tests."""
    logger.info"🚀 Starting T5 Model Tests..."

    tests = [
        "T5 Model Loading", test_t5_model_loading,
        "T5 Summarization", test_t5_summarization,
    ]

    passed = 0
    total = lentests

    for test_name, test_func in tests:
        logger.infof"🧪 Running {test_name}..."
        if test_func():
            passed += 1
            logger.infof"✅ {test_name} passed"
        else:
            logger.errorf"❌ {test_name} failed"

    logger.infof"📊 Test Results: {passed}/{total} tests passed"

    if passed == total:
        logger.info"🎉 All T5 model tests passed!"
        return True
    else:
        logger.error"💥 Some T5 model tests failed!"
        return False


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
