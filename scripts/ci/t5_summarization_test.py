#!/usr/bin/env python3
"""T5 Summarization Model Test for CircleCI.

This script tests that the T5 summarization model can be loaded
and performs basic text summarization without errors.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from models.summarization.t5_summarizer import T5TextSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_t5_model_loading():
    """Test T5 model initialization and basic summarization."""
    try:
        logger.info("📝 Testing T5 summarization model loading...")
        
        # Initialize model with smaller model for CI
        model = T5TextSummarizer(
            model_name="t5-small",  # Use smaller model for CI
            device="cpu"  # Use CPU for CI
        )
        
        logger.info(f"✅ T5 model initialized successfully")
        
        # Test summarization with simple text
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on the development 
        of algorithms and statistical models that enable computers to improve their performance 
        on a specific task through experience. It involves training models on data to make 
        predictions or decisions without being explicitly programmed for every scenario.
        """
        
        summary = model.summarize(
            text=test_text.strip(),
            max_length=50,
            min_length=10
        )
        
        # Verify summary is generated
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary.strip()) > 0, "Summary should not be empty"
        assert len(summary) < len(test_text), "Summary should be shorter than original text"
        
        logger.info(f"✅ Summarization successful")
        logger.info(f"Original length: {len(test_text)} chars")
        logger.info(f"Summary length: {len(summary)} chars")
        logger.info(f"Summary: {summary[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ T5 model test failed: {e}")
        return False


def main():
    """Run T5 model tests."""
    logger.info("🧪 Starting T5 Summarization Model Tests")
    
    success = test_t5_model_loading()
    
    if success:
        logger.info("🎉 All T5 model tests passed!")
        return 0
    else:
        logger.error("❌ T5 model tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 