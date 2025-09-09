#!/usr/bin/env python3
"""
Local Validation Debug Script

This script debugs local validation issues.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_validation():
    """Debug validation issues."""
    try:
        logger.info("🔍 Starting local validation debug...")
        
        # Test imports
        logger.info("Testing imports...")
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader
        
        # Test data loading
        logger.info("Testing data loading...")
        data_loader = create_goemotions_loader()
        _datasets = data_loader.load_data()

        # Test model creation
        logger.info("Testing model creation...")
        _model = create_bert_emotion_classifier()
        
        logger.info("✅ All validation tests passed!")
        return True
        
    except Exception as e:
        logger.error("❌ Validation debug failed: %s", e)
        return False


def main():
    """Main function."""
    if debug_validation():
        logger.info("🎉 Local validation debug completed!")
    else:
        logger.error("💥 Local validation debug failed!")


if __name__ == "__main__":
    main()