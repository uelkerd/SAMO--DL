#!/usr/bin/env python3
"""Debug Dataset Structure Script.

This script helps understand the structure of the GoEmotions dataset.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def debug_dataset_structure():
    """Debug the structure of the GoEmotions dataset."""
    logger.info("🔍 Debugging Dataset Structure")
    logger.info("=" * 50)

    try:
        # Load dataset
        logger.info("📊 Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        data_loader.download_dataset()
        datasets = data_loader.prepare_datasets()

        logger.info("📋 Dataset keys:")
        for key in datasets.keys():
            logger.info(f"   - {key}")

        # Check test data structure
        test_data = datasets["test_data"]
        logger.info(f"📊 Test data type: {type(test_data)}")
        logger.info(f"📊 Test data length: {len(test_data)}")

        if len(test_data) > 0:
            first_item = test_data[0]
            logger.info(f"📊 First item type: {type(first_item)}")
            logger.info(f"📊 First item: {first_item}")

            if hasattr(first_item, 'keys'):
                logger.info(f"📊 First item keys: {list(first_item.keys())}")
            elif hasattr(first_item, '__dict__'):
                logger.info(f"📊 First item attributes: {list(first_item.__dict__.keys())}")

        # Check train data structure
        train_data = datasets["train_data"]
        logger.info(f"📊 Train data type: {type(train_data)}")
        logger.info(f"📊 Train data length: {len(train_data)}")

        if len(train_data) > 0:
            first_train_item = train_data[0]
            logger.info(f"📊 First train item type: {type(first_train_item)}")
            logger.info(f"📊 First train item: {first_train_item}")

        # Check if it's a HuggingFace dataset
        if hasattr(test_data, 'features'):
            logger.info(f"📊 Dataset features: {test_data.features}")

        if hasattr(test_data, 'column_names'):
            logger.info(f"📊 Dataset columns: {test_data.column_names}")

        return True

    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_dataset_structure()
    if success:
        logger.info("✅ Debug completed successfully")
    else:
        logger.error("❌ Debug failed")
        sys.exit(1)
