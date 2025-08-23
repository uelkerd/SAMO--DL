#!/usr/bin/env python3
"""Test script to verify numpy compatibility fix for transformers."""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_numpy_compatibility():
    """Test numpy compatibility with transformers."""
    logger.info("🧪 Testing numpy compatibility...")

    try:
        # Test 1: Basic numpy import
        import numpy as np

        logger.info(f"✅ Numpy version: {np.__version__}")

        # Test 2: Check for broadcast_to function
        if hasattr(np.lib.stride_tricks, "broadcast_to"):
            logger.info("✅ broadcast_to function exists")
        else:
            logger.warning("⚠️ broadcast_to function missing, applying fix...")

            def broadcast_to(array, shape):
                return np.broadcast_arrays(array, np.empty(shape))[0]

            np.lib.stride_tricks.broadcast_to = broadcast_to
            logger.info("✅ broadcast_to function added")

        # Test 3: Test transformers import
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info("✅ Transformers import successful")
        except ImportError as e:
            if "broadcast_to" in str(e):
                logger.error("❌ Still getting broadcast_to error after fix")
                return False
            else:
                logger.error(f"❌ Other transformers import error: {e}")
                return False

        # Test 4: Test basic transformers functionality
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            logger.info("✅ Tokenizer loading successful")
        except Exception as e:
            logger.error(f"❌ Tokenizer loading failed: {e}")
            return False

        logger.info("🎉 All numpy compatibility tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_numpy_compatibility()
    if not success:
        sys.exit(1)
