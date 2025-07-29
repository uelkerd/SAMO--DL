#!/usr/bin/env python3
"""
Debug Calibration Script

This script helps debug the calibration test by checking file paths and permissions.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def debug_calibration_issue():
    """Debug the calibration test issue."""
    logger.info("🔍 Debugging calibration test issue...")

    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")

    # Check if test_checkpoints directory exists
    checkpoint_dir = Path("test_checkpoints")
    logger.info(f"test_checkpoints directory exists: {checkpoint_dir.exists()}")

    if checkpoint_dir.exists():
        logger.info(f"test_checkpoints contents: {list(checkpoint_dir.iterdir())}")

        # Check specific file
        checkpoint_file = checkpoint_dir / "best_model.pt"
        logger.info(f"best_model.pt exists: {checkpoint_file.exists()}")

        if checkpoint_file.exists():
            logger.info(f"File size: {checkpoint_file.stat().st_size} bytes")
            logger.info(f"File permissions: {oct(checkpoint_file.stat().st_mode)}")

            # Try to read the file
            try:
                import torch

                checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                logger.info("✅ Checkpoint loaded successfully")
                logger.info(
                    f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
        else:
            logger.error("❌ best_model.pt does not exist")
    else:
        logger.error("❌ test_checkpoints directory does not exist")

    # Check if we can import the required modules
    try:
        import torch

        logger.info("✅ PyTorch imported successfully")
    except ImportError as e:
        logger.error(f"❌ PyTorch import failed: {e}")

    try:
        from transformers import AutoTokenizer

        logger.info("✅ Transformers imported successfully")
    except ImportError as e:
        logger.error(f"❌ Transformers import failed: {e}")

    try:
        from sklearn.metrics import f1_score

        logger.info("✅ Scikit-learn imported successfully")
    except ImportError as e:
        logger.error(f"❌ Scikit-learn import failed: {e}")


if __name__ == "__main__":
    debug_calibration_issue()
