                import torch
            # Try to read the file
        # Check specific file
        import torch
    # Check current directory
    # Check if test_checkpoints directory exists
    # Check if we can import the required modules
    # If needed for future debugging, import them when actually used
    # Note: transformers and sklearn imports removed as they were unused
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
import logging



"""
Debug Calibration Script

This script helps debug the calibration test by checking file paths and permissions.
"""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def debug_calibration_issue():
    """Debug the calibration test issue."""
    logger.info("🔍 Debugging calibration test issue...")

    logger.info("Current directory: {Path.cwd()}")

    checkpoint_dir = Path("test_checkpoints")
    logger.info("test_checkpoints directory exists: {checkpoint_dir.exists()}")

    if checkpoint_dir.exists():
        logger.info("test_checkpoints contents: {list(checkpoint_dir.iterdir())}")

        checkpoint_file = checkpoint_dir / "best_model.pt"
        logger.info("best_model.pt exists: {checkpoint_file.exists()}")

        if checkpoint_file.exists():
            logger.info("File size: {checkpoint_file.stat().st_size} bytes")
            logger.info("File permissions: {oct(checkpoint_file.stat().st_mode)}")

            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                logger.info("✅ Checkpoint loaded successfully")
                logger.info(
                    "Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}"
                )
            except Exception as e:
                logger.error("❌ Failed to load checkpoint: {e}")
        else:
            logger.error("❌ best_model.pt does not exist")
    else:
        logger.error("❌ test_checkpoints directory does not exist")

    try:
        logger.info("✅ PyTorch imported successfully")
    except ImportError as _:
        logger.error("❌ PyTorch import failed: {e}")



if __name__ == "__main__":
    debug_calibration_issue()
