        # Start training
        # Training configuration with debugging

        from src.models.emotion_detection.training_pipeline import train_emotion_detection_model
        import traceback

# Add src to path
# Configure logging
#!/usr/bin/env python3

from pathlib import Path
import logging
import sys
import traceback






"""
Restart Training with Debugging Script for SAMO Deep Learning.

This script restarts the emotion detection training with comprehensive debugging
to identify the root cause of the 0.0000 loss issue.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("debug_training.log")],
)
logger = logging.getLogger(__name__)


def main():
    """Restart training with debugging enabled."""
    logger.info("🚀 Restarting training with comprehensive debugging...")

    try:
        config = {
            "model_name": "bert-base-uncased",
            "cache_dir": "./data/cache",
            "output_dir": "./models/emotion_detection",
            "batch_size": 8,  # Smaller batch for debugging
            "learning_rate": 2e-6,  # Reduced learning rate
            "num_epochs": 2,  # Fewer epochs for debugging
            "dev_mode": True,
            "debug_mode": True,
        }

        logger.info("📋 Training Configuration:")
        for key, value in config.items():
            logger.info("   {key}: {value}")

        logger.info("\n🔍 Starting training with debugging...")
        logger.info("⚠️  Watch for DEBUG messages to identify the 0.0000 loss issue!")

        results = train_emotion_detection_model(**config)

        logger.info("✅ Training completed!")
        logger.info("📊 Final results: {results}")

    except Exception as e:
        logger.error("❌ Training failed: {e}")
        logger.error("Traceback: {traceback.format_exc()}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
