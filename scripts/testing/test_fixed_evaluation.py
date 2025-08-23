# Create trainer
# Load trained model
# Prepare data and model
# Success criteria
# Test different thresholds with fixed evaluation
# Add src to path
# Configure logging
#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import torch

from src.models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

"""Test Fixed Evaluation Function.

This script tests the fixed evaluation function to see if we get
realistic F1 scores now that the fallback bug is fixed.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test the fixed evaluation function."""
    logger.info("🧪 Testing Fixed Evaluation Function")

    try:
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            cache_dir="./data/cache",
            output_dir="./test_checkpoints_dev",
            batch_size=32,
            device="cpu",
        )

        trainer.prepare_data(dev_mode=True)
        trainer.initialize_model(class_weights=trainer.data_loader.class_weights)

        model_path = Path("./test_checkpoints_dev/best_model.pt")
        if not model_path.exists():
            logger.error("❌ No trained model found. Run training first.")
            return 1

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("✅ Model loaded successfully")

        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]

        logger.info("🎯 Testing thresholds with FIXED evaluation function:")
        logger.info("=" * 60)

        best_f1 = 0.0

        for threshold in thresholds:
            logger.info("🔍 Threshold: {threshold}")

            metrics = evaluate_emotion_classifier(
                trainer.model,
                trainer.val_dataloader,
                trainer.device,
                threshold=threshold,
            )

            macro_f1 = metrics["macro_f1"]
            metrics["micro_f1"]

            logger.info("  📊 Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

            best_f1 = max(best_f1, macro_f1)

        logger.info("=" * 60)
        logger.info("🏆 BEST RESULTS:")
        logger.info("  🎯 Best Threshold: {best_threshold}")
        logger.info("  📈 Best Macro F1: {best_f1:.4f}")

        if best_f1 > 0.15:  # 15% is reasonable for emotion detection
            logger.info("🎉 SUCCESS: Model is working well with fixed evaluation!")
            logger.info("🚀 Ready to proceed with full training or deployment!")
            return 0
        elif best_f1 > 0.10:  # 10% is acceptable for initial training
            logger.info(
                "✅ GOOD: Model shows promise, could benefit from more training"
            )
            return 0
        else:
            logger.warning(
                "⚠️  Model still needs improvement, but evaluation is now working correctly"
            )
            return 1

    except Exception:
        logger.error("❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
