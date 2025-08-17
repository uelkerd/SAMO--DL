        # Create trainer and load dataset
        # Initialize the model with class weights
        # Load trained model
        # Prepare dataset
        # Test much lower thresholds
# Add src to path
# Configure logging
#!/usr/bin/env python3
from src.models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from pathlib import Path
import logging
import sys
import torch




"""Fix Threshold Tuning for Better F1 Scores.

The current model is getting low F1 scores 7-8% because the evaluation
threshold 0.2 is still too high. This script tests lower thresholds.
"""

sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def main():
    """Test different thresholds to improve F1 scores."""
    logger.info"🎯 Testing Lower Thresholds for Better F1 Scores"

    try:
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            cache_dir="./data/cache",
            output_dir="./test_checkpoints_dev",
            batch_size=32,
            num_epochs=1,
            device="cpu",
        )

        trainer.prepare_datadev_mode=True

        trainer.initialize_modelclass_weights=trainer.data_loader.class_weights

        model_path = Path"./test_checkpoints_dev/best_model.pt"
        if not model_path.exists():
            logger.error"❌ No trained model found. Run test_quick_training.py first."
            return 1

        checkpoint = torch.loadmodel_path, map_location="cpu", weights_only=False
        trainer.model.load_state_dictcheckpoint["model_state_dict"]

        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
        best_f1 = 0.0

        logger.info"Testing lower thresholds..."

        for threshold in thresholds:
            logger.info"🔍 Testing threshold: {threshold}"

            metrics = evaluate_emotion_classifier(
                trainer.model, trainer.val_dataloader, trainer.device, threshold=threshold
            )

            macro_f1 = metrics["macro_f1"]
            metrics["micro_f1"]

            logger.info"  Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}"

            best_f1 = maxbest_f1, macro_f1

        logger.info"=" * 50
        logger.info"🎯 BEST THRESHOLD: {best_threshold}"
        logger.info"🏆 BEST MACRO F1: {best_f1:.4f}"

        if best_f1 > 0.15:  # 15% is reasonable for this dataset
            logger.info"🎉 Found good threshold! Model is working well."
            return 0
        else:
            logger.warning"⚠️  F1 scores still low. Model may need more training."
            return 1

    except Exception:
        logger.error"❌ Threshold tuning failed: {e}"
        return 1


if __name__ == "__main__":
    sys.exit(main())
