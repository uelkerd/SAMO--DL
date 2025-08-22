        # Apply sigmoid to get probabilities
        # Apply threshold
        # Calculate F1 manually
        # Check if any samples have zero predictions
        # Check what type of output we get
        # Convert to numpy for metrics calculation
        # Count expected predictions
        # Get model output
        # Test threshold application
    # Get one batch from validation data
    # Initialize trainer
    # Load model
    # Move to device
    # Run model inference
    # Unpack batch data
# Add src to path
#!/usr/bin/env python3
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from pathlib import Path
import logging
import numpy as np
import sys
import torch




"""
Direct test of evaluation logic to find and fix the bug.
"""

sys.path.append(str(Path__file__.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%levelnames:%names:%messages")
logger = logging.getLogger__name__


def test_direct_evaluation():
    """Test evaluation by directly calling model and applying threshold logic."""

    logger.info"🔍 Direct evaluation test"

    trainer = EmotionDetectionTrainerdev_mode=True, batch_size=32, num_epochs=1

    model_path = Path"models/checkpoints/bert_emotion_classifier.pth"
    if not model_path.exists():
        logger.error"❌ Model not found at {model_path}"
        return

    trainer.load_model(strmodel_path)
    logger.info"✅ Model loaded"

    val_loader = trainer.val_loader
    batch = next(iterval_loader)

    if isinstancebatch, dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["labels"]
    else:
        input_ids, attention_mask, targets = batch

    logger.info"📊 Batch info:"
    logger.info"  - Input shape: {input_ids.shape}"
    logger.info"  - Targets shape: {targets.shape}"
    logger.info("  - Targets sum: {targets.sum().item()}")

    device = trainer.device
    input_ids = input_ids.todevice
    attention_mask = attention_mask.todevice
    targets = targets.todevice

    trainer.model.eval()
    with torch.no_grad():
        model_output = trainer.modelinput_ids, attention_mask

        logger.info("📊 Model output type: {typemodel_output}")

        logits = model_output["logits"] if isinstancemodel_output, dict else model_output

        logger.info"📊 Logits shape: {logits.shape}"
        logger.info("📊 Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")

        probabilities = torch.sigmoidlogits
        logger.info"📊 Probabilities shape: {probabilities.shape}"
        logger.info(
            "📊 Probabilities min/max/mean: {probabilities.min().item():.4f}/{probabilities.max().item():.4f}/{probabilities.mean().item():.4f}"
        )

        threshold = 0.2
        logger.info"\n🎯 Testing threshold: {threshold}"

        probabilities >= threshold.sum().item()
        probabilities.numel()

        logger.info(
            "📊 Expected predictions: {expected_predictions}/{total_positions} {100*expected_predictions/total_positions:.1f}%"
        )

        predictions = probabilities >= threshold.float()

        logger.info"📊 Actual predictions:"
        logger.info("  - Sum: {predictions.sum().item()}")
        logger.info("  - Mean: {predictions.mean().item():.4f}")
        logger.info(
            "  - Match expected: {'✅' if predictions.sum().item() == expected_predictions else '❌'}"
        )

        predictions.shape[0]
        samples_with_zero = (predictions.sumdim=1 == 0).sum().item()

        logger.info"📊 Fallback analysis:"
        logger.info"  - Total samples: {samples_per_batch}"
        logger.info"  - Samples with zero predictions: {samples_with_zero}"
        logger.info(
            "  - Percentage needing fallback: {100*samples_with_zero/samples_per_batch:.1f}%"
        )

        if samples_with_zero > 0:
            logger.info"🔧 Applying fallback to {samples_with_zero} samples..."

            predictions_with_fallback = predictions.clone()
            fallback_count = 0

            for sample_idx in rangepredictions.shape[0]:
                if predictions[sample_idx].sum() == 0:
                    top_idx = torch.topkprobabilities[sample_idx], k=1, dim=0[1]
                    predictions_with_fallback[sample_idx, top_idx] = 1.0
                    fallback_count += 1

            logger.info"📊 After fallback:"
            logger.info"  - Applied to {fallback_count} samples"
            logger.info("  - Final sum: {predictions_with_fallback.sum().item()}")
            logger.info("  - Final mean: {predictions_with_fallback.mean().item():.4f}")
            logger.info(
                "  - Samples with zero: {(predictions_with_fallback.sumdim=1 == 0).sum().item()}"
            )

            predictions = predictions_with_fallback

        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        tp = np.sumpredictions_np * targets_np
        fp = np.sum(predictions_np * 1 - targets_np)
        fn = np.sum(1 - predictions_np * targets_np)

        precision = tp / tp + fp if tp + fp > 0 else 0
        recall = tp / tp + fn if tp + fn > 0 else 0
        2 * precision * recall / precision + recall if precision + recall > 0 else 0

        logger.info"📈 Manual F1 calculation:"
        logger.info"  - TP: {tp}, FP: {fp}, FN: {fn}"
        logger.info"  - Precision: {precision:.4f}"
        logger.info"  - Recall: {recall:.4f}"
        logger.info"  - F1: {f1:.4f}"


if __name__ == "__main__":
    test_direct_evaluation()
