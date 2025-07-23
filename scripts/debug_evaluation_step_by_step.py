#!/usr/bin/env python3
"""
Debug the evaluation function step by step to find the exact issue.
"""

import logging
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def debug_evaluation_step_by_step():
    """Debug the evaluation function with detailed step-by-step analysis."""

    logger.info("🔍 Step-by-step evaluation debugging")

    # Initialize trainer
    trainer = EmotionDetectionTrainer(dev_mode=True, batch_size=128, num_epochs=1)

    logger.info("✅ Trainer initialized")

    # Load model
    model_path = Path("models/checkpoints/bert_emotion_classifier.pth")
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        return

    trainer.load_model(str(model_path))
    logger.info("✅ Model loaded")

    # Get validation data (small batch for debugging)
    val_loader = trainer.val_loader

    # Take just one batch for detailed analysis
    batch = next(iter(val_loader))
    input_ids, attention_mask, targets = batch

    logger.info("🔍 Analyzing single batch:")
    logger.info(f"  📊 Batch size: {input_ids.shape[0]}")
    logger.info(f"  📊 Sequence length: {input_ids.shape[1]}")
    logger.info(f"  📊 Number of emotions: {targets.shape[1]}")
    logger.info(f"  📊 Target sum: {targets.sum().item()}")
    logger.info(f"  📊 Target mean: {targets.mean().item():.4f}")

    # Run model inference
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(input_ids, attention_mask)
        logits = outputs["logits"]
        probabilities = torch.sigmoid(logits)

    logger.info("🔍 Model outputs:")
    logger.info(f"  📊 Logits shape: {logits.shape}")
    logger.info(f"  📊 Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
    logger.info(f"  📊 Probabilities shape: {probabilities.shape}")
    logger.info(
        f"  📊 Probabilities min/max: {probabilities.min().item():.4f} / {probabilities.max().item():.4f}"
    )
    logger.info(f"  📊 Probabilities mean: {probabilities.mean().item():.4f}")

    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.5]

    for threshold in thresholds:
        logger.info(f"\n🎯 Testing threshold: {threshold}")

        # Apply threshold
        predictions_before_fallback = (probabilities >= threshold).float()
        logger.info("  📊 Predictions before fallback:")
        logger.info(f"    - Shape: {predictions_before_fallback.shape}")
        logger.info(f"    - Sum: {predictions_before_fallback.sum().item()}")
        logger.info(f"    - Mean: {predictions_before_fallback.mean().item():.4f}")
        logger.info(
            f"    - Samples with 0 predictions: {(predictions_before_fallback.sum(dim=1) == 0).sum().item()}"
        )
        logger.info(
            f"    - Samples with >0 predictions: {(predictions_before_fallback.sum(dim=1) > 0).sum().item()}"
        )

        # Check which samples need fallback
        samples_needing_fallback = predictions_before_fallback.sum(dim=1) == 0
        num_samples_needing_fallback = samples_needing_fallback.sum().item()

        logger.info("  🔧 Fallback analysis:")
        logger.info(f"    - Samples needing fallback: {num_samples_needing_fallback}")
        logger.info(
            f"    - Percentage needing fallback: {100 * num_samples_needing_fallback / predictions_before_fallback.shape[0]:.1f}%"
        )

        # Apply fallback manually to see what happens
        predictions_after_fallback = predictions_before_fallback.clone()

        if num_samples_needing_fallback > 0:
            logger.info(f"  🔧 Applying fallback to {num_samples_needing_fallback} samples...")

            for sample_idx in range(predictions_after_fallback.shape[0]):
                if predictions_after_fallback[sample_idx].sum() == 0:
                    # Find top-1 prediction
                    top_idx = torch.topk(probabilities[sample_idx], k=1, dim=0)[1]
                    predictions_after_fallback[sample_idx, top_idx] = 1.0
                    logger.info(
                        f"    - Sample {sample_idx}: Applied fallback to emotion {top_idx.item()}"
                    )

        logger.info("  📊 Predictions after fallback:")
        logger.info(f"    - Sum: {predictions_after_fallback.sum().item()}")
        logger.info(f"    - Mean: {predictions_after_fallback.mean().item():.4f}")
        logger.info(
            f"    - Samples with 0 predictions: {(predictions_after_fallback.sum(dim=1) == 0).sum().item()}"
        )

        # Calculate F1 scores manually
        predictions_np = predictions_after_fallback.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Micro F1
        tp = np.sum(predictions_np * targets_np)
        fp = np.sum(predictions_np * (1 - targets_np))
        fn = np.sum((1 - predictions_np) * targets_np)

        micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )

        logger.info("  📈 Manual F1 calculation:")
        logger.info(f"    - TP: {tp}, FP: {fp}, FN: {fn}")
        logger.info(f"    - Micro Precision: {micro_precision:.4f}")
        logger.info(f"    - Micro Recall: {micro_recall:.4f}")
        logger.info(f"    - Micro F1: {micro_f1:.4f}")


if __name__ == "__main__":
    debug_evaluation_step_by_step()
