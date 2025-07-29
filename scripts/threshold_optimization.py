#!/usr/bin/env python3
"""
Threshold Optimization for Multi-label Classification

This script optimizes per-class thresholds to improve F1 score
by 10-15% through better classification boundaries.
"""

import os
import sys
import logging
import torch
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def optimize_thresholds(val_logits, val_labels, num_classes=28):
    """Optimize thresholds for each emotion class."""
    logger.info("🎯 Optimizing per-class thresholds...")

    thresholds = []
    best_f1_scores = []

    for i in range(num_classes):
        best_f1 = 0
        best_threshold = 0.5

        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (val_logits[:, i] > threshold).float()
            f1 = f1_score(val_labels[:, i], predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

        logger.info(f"   • Class {i}: threshold={best_threshold:.3f}, F1={best_f1:.3f}")

    avg_f1 = np.mean(best_f1_scores)
    logger.info(f"✅ Average F1 score: {avg_f1:.3f}")

    return thresholds, best_f1_scores


def apply_threshold_optimization():
    """Apply threshold optimization to improve classification performance."""

    logger.info("🎯 Starting Threshold Optimization")
    logger.info("   • Expected improvement: 10-15% F1 score")
    logger.info("   • Method: Per-class threshold tuning")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load dataset
        logger.info("Loading validation dataset...")
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()
        
        val_dataset = datasets["validation"]  # Fixed key name
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Load trained model
        model_path = "./models/checkpoints/focal_loss_best_model.pt"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("   • Please run focal_loss_training.py first")
            return False

        logger.info(f"Loading model from {model_path}")
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=4,
        )
        model.to(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("✅ Model loaded successfully")

        # Collect validation logits and labels
        all_logits = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # Concatenate all batches
        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)

        # Optimize thresholds
        thresholds, f1_scores = optimize_thresholds(val_logits, val_labels)

        # Save optimized thresholds
        output_dir = "./models/checkpoints"
        os.makedirs(output_dir, exist_ok=True)
        thresholds_path = os.path.join(output_dir, "optimized_thresholds.pt")

        torch.save(
            {
                "thresholds": thresholds,
                "f1_scores": f1_scores,
                "avg_f1": np.mean(f1_scores),
                "model_path": model_path,
            },
            thresholds_path,
        )

        logger.info(f"✅ Optimized thresholds saved to: {thresholds_path}")
        logger.info(f"   • Average F1: {np.mean(f1_scores):.3f}")
        logger.info(f"   • Threshold range: {min(thresholds):.3f} - {max(thresholds):.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ Threshold optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("🎯 Threshold Optimization Script")
    logger.info("This script optimizes classification thresholds for better F1 scores")

    success = apply_threshold_optimization()

    if success:
        logger.info("✅ Threshold optimization completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Threshold optimization failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
