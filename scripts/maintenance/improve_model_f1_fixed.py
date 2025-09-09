#!/usr/bin/env python3
"""
Improved Model F1 Fixed Script

This script implements focal loss training for emotion detection models.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from typing import Optional
import argparse
import logging
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5


def create_focal_loss(alpha: float = 1.0, gamma: float = 2.0):
    """Create focal loss function for handling class imbalance."""
    def focal_loss_fn(inputs, targets):
        """Compute focal loss for handling class imbalance in multi-class classification.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    return focal_loss_fn


def improve_with_focal_loss(
    model_path: str,
    output_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    alpha: float = 1.0,
    gamma: float = 2.0
) -> dict:
    """
    Improve model performance using focal loss.

    Args:
        model_path: Path to the base model
        output_path: Path to save the improved model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter

    Returns:
        Dictionary containing training results
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Create data loader
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.load_data()

        # Create model with optimal settings
        _ = create_bert_emotion_classifier()

        # Create focal loss function
        _ = create_focal_loss(alpha=alpha, gamma=gamma)

        # Create trainer with development mode disabled for better results
        trainer = EmotionDetectionTrainer(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            early_stopping_patience=3,
        )

        logger.info("Training model for %s epochs with batch_size=%s", epochs, batch_size)
        trainer.train()

        metrics = trainer.evaluate(datasets["test"])

        logger.info(
            "Fresh model results - Micro F1: %.4f, Macro F1: %.4f",
            metrics['micro_f1'], metrics['macro_f1']
        )

        # Save model
        trainer.save_model(output_path)
        logger.info("Model saved to: %s", output_path)

        return {
            "success": True,
            "metrics": metrics,
            "model_path": output_path
        }

    except Exception as e:
        logger.error("Error in focal loss improvement: %s", e)
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main function to run the focal loss improvement."""
    parser = argparse.ArgumentParser(description="Improve model F1 with focal loss")
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--output_path", required=True, help="Path to save improved model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Focal loss alpha")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")

    args = parser.parse_args()

    logger.info("Starting focal loss improvement...")
    logger.info("Model path: %s", args.model_path)
    logger.info("Output path: %s", args.output_path)

    result = improve_with_focal_loss(
        model_path=args.model_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        gamma=args.gamma
    )

    if result["success"]:
        logger.info("✅ Focal loss improvement completed successfully!")
        logger.info("Final metrics: %s", result["metrics"])
    else:
        logger.error("❌ Focal loss improvement failed: %s", result["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
