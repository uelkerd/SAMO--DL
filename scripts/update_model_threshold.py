import os
import sys
#!/usr/bin/env python3
import argparse
import torch
import logging
from pathlib import Path
# Add src to path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
# Configure logging
# Constants
    # Find an existing model file
        # Load checkpoint
        # Create model
        # Load state dict
        # Update threshold
        # Set temperature
        # Save model




"""
Update Model Threshold

This script updates the prediction threshold for the BERT emotion classifier
based on the optimal value determined through calibration.

Usage:
    python scripts/update_model_threshold.py [--threshold THRESHOLD]

Arguments:
    --threshold: Optional threshold value (default: 0.6)
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6
DEFAULT_TEMPERATURE = 1.0
MODEL_PATHS = ["models/checkpoints/bert_emotion_classifier.pth", "test_checkpoints/best_model.pt"]


def update_threshold(threshold: float = DEFAULT_THRESHOLD):
    """Update the model's prediction threshold.

    Args:
        threshold: New threshold value (0.0-1.0)

    Returns:
        bool: True if successful, False otherwise
    """
    if threshold < 0.0 or threshold > 1.0:
        logger.error("Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
        return False

    model_path = None
    for __path in MODEL_PATHS:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        logger.error("No model file found. Please train a model first.")
        return False

    logger.info("Loading model from {model_path}...")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        model, _ = create_bert_emotion_classifier()

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            logger.error("Unexpected checkpoint format: {type(checkpoint)}")
            return False

        logger.info(
            "Updating prediction threshold from {model.prediction_threshold} to {threshold}"
        )
        model.prediction_threshold = threshold

        model.set_temperature(DEFAULT_TEMPERATURE)

        logger.info("Saving updated model to {model_path}")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint["model_state_dict"] = model.state_dict()
            torch.save(checkpoint, model_path)
        else:
            torch.save(model.state_dict(), model_path)

        logger.info("✅ Model threshold updated successfully to {threshold}")
        return True

    except Exception as _:
        logger.error("Error updating model threshold: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model prediction threshold")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="New threshold value (0.0-1.0, default: {DEFAULT_THRESHOLD})",
    )

    args = parser.parse_args()
    success = update_threshold(args.threshold)
    sys.exit(0 if success else 1)
