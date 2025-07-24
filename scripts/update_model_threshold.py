#!/usr/bin/env python3
"""
Update Model Threshold

This script updates the prediction threshold for the BERT emotion classifier
based on the optimal value determined through calibration.

Usage:
    python scripts/update_model_threshold.py [--threshold THRESHOLD]

Arguments:
    --threshold: Optional threshold value (default: 0.6)
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
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
        logger.error(f"Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
        return False

    # Find an existing model file
    model_path = None
    for path in MODEL_PATHS:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        logger.error("No model file found. Please train a model first.")
        return False

    logger.info(f"Loading model from {model_path}...")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Create model
        model, _ = create_bert_emotion_classifier()

        # Load state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            logger.error(f"Unexpected checkpoint format: {type(checkpoint)}")
            return False

        # Update threshold
        logger.info(
            f"Updating prediction threshold from {model.prediction_threshold} to {threshold}"
        )
        model.prediction_threshold = threshold

        # Set temperature
        model.set_temperature(DEFAULT_TEMPERATURE)

        # Save model
        logger.info(f"Saving updated model to {model_path}")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint["model_state_dict"] = model.state_dict()
            torch.save(checkpoint, model_path)
        else:
            torch.save(model.state_dict(), model_path)

        logger.info(f"✅ Model threshold updated successfully to {threshold}")
        return True

    except Exception as e:
        logger.error(f"Error updating model threshold: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model prediction threshold")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"New threshold value (0.0-1.0, default: {DEFAULT_THRESHOLD})",
    )

    args = parser.parse_args()
    success = update_threshold(args.threshold)
    sys.exit(0 if success else 1)
