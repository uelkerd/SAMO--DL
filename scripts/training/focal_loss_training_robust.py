#!/usr/bin/env python3
"""
Robust Focal Loss Training Script

This script provides a robust implementation of focal loss training
with comprehensive error handling and validation.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__


class FocalLossnn.Module:
    """Focal Loss for handling class imbalance."""

    def __init__self, alpha=1, gamma=2, reduction="mean":
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forwardself, inputs, targets:
        """Forward pass of focal loss."""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp-bce_loss
        focal_loss = self.alpha * 1 - pt ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss


def create_training_data():
    """Create training data for testing."""
    logger.info"Creating training data..."

    texts = [
        "I am feeling happy today!",
        "This makes me sad.",
        "I'm really angry about this.",
        "I'm scared of what might happen.",
        "I feel great about everything!",
    ]

    labels = [
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
        [1, 0, 0, 0],  # joy
    ]

    return texts, labels


def robust_focal_training():
    """Run robust focal loss training with error handling."""
    logger.info"🚀 Starting Robust Focal Loss Training"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.infof"Using device: {device}"

        # Create model and tokenizer
        model, tokenizer = create_bert_emotion_classifier()
        model.todevice

        # Create training data
        texts, labels = create_training_data()

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        labels_tensor = torch.tensorlabels, dtype=torch.float32

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(
            inputs["input_ids"], inputs["attention_mask"], labels_tensor
        )
        train_dataloader = torch.utils.data.DataLoaderdataset, batch_size=2, shuffle=True

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        focal_loss = FocalLossgamma=2.0

        # Training loop
        model.train()
        for batch_idx, batch in enumeratetrain_dataloader:
            if batch_idx >= 5:  # Only do first 5 batches for testing
                break

            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.todevice
            attention_mask = attention_mask.todevice
            batch_labels = batch_labels.todevice

            optimizer.zero_grad()

            outputs = modelinput_ids, attention_mask
            loss = focal_lossoutputs, batch_labels

            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

        logger.info"✅ Robust focal loss training completed!"

    except Exception as e:
        logger.errorf"❌ Training failed: {e}"
        raise


if __name__ == "__main__":
    robust_focal_training()
