#!/usr/bin/env python3
"""
Full Focal Loss Training Script

This script provides a complete focal loss training implementation
for the emotion detection model.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass of focal loss."""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def create_training_data():
    """Create training data for testing."""
    logger.info("Creating training data...")

    texts = [
        "I am feeling happy today!",
        "This makes me sad.",
        "I'm really angry about this.",
        "I'm scared of what might happen.",
        "I feel great about everything!",
        "This is disappointing.",
        "I'm furious with you!",
        "I'm terrified of the dark.",
    ]

    labels = [
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
    ]

    return texts, labels


def full_focal_training():
    """Run full focal loss training."""
    logger.info("🚀 Starting Full Focal Loss Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model and tokenizer
    model, tokenizer = create_bert_emotion_classifier()
    model.to(device)

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

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"], inputs["attention_mask"], labels_tensor
    )
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    focal_loss = FocalLoss(gamma=2.0)

    # Training loop
    model.train()
    for epoch in range(3):
        logger.info(f"📚 Epoch {epoch + 1}/3")
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = focal_loss(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

    logger.info("✅ Full focal loss training completed!")


if __name__ == "__main__":
    full_focal_training()
