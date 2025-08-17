#!/usr/bin/env python3
"""
Model F1 Score Improvement Script

This script focuses on improving the F1 score of the emotion detection model
through various optimization techniques.
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


def create_balanced_training_data():
    """Create balanced training data for F1 improvement."""
    logger.info("Creating balanced training data...")

    # Create diverse emotion examples
    emotion_data = {
        "joy": [
            "I am feeling happy today!",
            "This is wonderful news!",
            "I'm so excited about this!",
            "What a great day!",
            "I love this so much!",
        ],
        "sadness": [
            "I'm feeling down today.",
            "This is really disappointing.",
            "I'm sad about what happened.",
            "This makes me feel blue.",
            "I'm not feeling great.",
        ],
        "anger": [
            "I'm really angry about this!",
            "This makes me furious!",
            "I'm so mad right now!",
            "This is infuriating!",
            "I can't believe this!",
        ],
        "fear": [
            "I'm scared of what might happen.",
            "This is terrifying!",
            "I'm afraid of the consequences.",
            "This worries me a lot.",
            "I'm anxious about this.",
        ],
    }

    texts = []
    labels = []

    for emotion_idx, (emotion, emotion_texts) in enumerate(emotion_data.items()):
        for text in emotion_texts:
            texts.append(text)
            # Create one-hot encoded label
            label = [0] * 28
            label[emotion_idx] = 1
            labels.append(label)

    logger.info(f"Created {len(texts)} balanced training samples")
    return texts, labels


def improve_model_f1():
    """Improve model F1 score through various techniques."""
    logger.info("🚀 Starting Model F1 Improvement")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model and tokenizer
    model, tokenizer = create_bert_emotion_classifier()
    model.to(device)

    # Create balanced training data
    texts, labels = create_balanced_training_data()

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

    # Setup optimizer and focal loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    focal_loss = FocalLoss(gamma=2.0)

    # Training loop with F1 focus
    model.train()
    for epoch in range(5):
        logger.info(f"📚 Epoch {epoch + 1}/5")
        epoch_loss = 0.0

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

            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"📊 Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    logger.info("✅ Model F1 improvement training completed!")


if __name__ == "__main__":
    improve_model_f1()
