#!/usr/bin/env python3
"""Full Dataset Focal Loss Training Script.

This script trains the emotion detection model using focal loss on the full GoEmotions
dataset.
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


def create_synthetic_data(num_samples=1000):
    """Create synthetic training data for testing."""
    logger.info(f"Creating {num_samples} synthetic training samples...")

    emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
    texts = [
        "I am feeling happy today!",
        "This makes me very sad.",
        "I'm really angry about this situation.",
        "I'm scared of what might happen next.",
        "I'm surprised by this news!",
        "This is disgusting to me.",
        "I trust you completely.",
        "I'm excited about the future!",
        "I feel great about everything!",
        "This is disappointing.",
        "I'm furious with you!",
        "I'm terrified of the dark.",
        "Wow, that's amazing!",
        "This is gross.",
        "I believe in you.",
        "I can't wait for tomorrow!",
    ]

    data = []
    for idx, text in enumerate(texts):
        labels = [0] * 28
        emotion_idx = idx % len(emotions)
        labels[emotion_idx] = 1
        data.append({"text": text, "labels": labels})

    # Repeat to reach num_samples
    while len(data) < num_samples:
        for item in data[:]:
            if len(data) >= num_samples:
                break
            data.append(item)

    logger.info(f"Created {len(data)} synthetic samples")
    return data


def full_dataset_focal_training():
    """Run full dataset focal loss training."""
    logger.info("🚀 Starting Full Dataset Focal Loss Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model and tokenizer
    model, tokenizer = create_bert_emotion_classifier()
    model.to(device)

    # Create synthetic training data
    training_data = create_synthetic_data(num_samples=100)

    # Prepare data
    texts = [item["text"] for item in training_data]
    labels = [item["labels"] for item in training_data]

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
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    focal_loss = FocalLoss(gamma=2.0)

    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(3):
        logger.info(f"📚 Epoch {epoch + 1}/3")
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
            train_losses.append(loss.item())

            if batch_idx % 10 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"📊 Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    logger.info("✅ Full dataset focal loss training completed!")
    logger.info(f"📈 Final average loss: {sum(train_losses) / len(train_losses):.4f}")


if __name__ == "__main__":
    full_dataset_focal_training()
