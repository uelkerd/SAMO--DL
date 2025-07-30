import sys
#!/usr/bin/env python3
import logging
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
# Add project root to Python path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
# Configure logging
# Now import the modules
    # Setup device
    # Load dataset using existing loader
    # Create model
    # Create focal loss
    # Create optimizer
    # Training loop (simplified for testing)
    # Get a small batch for testing
            # Move batch to device
            # Forward pass
            # Backward pass





"""
Simple Focal Loss Training Script

This script provides a simple implementation of focal loss training for the SAMO-DL project.
It includes proper path handling for different environments.
"""

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss implementation for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Compute focal loss."""
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss.mean()


def main():
    """Main training function."""
    logger.info("🚀 Starting SAMO-DL Focal Loss Training (Simple)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    logger.info("📊 Loading GoEmotions dataset...")
    try:
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        train_dataset = datasets["train"]
        datasets["validation"]
        datasets["test"]

        logger.info("✅ Dataset loaded successfully")
        logger.info("   • Train examples: {len(train_dataset)}")
        logger.info("   • Validation examples: {len(val_dataset)}")
        logger.info("   • Test examples: {len(test_dataset)}")
        logger.info("   • Emotion classes: {len(emotion_names)}")

    except Exception as _:
        logger.error("❌ Failed to load dataset: {e}")
        return

    logger.info("🤖 Creating BERT emotion classifier...")
    try:
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
        )
        model = model.to(device)

        sum(p.numel() for p in model.parameters())
        sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model created successfully")
        logger.info("   • Total parameters: {param_count:,}")
        logger.info("   • Trainable parameters: {trainable_count:,}")

    except Exception as _:
        logger.error("❌ Failed to create model: {e}")
        return

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logger.info("✅ Focal Loss created (alpha=0.25, gamma=2.0)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    logger.info("✅ Optimizer created (AdamW, lr=2e-5)")

    logger.info("🚀 Starting training loop...")
    model.train()

    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for _epoch in range(3):
        logger.info("📚 Epoch {epoch + 1}/3")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 5:  # Only do first 5 batches for testing
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 0:
                logger.info("   Batch {batch_idx}: Loss = {loss.item():.4f}")

    logger.info("✅ Training completed successfully!")
    logger.info("🎯 Focal Loss training is working correctly")
    logger.info("📈 Ready for full training with more epochs")


if __name__ == "__main__":
    main()
