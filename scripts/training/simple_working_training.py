                # Backward pass
                # Forward pass
                # Log progress every 100 batches
                # Save model
            # Log progress
            # Save best model
            # Training phase
            # Validation phase
        # BCE loss
        # Create data loaders
        # Create focal loss
        # Create model
        # Focal loss components
        # Load dataset
        # Setup optimizer
        # Training loop
        import traceback
    # Setup device
import logging
import os
import sys
import traceback

# Add project root to path
# Configure logging
#!/usr/bin/env python3
from pathlib import Path

import torch
from torch import nn

from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import (
    create_bert_emotion_classifier,
)

"""
Simple Working Training Script - FIXES ALL ISSUES

This script addresses the critical issues:
1. Method name mismatch (prepare_data vs prepare_datasets)
2. Missing model files
3. Proper error handling
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass with focal loss calculation."""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def train_simple_model():
    """Train a simple BERT model with focal loss."""

    logger.info("🚀 Starting Simple Working Training")
    logger.info("   • Focal Loss: alpha=0.25, gamma=2.0")
    logger.info("   • Learning Rate: 2e-05")
    logger.info("   • Epochs: 2 (quick training)")
    logger.info("   • Batch Size: 16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {device}")

    try:
        logger.info("Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()  # Use correct method name

        train_dataset = datasets["train_dataset"]
        val_dataset = datasets["val_dataset"]
        test_dataset = datasets["test_dataset"]
        datasets["class_weights"]

        logger.info("Dataset loaded successfully:")
        logger.info("   • Train: {len(train_dataset)} examples")
        logger.info("   • Validation: {len(val_dataset)} examples")
        logger.info("   • Test: {len(test_dataset)} examples")

        logger.info("Creating BERT model...")
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,  # Use focal loss instead
            freeze_bert_layers=4,
        )
        model.to(device)

        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        best_val_loss = float("in")
        training_history = []

        for epoch in range(2):  # Quick 2 epochs
            logger.info("\nEpoch {epoch + 1}/2")

            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = focal_loss(outputs["logits"], labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                if num_batches % 100 == 0:
                    logger.info("   • Batch {num_batches}: Loss = {loss.item():.4f}")

            avg_train_loss = train_loss / num_batches

            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].float().to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = focal_loss(outputs["logits"], labels)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches

            logger.info("   • Train Loss: {avg_train_loss:.4f}")
            logger.info("   • Val Loss: {avg_val_loss:.4f}")

            training_history.append(
                {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info("   • New best validation loss: {best_val_loss:.4f}")

                output_dir = "./models/checkpoints"
                os.makedirs(output_dir, exist_ok=True)
                model_path = Path(output_dir, "simple_working_model.pt")

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": best_val_loss,
                        "training_history": training_history,
                    },
                    model_path,
                )

                logger.info("   • Model saved to: {model_path}")

        logger.info("🎉 Training completed successfully!")
        logger.info("   • Best validation loss: {best_val_loss:.4f}")
        logger.info("   • Model saved to: ./models/checkpoints/simple_working_model.pt")

        return True

    except Exception as e:
        logger.error("❌ Training failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("🧪 Simple Working Training Script")
    logger.info("This script fixes all the critical issues from previous attempts")

    success = train_simple_model()

    if success:
        logger.info("✅ All issues resolved! Training completed successfully.")
        sys.exit(0)
    else:
        logger.error("❌ Training failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
