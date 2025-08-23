# Backward pass
# Forward pass
# Log progress every 100 batches
# Apply alpha weighting
# Apply reduction
# Apply sigmoid to get probabilities
# Calculate binary cross entropy
# Calculate focal loss components
# Combine all components
# Log progress
# Save best model
# Training phase
# Validation phase
# Create data loaders
# Create focal loss
# Create model
# Load dataset using existing loader
# Run training
# Save final model
# Setup device
# Setup optimizer
# Training loop
# Add src to path
# Configure logging
#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

"""
Focal Loss Training for Emotion Detection (Fixed Version)

This script implements Focal Loss to address class imbalance and improve F1 scores.
Fixed to use the existing dataset loader and avoid compatibility issues.

Usage:
    python scripts/focal_loss_training_fixed.py [--gamma 2.0] [--alpha 0.25]
"""

sys.path.append(str(Path(__file__).parent.parent.resolve()))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss implementation for multi-label classification."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def train_with_focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 512,
    output_dir: str = "./models/checkpoints",
) -> dict:
    """Train emotion detection model with Focal Loss.

    Args:
        gamma: Focal loss gamma parameter (focusing parameter)
        alpha: Focal loss alpha parameter (class balancing)
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Maximum sequence length
        output_dir: Directory to save model checkpoints

    Returns:
        Training results dictionary
    """
    logger.info("🚀 Starting SAMO-DL Focal Loss Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")
    logger.info("CPU Threads: {torch.get_num_threads()}")
    logger.info("Parameters: gamma={gamma}, alpha={alpha}, lr={learning_rate}")
    logger.info("Batch size: {batch_size}, Max length: {max_length}")

    logger.info("📊 Loading GoEmotions dataset...")
    try:
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()  # Use the correct method name

        train_dataset = datasets["train"]
        val_dataset = datasets["validation"]
        datasets["test"]
        datasets["class_weights"]

        logger.info("Dataset loaded successfully:")
        logger.info("   • Train: {len(train_dataset)} examples")
        logger.info("   • Validation: {len(val_dataset)} examples")
        logger.info("   • Test: {len(test_dataset)} examples")

    except Exception:
        logger.error("Failed to load dataset: {e}")
        raise

    logger.info("🤖 Creating BERT model...")
    model, _ = create_bert_emotion_classifier(
        model_name="bert-base-uncased",
        class_weights=None,  # We'll use focal loss instead
        freeze_bert_layers=4,
    )
    model.to(device)

    focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    best_val_loss = float("in")
    training_history = []

    for epoch in range(num_epochs):
        logger.info("\n📈 Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        num_batches = 0

        for _batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    "   Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

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
                loss = focal_loss(outputs, labels)

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
            checkpoint_path = Path(output_dir) / "focal_loss_best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "training_history": training_history,
                    "focal_loss_params": {"alpha": alpha, "gamma": gamma},
                },
                checkpoint_path,
            )

            logger.info("   💾 Saved best model (val_loss: {avg_val_loss:.4f})")

    final_checkpoint_path = Path(output_dir) / "focal_loss_final_model.pt"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "training_history": training_history,
            "focal_loss_params": {"alpha": alpha, "gamma": gamma},
        },
        final_checkpoint_path,
    )

    logger.info("✅ Training completed!")
    logger.info("   • Best validation loss: {best_val_loss:.4f}")
    logger.info("   • Final validation loss: {avg_val_loss:.4f}")
    logger.info("   • Models saved to: {output_dir}")

    return {
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
        "training_history": training_history,
        "model_path": str(final_checkpoint_path),
    }


def main():
    """Main function to run focal loss training."""
    parser = argparse.ArgumentParser(
        description="Train emotion detection with Focal Loss"
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Focal loss gamma parameter"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Focal loss alpha parameter"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/checkpoints",
        help="Output directory",
    )

    args = parser.parse_args()

    train_with_focal_loss(
        gamma=args.gamma,
        alpha=args.alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    logger.info("🎉 Focal Loss training completed successfully!")
    logger.info("📊 Results: {results}")


if __name__ == "__main__":
    main()
