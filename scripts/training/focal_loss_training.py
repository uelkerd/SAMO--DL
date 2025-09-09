# Extract texts and labels from raw datasets
# Focal loss components
# Load dataset
# Setup optimizer
# Training loop
from src.models.emotion_detection.bert_classifier import EmotionDataset
from transformers import AutoTokenizer
import traceback
# Setup device
# Add project root to path
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import create_bert_emotion_classifier
from torch import nn
import logging
import os
import sys
import torch
import traceback






"""
Focal Loss Training Script for SAMO Emotion Detection

This script implements focal loss training to improve F1 score
"""

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
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


def train_with_focal_loss():
    """Train BERT model with focal loss for improved F1 score."""

    logger.info("🚀 Starting Focal Loss Training")
    logger.info("   • Gamma: 2.0")
    logger.info("   • Alpha: 0.25")
    logger.info("   • Learning Rate: 2e-05")
    logger.info("   • Epochs: 3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {device}")

    try:
        logger.info("Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()  # Fixed method name

        train_raw = datasets["train"]
        val_raw = datasets["validation"]
        test_raw = datasets["test"]
        _ = datasets["class_weights"]

        train_texts = [item["text"] for item in train_raw]
        train_labels = [item["labels"] for item in train_raw]

        val_texts = [item["text"] for item in val_raw]
        val_labels = [item["labels"] for item in val_raw]

        test_texts = [item["text"] for item in test_raw]
        test_labels = [item["labels"] for item in test_raw]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=512)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=512)
        _ = EmotionDataset(test_texts, test_labels, tokenizer, max_length=512)

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

        best_val_loss = float("inf")
        training_history = []

        for epoch in range(3):  # Quick 3 epochs
            logger.info("\nEpoch {epoch + 1}/3")

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
                {"epoch": epoch + 1, "train_loss": avg_train_loss,
                 "val_loss": avg_val_loss}
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info("   • New best validation loss: {best_val_loss:.4f}")

                output_dir = "./models/checkpoints"
                os.makedirs(output_dir, exist_ok=True)
                model_path = Path(output_dir, "focal_loss_best_model.pt")

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

        logger.info("🎉 Focal Loss Training completed successfully!")
        logger.info("   • Best validation loss: {best_val_loss:.4f}")
        logger.info("   • Model saved to: ./models/checkpoints/focal_loss_best_model.pt")

        return True

    except Exception as e:
        logger.error("❌ Training failed: %s", e)
        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("🧪 Focal Loss Training Script")
    logger.info("This script implements focal loss to improve F1 score")

    success = train_with_focal_loss()

    if success:
        logger.info("✅ Focal loss training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
