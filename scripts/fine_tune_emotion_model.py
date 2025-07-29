#!/usr/bin/env python3
"""
Fine-tune Emotion Detection Model on GoEmotions Dataset

This script fine-tunes the BERT model on the GoEmotions dataset
to improve emotion detection performance.
"""

import os
import sys
import logging
import torch
from torch import nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fine_tune_model():
    """Fine-tune the emotion detection model on GoEmotions dataset."""

    logger.info("🎯 Starting Model Fine-tuning")
    logger.info("   • Dataset: GoEmotions")
    logger.info("   • Model: BERT-base-uncased")
    logger.info("   • Epochs: 5")
    logger.info("   • Learning Rate: 1e-05")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load dataset
        logger.info("Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()
        
        train_dataset = datasets["train"]  # Fixed key name
        val_dataset = datasets["validation"]  # Fixed key name
        test_dataset = datasets["test"]  # Fixed key name
        class_weights = datasets["class_weights"]

        logger.info("Dataset loaded successfully:")
        logger.info(f"   • Train: {len(train_dataset)} examples")
        logger.info(f"   • Validation: {len(val_dataset)} examples")
        logger.info(f"   • Test: {len(test_dataset)} examples")

        # Create model
        logger.info("Creating BERT model...")
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=class_weights,  # Use class weights for imbalance
            freeze_bert_layers=2,  # Freeze fewer layers for fine-tuning
        )
        model.to(device)

        # Setup loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        training_history = []

        for epoch in range(5):  # 5 epochs for fine-tuning
            logger.info(f"\nEpoch {epoch + 1}/5")

            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs["logits"], labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # Log progress every 100 batches
                if num_batches % 100 == 0:
                    logger.info(f"   • Batch {num_batches}: Loss = {loss.item():.4f}")

            avg_train_loss = train_loss / num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].float().to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs["logits"], labels)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Log progress
            logger.info(f"   • Train Loss: {avg_train_loss:.4f}")
            logger.info(f"   • Val Loss: {avg_val_loss:.4f}")
            logger.info(f"   • Learning Rate: {current_lr:.2e}")

            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                }
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"   • New best validation loss: {best_val_loss:.4f}")

                # Save model
                output_dir = "./models/checkpoints"
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, "fine_tuned_model.pt")

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": best_val_loss,
                        "training_history": training_history,
                        "class_weights": class_weights,
                    },
                    model_path,
                )

                logger.info(f"   • Model saved to: {model_path}")

        logger.info("🎉 Fine-tuning completed successfully!")
        logger.info(f"   • Best validation loss: {best_val_loss:.4f}")
        logger.info("   • Model saved to: ./models/checkpoints/fine_tuned_model.pt")

        return True

    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("🎯 Fine-tuning Script")
    logger.info("This script fine-tunes the emotion detection model on GoEmotions")

    success = fine_tune_model()

    if success:
        logger.info("✅ Fine-tuning completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Fine-tuning failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
