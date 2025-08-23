#!/usr/bin/env python3
"""Debug Training Loss Script for SAMO Deep Learning.

This script investigates why training loss is showing 0.0000 by examining:
- Data loading and label distribution
- Model outputs and predictions
- Loss function calculation
- Numerical precision issues
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import WeightedBCELoss
from src.models.emotion_detection.dataset_loader import create_goemotions_loader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_data_loading():
    """Debug data loading and label distribution."""
    logger.info("🔍 Debugging data loading...")

    try:
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            batch_size=4,  # Small batch for debugging
            num_epochs=1,
            dev_mode=True
        )

        datasets = trainer.prepare_data(dev_mode=True)

        train_dataloader = datasets["train_dataloader"]
        val_dataloader = datasets["val_dataloader"]

        logger.info(f"✅ Train dataloader: {len(train_dataloader)} batches")
        logger.info(f"✅ Val dataloader: {len(val_dataloader)} batches")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 2:  # Only check first 2 batches
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            logger.info(f"📊 Batch {batch_idx + 1} Statistics:")
            logger.info(f"   Input shape: {input_ids.shape}")
            logger.info(f"   Labels shape: {labels.shape}")
            logger.info(f"   Labels dtype: {labels.dtype}")
            logger.info(f"   Labels min: {labels.min().item()}")
            logger.info(f"   Labels max: {labels.max().item()}")
            logger.info(f"   Labels mean: {labels.float().mean().item():.4f}")
            logger.info(f"   Non-zero labels: {(labels > 0).sum().item()}")
            logger.info(f"   Total labels: {labels.numel()}")

            # Check for all-zero or all-one labels
            if labels.sum() == 0:
                logger.warning("⚠️ All labels are zero!")
            elif labels.sum() == labels.numel():
                logger.warning("⚠️ All labels are one!")

            # Check for extreme values
            if labels.max() > 1.0 or labels.min() < 0.0:
                logger.warning(f"⚠️ Labels outside [0,1] range: min={labels.min().item()}, max={labels.max().item()}")

            # Check label distribution per class
            for class_idx in range(labels.shape[1]):
                class_labels = labels[:, class_idx]
                positive_count = (class_labels > 0).sum().item()
                total_count = class_labels.numel()
                logger.info(f"   Class {class_idx}: {positive_count}/{total_count} positive ({positive_count/total_count:.2%})")

        return True

    except Exception as e:
        logger.error(f"❌ Data loading debug failed: {e}")
        return False


def debug_model_outputs(datasets):
    """Debug model outputs and predictions."""
    logger.info("🤖 Debugging model outputs...")

    try:
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            batch_size=4,
            num_epochs=1,
            dev_mode=True
        )

        # Initialize trainer and model
        trainer.initialize_model()
        model = trainer.model

        # Get first batch
        train_dataloader = datasets["train_dataloader"]
        batch = next(iter(train_dataloader))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(logits)

        logger.info("📊 Model Output Statistics:")
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Predictions shape: {predictions.shape}")
        logger.info(f"   Logits min: {logits.min().item():.4f}")
        logger.info(f"   Logits max: {logits.max().item():.4f}")
        logger.info(f"   Logits mean: {logits.mean().item():.4f}")
        logger.info(f"   Predictions min: {predictions.min().item():.4f}")
        logger.info(f"   Predictions max: {predictions.max().item():.4f}")
        logger.info(f"   Predictions mean: {predictions.mean().item():.4f}")

        # Check for extreme values
        if predictions.max() > 0.999 or predictions.min() < 0.001:
            logger.warning(f"⚠️ Predictions near extremes: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}")

        # Examine first batch
        logger.info("📋 First batch details:")
        for i in range(min(3, predictions.shape[0])):
            logger.info(f"   Sample {i}:")
            logger.info(f"     Labels: {labels[i][:5].tolist()}...")
            logger.info(f"     Predictions: {predictions[i][:5].tolist()}...")

        return logits, predictions, labels

    except Exception as e:
        logger.error(f"❌ Model output debug failed: {e}")
        return None, None, None


def debug_loss_calculation(logits, predictions, labels):
    """Debug loss function calculation."""
    logger.info("💔 Debugging loss calculation...")

    try:
        # 1. Standard BCE loss
        bce_loss = nn.BCEWithLogitsLoss()
        loss_bce = bce_loss(logits, labels.float())
        logger.info(f"📊 BCE Loss: {loss_bce.item():.6f}")

        # 2. Manual BCE calculation
        epsilon = 1e-7
        predictions_clipped = torch.clamp(predictions, epsilon, 1 - epsilon)
        manual_loss = -torch.mean(
            labels * torch.log(predictions_clipped) +
            (1 - labels) * torch.log(1 - predictions_clipped)
        )
        logger.info(f"📊 Manual BCE Loss: {manual_loss.item():.6f}")

        # 3. Weighted BCE loss (no weights)
        weighted_bce = WeightedBCELoss()
        loss_weighted = weighted_bce(logits, labels.float())
        logger.info(f"📊 Weighted BCE Loss: {loss_weighted.item():.6f}")

        # 4. Check individual components
        logger.info("📊 Individual Loss Components:")
        for i in range(min(5, logits.shape[1])):
            class_logits = logits[:, i]
            class_labels = labels[:, i].float()
            class_loss = bce_loss(class_logits.unsqueeze(1), class_labels.unsqueeze(1))
            logger.info(f"   Class {i}: {class_loss.item():.6f}")

        # 5. Check for numerical precision issues
        if loss_bce.item() < 1e-10:
            logger.warning("⚠️ Loss is extremely small (< 1e-10)")

        # 6. Test with small epsilon
        predictions_eps = torch.clamp(predictions, 1e-10, 1 - 1e-10)
        loss_eps = -torch.mean(
            labels * torch.log(predictions_eps) +
            (1 - labels) * torch.log(1 - predictions_eps)
        )
        logger.info(f"📊 Loss with epsilon: {loss_eps.item():.6f}")

        return True

    except Exception as e:
        logger.error(f"❌ Loss calculation debug failed: {e}")
        return False


def debug_class_weights():
    """Debug class weights calculation."""
    logger.info("⚖️ Debugging class weights...")

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()
        train_data = datasets["train"]

        # Calculate class weights
        num_classes = 28
        class_counts = torch.zeros(num_classes)

        for example in train_data[:1000]:  # Sample first 1000 examples
            if "labels" in example:
                labels = torch.tensor(example["labels"])
                class_counts += labels

        # Show first 10 class weights
        logger.info(f"📊 First 10 class counts: {class_counts[:10].tolist()}")
        logger.info(f"📊 Total samples: {len(train_data)}")

        # Calculate weights
        total_samples = len(train_data)
        class_weights = total_samples / (num_classes * class_counts + 1)  # Add 1 to avoid division by zero

        logger.info(f"📊 First 10 class weights: {class_weights[:10].tolist()}")
        logger.info(f"📊 Weight range: {class_weights.min().item():.2f} - {class_weights.max().item():.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ Class weights debug failed: {e}")
        return False


def main():
    """Main debug function."""
    logger.info("🚀 Starting Training Loss Debug...")

    # Debug data loading
    if not debug_data_loading():
        return False

    # Debug class weights
    if not debug_class_weights():
        return False

    # Debug model outputs
    trainer = EmotionDetectionTrainer(
        model_name="bert-base-uncased",
        batch_size=4,
        num_epochs=1,
        dev_mode=True
    )
    datasets = trainer.prepare_data(dev_mode=True)

    logits, predictions, labels = debug_model_outputs(datasets)
    if logits is None:
        return False

    # Debug loss calculation
    if not debug_loss_calculation(logits, predictions, labels):
        return False

    # Summary
    logger.info("🎉 Training Loss Debug Complete!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
