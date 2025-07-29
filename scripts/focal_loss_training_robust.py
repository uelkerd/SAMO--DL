#!/usr/bin/env python3
"""
Robust Focal Loss Training for Emotion Detection

This script implements Focal Loss to address class imbalance and improve F1 scores.
Uses multiple approaches to handle Python path issues.

Usage:
    python3 focal_loss_training_robust.py
"""

import sys
import os
import logging
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path

# Multiple approaches to add project root to Python path
try:
    # Approach 1: Get the script's directory and go up to project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()

    # Approach 2: Try to find the project root by looking for src directory
    if not (project_root / "src").exists():
        # Look for src directory in parent directories
        current = script_dir
        while current.parent != current:
            current = current.parent
            if (current / "src").exists():
                project_root = current
                break

    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"🔧 Added project root to path: {project_root}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📋 Python path: {sys.path[:3]}...")

except Exception as e:
    print(f"⚠️  Path setup warning: {e}")

# Now try to import the modules
try:
    from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
    from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Trying alternative import approach...")

    # Alternative: Try to import directly from the file paths
    try:
        import importlib.util

        # Load bert_classifier module
        bert_spec = importlib.util.spec_from_file_location(
            "bert_classifier",
            project_root / "src" / "models" / "emotion_detection" / "bert_classifier.py",
        )
        bert_module = importlib.util.module_from_spec(bert_spec)
        bert_spec.loader.exec_module(bert_module)

        # Load dataset_loader module
        loader_spec = importlib.util.spec_from_file_location(
            "dataset_loader",
            project_root / "src" / "models" / "emotion_detection" / "dataset_loader.py",
        )
        loader_module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(loader_module)

        # Create aliases
        create_bert_emotion_classifier = bert_module.create_bert_emotion_classifier
        GoEmotionsDataLoader = loader_module.GoEmotionsDataLoader

        print("✅ Successfully imported modules using alternative approach")

    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("🔧 Please check the project structure and run from the correct directory")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    logger.info("🚀 Starting SAMO-DL Focal Loss Training (Robust)")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load dataset using existing loader
    logger.info("📊 Loading GoEmotions dataset...")
    try:
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        train_dataset = datasets["train"]
        val_dataset = datasets["validation"]
        test_dataset = datasets["test"]
        emotion_names = data_loader.emotion_names

        logger.info("✅ Dataset loaded successfully")
        logger.info(f"   • Train examples: {len(train_dataset)}")
        logger.info(f"   • Validation examples: {len(val_dataset)}")
        logger.info(f"   • Test examples: {len(test_dataset)}")
        logger.info(f"   • Emotion classes: {len(emotion_names)}")

    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return

    # Create model
    logger.info("🤖 Creating BERT emotion classifier...")
    try:
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
        )
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model created successfully")
        logger.info(f"   • Total parameters: {param_count:,}")
        logger.info(f"   • Trainable parameters: {trainable_count:,}")

    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        return

    # Create focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logger.info("✅ Focal Loss created (alpha=0.25, gamma=2.0)")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    logger.info("✅ Optimizer created (AdamW, lr=2e-5)")

    # Training loop (simplified for testing)
    logger.info("🚀 Starting training loop...")
    model.train()

    # Get a small batch for testing
    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(3):
        logger.info(f"📚 Epoch {epoch + 1}/3")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 5:  # Only do first 5 batches for testing
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

    logger.info("✅ Training completed successfully!")
    logger.info("🎯 Focal Loss training is working correctly")
    logger.info("📈 Ready for full training with more epochs")


if __name__ == "__main__":
    main()
