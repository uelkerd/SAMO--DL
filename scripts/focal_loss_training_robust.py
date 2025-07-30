            # Backward pass
            # Forward pass
            # Move batch to device
        # Create aliases
        # Load bert_classifier module
        # Load dataset_loader module
        # Look for src directory in parent directories
        import importlib.util
    # Add to Python path
    # Alternative: Try to import directly from the file paths
    # Approach 1: Get the script's directory and go up to project root
    # Approach 2: Try to find the project root by looking for src directory
    # Create focal loss
    # Create model
    # Create optimizer
    # Get a small batch for testing
    # Load dataset using existing loader
    # Setup device
    # Training loop (simplified for testing)
    from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
    from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
# Configure logging
# Multiple approaches to add project root to Python path
# Now try to import the modules
#!/usr/bin/env python3
from pathlib import Path
from torch import nn
import logging
import sys
import torch
import torch.nn.functional as F





"""
Robust Focal Loss Training for Emotion Detection

This script implements Focal Loss to address class imbalance and improve F1 scores.
Uses multiple approaches to handle Python path issues.

Usage:
    python3 focal_loss_training_robust.py
"""

try:
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()

    if not (project_root / "src").exists():
        current = script_dir
        while current.parent != current:
            current = current.parent
            if (current / "src").exists():
                project_root = current
                break

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("🔧 Added project root to path: {project_root}")
    print("📁 Current working directory: {Path.cwd()}")
    print("📋 Python path: {sys.path[:3]}...")

except Exception as e:
    print("⚠️  Path setup warning: {e}")

try:
    print("✅ Successfully imported modules")
except ImportError as _:
    print("❌ Import error: {e}")
    print("🔧 Trying alternative import approach...")

    try:
        bert_spec = importlib.util.spec_from_file_location(
            "bert_classifier",
            project_root / "src" / "models" / "emotion_detection" / "bert_classifier.py",
        )
        bert_module = importlib.util.module_from_spec(bert_spec)
        bert_spec.loader.exec_module(bert_module)

        loader_spec = importlib.util.spec_from_file_location(
            "dataset_loader",
            project_root / "src" / "models" / "emotion_detection" / "dataset_loader.py",
        )
        loader_module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(loader_module)

        create_bert_emotion_classifier = bert_module.create_bert_emotion_classifier
        GoEmotionsDataLoader = loader_module.GoEmotionsDataLoader

        print("✅ Successfully imported modules using alternative approach")

    except Exception as e2:
        print("❌ Alternative import also failed: {e2}")
        print("🔧 Please check the project structure and run from the correct directory")
        sys.exit(1)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    logger.info("📊 Loading GoEmotions dataset...")
    try:
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        train_dataset = datasets["train"]
        val_dataset = datasets["validation"]
        test_dataset = datasets["test"]
        emotion_names = data_loader.emotion_names

        logger.info("✅ Dataset loaded successfully")
        logger.info("   • Train examples: {len(train_dataset)}")
        logger.info("   • Validation examples: {len(val_dataset)}")
        logger.info("   • Test examples: {len(test_dataset)}")
        logger.info("   • Emotion classes: {len(emotion_names)}")

    except Exception as e:
        logger.error("❌ Failed to load dataset: {e}")
        return

    logger.info("🤖 Creating BERT emotion classifier...")
    try:
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
        )
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model created successfully")
        logger.info("   • Total parameters: {param_count:,}")
        logger.info("   • Trainable parameters: {trainable_count:,}")

    except Exception as e:
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

    for epoch in range(3):
        logger.info("📚 Epoch {epoch + 1}/3")

        for _batch_idx, batch in enumerate(train_dataloader):
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
