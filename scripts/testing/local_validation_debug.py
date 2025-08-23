            # Check per-class distribution
            # Count positive labels
        # Analyze first few examples
        # Calculate statistics
        # Check CUDA
        # Check for critical issues
        # Check for issues
        # Check for issues
        # Check if we have the expected keys
        # Check statistics
        # Compare with manual BCE
        # Create loader without dev_mode parameter
        # Create model
        # Ensure some positive labels
        # Get training data
        # Load data
        # Log class distribution
        # Prepare datasets
        # Scenario 1: Mixed labels
        # Test different scenarios
        # Test forward pass
        from src.models.emotion_detection.bert_classifier import WeightedBCELoss
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader
        import pandas as pd
        import torch
        import torch
        import torch
        import torch.nn.functional as F
        import transformers
    # Run all validations
    # Run validations
    # Summary
# Add src to path
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
import logging
import numpy as np
import sys

"""
Local Validation and Debug Script for SAMO Deep Learning.

This script performs targeted validation to identify the root cause of the 0.0000 loss issue.
It can be run locally to diagnose problems before deploying to GCP.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_environment():
    """Check basic environment setup."""
    logger.info("🔍 Checking environment...")

    try:
        logger.info("✅ PyTorch: {torch.__version__}")
        logger.info("✅ Transformers: {transformers.__version__}")
        logger.info("✅ NumPy: {np.__version__}")
        logger.info("✅ Pandas: {pd.__version__}")

        if torch.cuda.is_available():
            logger.info("✅ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("✅ CPU mode available")

        return True

    except Exception as e:
        logger.error("❌ Environment check failed: {e}")
        return False


def check_data_loading():
    """Check data loading functionality."""
    logger.info("🔍 Checking data loading...")

    try:
        logger.info("   Loading dataset...")
        loader = create_goemotions_loader()

        datasets = loader.prepare_datasets()

        expected_keys = ["train", "validation", "test", "statistics", "class_weights"]
        for key in expected_keys:
            if key not in datasets:
                logger.error("❌ Missing key in datasets: {key}")
                return False

        logger.info("✅ Train set: {len(datasets['train'])} examples")
        logger.info("✅ Validation set: {len(datasets['validation'])} examples")
        logger.info("✅ Test set: {len(datasets['test'])} examples")

        stats = datasets["statistics"]
        logger.info("✅ Total examples: {stats.get('total_examples', 'N/A')}")
        logger.info("✅ Emotion distribution: {len(stats.get('emotion_counts', {}))} emotions")

        return True

    except Exception as e:
        logger.error("❌ Data loading failed: {e}")
        return False


def check_model_creation():
    """Check model creation and forward pass."""
    logger.info("🔍 Checking model creation...")

    try:
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=6,
        )

        logger.info("✅ Model created: {model.count_parameters():,} parameters")
        logger.info("✅ Loss function: {type(loss_fn).__name__}")

        batch_size = 2
        seq_length = 64
        num_classes = 28

        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones(batch_size, seq_length)
        dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).float()

        dummy_labels[:, 0] = 1.0

        model.eval()
        with torch.no_grad():
            logits = model(dummy_input_ids, dummy_attention_mask)
            loss = loss_fn(logits, dummy_labels)

        logger.info("✅ Forward pass successful")
        logger.info("   Logits shape: {logits.shape}")
        logger.info("   Loss value: {loss.item():.8f}")

        if loss.item() <= 0:
            logger.error("❌ CRITICAL: Loss is zero or negative: {loss.item()}")
            return False

        if torch.isnan(loss).any():
            logger.error("❌ CRITICAL: NaN loss!")
            return False

        return True

    except Exception as e:
        logger.error("❌ Model creation failed: {e}")
        return False


def check_loss_function():
    """Check loss function implementation."""
    logger.info("🔍 Checking loss function...")

    try:
        batch_size = 4
        num_classes = 28

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        labels[:, 0] = 1.0  # Ensure some positive labels

        loss_fn = WeightedBCELoss()
        loss1 = loss_fn(logits, labels)

        bce_manual = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        logger.info("✅ Mixed labels loss: {loss1.item():.8f}")
        logger.info("✅ All positive loss: {loss_fn(logits, torch.ones(batch_size, num_classes)).item():.8f}")
        logger.info("✅ All negative loss: {loss_fn(logits, torch.zeros(batch_size, num_classes)).item():.8f}")
        logger.info("✅ Manual BCE loss: {bce_manual.item():.8f}")

        if loss1.item() <= 0:
            logger.error("❌ CRITICAL: Loss function producing zero/negative values!")
            return False

        return True

    except Exception as e:
        logger.error("❌ Loss function check failed: {e}")
        return False


def check_data_distribution():
    """Check data distribution to identify 0.0000 loss causes."""
    logger.info("🔍 Checking data distribution...")

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        train_dataset = datasets["train"]

        total_samples = min(100, len(train_dataset))
        total_positive_labels = 0
        label_distribution = {}

        for i in range(total_samples):
            example = train_dataset[i]
            labels = example["labels"]

            positive_count = sum(labels)
            total_positive_labels += positive_count

            for _class_idx, label in enumerate(labels):
                if class_idx not in label_distribution:
                    label_distribution[class_idx] = 0
                if label == 1:
                    label_distribution[class_idx] += 1

        total_possible_labels = total_samples * 28  # 28 emotion classes
        positive_rate = total_positive_labels / total_possible_labels

        logger.info("✅ Total samples analyzed: {total_samples}")
        logger.info("✅ Total positive labels: {total_positive_labels}")
        logger.info("✅ Positive label rate: {positive_rate:.6f}")

        if positive_rate == 0:
            logger.error("❌ CRITICAL: No positive labels found!")
            logger.error("   This will cause 0.0000 loss with BCE")
            return False
        elif positive_rate == 1:
            logger.error("❌ CRITICAL: All labels are positive!")
            logger.error("   This will cause 0.0000 loss with BCE")
            return False
        elif positive_rate < 0.01:
            logger.warning("⚠️  Very low positive label rate")
            logger.warning("   Consider using focal loss or class weights")

        logger.info("📊 Class distribution (first 10 classes):")
        for class_idx in range(min(10, len(label_distribution))):
            count = label_distribution.get(class_idx, 0)
            if count > 0:
                logger.info("   Class {class_idx}: {count} positive samples")

        return True

    except Exception as e:
        logger.error("❌ Data distribution check failed: {e}")
        return False


def main():
    """Main function to run all validations."""
    logger.info("🚀 SAMO-DL Local Validation and Debug")
    logger.info("=" * 50)

    validations = [
        ("Environment", check_environment),
        ("Data Loading", check_data_loading),
        ("Model Creation", check_model_creation),
        ("Loss Function", check_loss_function),
        ("Data Distribution", check_data_distribution),
    ]

    results = {}
    for name, validation_func in validations:
        logger.info("\n{'='*40}")
        logger.info("Running: {name}")
        logger.info("{'='*40}")

        try:
            success = validation_func()
            results[name] = success

            if success:
                logger.info("✅ {name} PASSED")
            else:
                logger.error("❌ {name} FAILED")

        except Exception as e:
            logger.error("❌ {name} ERROR: {e}")
            results[name] = False

    passed = sum(results.values())
    total = len(results)

    logger.info("\n{'='*50}")
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("{'='*50}")
    logger.info("Total checks: {total}")
    logger.info("Passed: {passed}")
    logger.info("Failed: {total - passed}")

    if passed == total:
        logger.info("\n✅ ALL VALIDATIONS PASSED!")
        logger.info("   The 0.0000 loss issue is likely due to:")
        logger.info("   1. Learning rate too high (try 2e-6 instead of 2e-5)")
        logger.info("   2. Need focal loss for class imbalance")
        logger.info("   3. Need class weights for imbalanced data")
        logger.info("   Ready for training with optimized configuration!")
    else:
        logger.error("\n❌ SOME CHECKS FAILED!")
        logger.error("   Fix the issues above before proceeding")
        logger.error("   This will prevent the 0.0000 loss problem")

    return passed == total


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
