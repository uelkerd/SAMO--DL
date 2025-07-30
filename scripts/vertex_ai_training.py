import os
import sys
import traceback

#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
        import torch
        import transformers
        from google.cloud import aiplatform

        from models.emotion_detection.dataset_loader import create_goemotions_loader

        # Load data
        from models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        import torch

        # Create model
        from models.emotion_detection.bert_classifier import WeightedBCELoss
        import torch
        import torch.nn.functional as F

        # Test different scenarios
        from models.emotion_detection.training_pipeline import EmotionDetectionTrainer

        # Create trainer with optimized configuration
        import traceback

"""
Vertex AI Training Script for SAMO Deep Learning.

This script runs training on Vertex AI with optimized configuration
to solve the 0.0000 loss issue and achieve >75% F1 score.
"""

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/vertex_training.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vertex AI Training for SAMO Deep Learning")

    # Model configuration
    parser.add_argument("--model_name", default="bert-base-uncased", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate (optimized for stability)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--freeze_bert_layers", type=int, default=6, help="Number of BERT layers to freeze")

    # Training configuration
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss instead of BCE")
    parser.add_argument("--class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--dev_mode", action="store_true", help="Run in development mode")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debugging mode")

    # Validation configuration
    parser.add_argument("--validation_mode", action="store_true", help="Run validation only")
    parser.add_argument("--check_data_distribution", action="store_true", help="Check data distribution")
    parser.add_argument("--check_model_architecture", action="store_true", help="Check model architecture")
    parser.add_argument("--check_loss_function", action="store_true", help="Check loss function")
    parser.add_argument("--check_training_config", action="store_true", help="Check training configuration")

    return parser.parse_args()


def validate_environment():
    """Validate Vertex AI environment."""
    logger.info("🔍 Validating Vertex AI environment...")

    try:
        logger.info("✅ PyTorch: {torch.__version__}")
        logger.info("✅ Transformers: {transformers.__version__}")
        logger.info("✅ Vertex AI: Available")

        # Check CUDA
        if torch.cuda.is_available():
            logger.info("✅ CUDA: {torch.cuda.get_device_name(0)}")
            logger.info("✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️  CUDA not available, using CPU")

        # Check Vertex AI environment
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        region = os.getenv("VERTEX_AI_REGION", "us-central1")

        logger.info("✅ Project ID: {project_id}")
        logger.info("✅ Region: {region}")

        return True

    except Exception as _:
        logger.error("❌ Environment validation failed: {e}")
        return False


def validate_data_distribution():
    """Validate data distribution to identify 0.0000 loss causes."""
    logger.info("🔍 Validating data distribution...")

    try:
        datasets = create_goemotions_loader(dev_mode=True)
        train_dataloader = datasets["train_dataloader"]

        # Analyze first few batches
        total_samples = 0
        total_positive_labels = 0
        label_distribution = {}

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 10:  # Check first 10 batches
                break

            labels = batch["labels"]
            total_samples += labels.shape[0]
            total_positive_labels += labels.sum().item()

            # Check per-class distribution
            for class_idx in range(labels.shape[1]):
                if class_idx not in label_distribution:
                    label_distribution[class_idx] = 0
                label_distribution[class_idx] += labels[:, class_idx].sum().item()

        # Calculate statistics
        positive_rate = total_positive_labels / (total_samples * 28)  # 28 emotion classes
        logger.info("✅ Total samples analyzed: {total_samples}")
        logger.info("✅ Total positive labels: {total_positive_labels}")
        logger.info("✅ Positive label rate: {positive_rate:.6f}")

        # Check for critical issues
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

        # Log class distribution
        logger.info("📊 Class distribution (first 10 classes):")
        for class_idx in range(min(10, len(label_distribution))):
            count = label_distribution.get(class_idx, 0)
            if count > 0:
                logger.info("   Class {class_idx}: {count} positive samples")

        return True

    except Exception as _:
        logger.error("❌ Data distribution validation failed: {e}")
        return False


def validate_model_architecture():
    """Validate model architecture."""
    logger.info("🔍 Validating model architecture...")

    try:
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=6,
        )

        logger.info("✅ Model created: {model.count_parameters():,} parameters")
        logger.info("✅ Loss function: {type(loss_fn).__name__}")

        # Test forward pass
        batch_size = 2
        seq_length = 64
        num_classes = 28

        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones(batch_size, seq_length)
        dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Ensure some positive labels
        dummy_labels[:, 0] = 1.0

        model.eval()
        with torch.no_grad():
            logits = model(dummy_input_ids, dummy_attention_mask)
            loss = loss_fn(logits, dummy_labels)

        logger.info("✅ Forward pass successful")
        logger.info("   Logits shape: {logits.shape}")
        logger.info("   Loss value: {loss.item():.8f}")

        # Check for issues
        if loss.item() <= 0:
            logger.error("❌ CRITICAL: Loss is zero or negative: {loss.item()}")
            return False

        if torch.isnan(loss).any():
            logger.error("❌ CRITICAL: NaN loss!")
            return False

        return True

    except Exception as _:
        logger.error("❌ Model architecture validation failed: {e}")
        return False


def validate_loss_function():
    """Validate loss function implementation."""
    logger.info("🔍 Validating loss function...")

    try:
        batch_size = 4
        num_classes = 28

        # Scenario 1: Mixed labels
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        labels[:, 0] = 1.0  # Ensure some positive labels

        loss_fn = WeightedBCELoss()
        loss1 = loss_fn(logits, labels)

        # Compare with manual BCE
        bce_manual = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        logger.info("✅ Mixed labels loss: {loss1.item():.8f}")
        logger.info("✅ Manual BCE loss: {bce_manual.item():.8f}")

        # Check difference
        loss_diff = abs(loss1.item() - bce_manual.item())
        if loss_diff > 1.0:
            logger.warning("⚠️  Large difference between custom and manual loss: {loss_diff}")

        # Test edge cases
        # All positive
        labels_all_pos = torch.ones(batch_size, num_classes)
        loss2 = loss_fn(logits, labels_all_pos)
        logger.info("✅ All positive loss: {loss2.item():.8f}")

        # All negative
        labels_all_neg = torch.zeros(batch_size, num_classes)
        loss3 = loss_fn(logits, labels_all_neg)
        logger.info("✅ All negative loss: {loss3.item():.8f}")

        # Check for issues
        if loss1.item() <= 0 or loss2.item() <= 0 or loss3.item() <= 0:
            logger.error("❌ CRITICAL: Loss function producing zero/negative values!")
            return False

        return True

    except Exception as _:
        logger.error("❌ Loss function validation failed: {e}")
        return False


def validate_training_config(args):
    """Validate training configuration."""
    logger.info("🔍 Validating training configuration...")

    try:
        logger.info("📋 Training Configuration:")
        logger.info("   Model: {args.model_name}")
        logger.info("   Batch size: {args.batch_size}")
        logger.info("   Learning rate: {args.learning_rate}")
        logger.info("   Epochs: {args.num_epochs}")
        logger.info("   Max length: {args.max_length}")
        logger.info("   Frozen layers: {args.freeze_bert_layers}")
        logger.info("   Focal loss: {args.use_focal_loss}")
        logger.info("   Class weights: {args.class_weights}")

        # Check for potential issues
        if args.learning_rate > 1e-4:
            logger.warning("⚠️  Learning rate might be too high")
            logger.warning("   Consider reducing to 2e-6 or lower")

        if args.batch_size > 32:
            logger.warning("⚠️  Large batch size might cause memory issues")

        if not args.use_focal_loss and not args.class_weights:
            logger.warning("⚠️  No class balancing strategy")
            logger.warning("   Consider using focal loss or class weights for imbalanced data")

        return True

    except Exception as _:
        logger.error("❌ Training configuration validation failed: {e}")
        return False


def run_training(args):
    """Run the actual training."""
    logger.info("🚀 Starting Vertex AI training...")

    try:
        trainer = EmotionDetectionTrainer(
            model_name=args.model_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            freeze_initial_layers=args.freeze_bert_layers,
            device=None,  # Let it auto-detect
        )

        # Prepare data
        logger.info("📊 Preparing data...")
        trainer.prepare_data(dev_mode=args.dev_mode)

        # Initialize model
        logger.info("🏗️  Initializing model...")
        trainer.initialize_model()

        # Start training
        logger.info("🎯 Starting training...")
        results = trainer.train()

        # Log results
        logger.info("📊 Training Results:")
        for key, value in results.items():
            logger.info("   {key}: {value}")

        return results

    except Exception as _:
        logger.error("❌ Training failed: {e}")
        logger.error("Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main function."""
    logger.info("🚀 SAMO Deep Learning - Vertex AI Training")
    logger.info("=" * 50)

    # Parse arguments
    args = parse_arguments()

    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)

    # Run validation if requested
    if args.validation_mode:
        logger.info("🔍 Running validation mode...")

        validations = []

        if args.check_data_distribution:
            validations.append(("Data Distribution", validate_data_distribution))

        if args.check_model_architecture:
            validations.append(("Model Architecture", validate_model_architecture))

        if args.check_loss_function:
            validations.append(("Loss Function", validate_loss_function))

        if args.check_training_config:
            validations.append(("Training Config", lambda: validate_training_config(args)))

        # Run all validations if none specified
        if not validations:
            validations = [
                ("Data Distribution", validate_data_distribution),
                ("Model Architecture", validate_model_architecture),
                ("Loss Function", validate_loss_function),
                ("Training Config", lambda: validate_training_config(args)),
            ]

        # Run validations
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

            except Exception as _:
                logger.error("❌ {name} ERROR: {e}")
                results[name] = False

        # Summary
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
            logger.info("   Ready for training on Vertex AI")
        else:
            logger.error("\n❌ SOME VALIDATIONS FAILED!")
            logger.error("   Fix issues before training")
            sys.exit(1)

    else:
        # Run training
        logger.info("🎯 Running training mode...")
        results = run_training(args)

        if results:
            logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("📊 Final Results:")
            for key, value in results.items():
                logger.info("   {key}: {value}")
        else:
            logger.error("\n❌ TRAINING FAILED!")
            sys.exit(1)


if __name__ == "__main__":
    main()
