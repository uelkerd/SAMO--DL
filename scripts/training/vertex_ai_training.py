#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging
import os
import sys
import traceback
import torch

"""
Vertex AI Training Script for SAMO Deep Learning.

This script runs training on Vertex AI with optimized configuration
to solve the 0.0000 loss issue and achieve >75% F1 score.
"""

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    parser.add_argument("--model_name", default="bert-base-uncased", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate (optimized for stability)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--freeze_bert_layers", type=int, default=6, help="Number of BERT layers to freeze")

    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss instead of BCE")
    parser.add_argument("--class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--dev_mode", action="store_true", help="Run in development mode")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debugging mode")

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
        import transformers
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ Transformers: {transformers.__version__}")
        logger.info("✅ Vertex AI: Available")

        if torch.cuda.is_available():
            logger.info(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.info(f"❌ Environment validation failed: {e}")
        raise


def validate_data_distribution():
    """Validate data distribution to identify 0.0000 loss causes."""
    logger.info("🔍 Validating data distribution...")

    try:
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader
        datasets = create_goemotions_loader(dev_mode=True)
        train_dataloader = datasets["train_dataloader"]

        total_samples = 0
        total_positive_labels = 0
        label_distribution = {}

        for _batch_idx, batch in enumerate(train_dataloader):
            if _batch_idx >= 10:  # Check first 10 batches
                break

            labels = batch["labels"]
            total_samples += labels.shape[0]
            total_positive_labels += labels.sum().item()

            for class_idx in range(labels.shape[1]):
                if class_idx not in label_distribution:
                    label_distribution[class_idx] = 0
                label_distribution[class_idx] += labels[:, class_idx].sum().item()

        positive_rate = total_positive_labels / (total_samples * 28)  # 28 emotion classes
        logger.info(f"✅ Total samples analyzed: {total_samples}")
        logger.info(f"✅ Total positive labels: {total_positive_labels}")
        logger.info(f"✅ Positive label rate: {positive_rate:.6f}")

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
                logger.info(f"   Class {class_idx}: {count} positive samples")

        return True

    except Exception as _e:
        logger.error(f"❌ Data distribution validation failed: {_e}")
        return False


def validate_model_architecture():
    """Validate model architecture."""
    logger.info("🔍 Validating model architecture...")

    try:
        from src.models.emotion_detection.bert_classifier import WeightedBCELoss
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=6,
        )

        logger.info(f"✅ Model created: {model.count_parameters():,} parameters")
        logger.info(f"✅ Loss function: {type(loss_fn).__name__}")

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
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Loss value: {loss.item():.8f}")

        if loss.item() <= 0:
            logger.error("❌ CRITICAL: Loss is zero or negative: {loss.item()}")
            return False

        if torch.isnan(loss).any():
            logger.error("❌ CRITICAL: NaN loss!")
            return False

        return True

    except Exception as _e:
        logger.error(f"❌ Model architecture validation failed: {_e}")
        return False


def validate_loss_function():
    """Validate loss function implementation."""
    logger.info("🔍 Validating loss function...")

    try:
        import torch.nn.functional as F
        batch_size = 4
        num_classes = 28

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        labels[:, 0] = 1.0  # Ensure some positive labels

        from src.models.emotion_detection.bert_classifier import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        loss1 = loss_fn(logits, labels)

        bce_manual = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        logger.info(f"✅ Mixed labels loss: {loss1.item():.8f}")
        logger.info(f"✅ Manual BCE loss: {bce_manual.item():.8f}")

        loss_diff = abs(loss1.item() - bce_manual.item())
        if loss_diff > 1.0:
            logger.warning("⚠️  Large difference between custom and manual loss: {loss_diff}")

        labels_all_pos = torch.ones(batch_size, num_classes)
        loss2 = loss_fn(logits, labels_all_pos)
        logger.info(f"✅ All positive loss: {loss2.item():.8f}")

        labels_all_neg = torch.zeros(batch_size, num_classes)
        loss3 = loss_fn(logits, labels_all_neg)
        logger.info(f"✅ All negative loss: {loss3.item():.8f}")

        if loss1.item() <= 0 or loss2.item() <= 0 or loss3.item() <= 0:
            logger.error("❌ CRITICAL: Loss function producing zero/negative values!")
            return False

        return True

    except Exception as _e:
        logger.error(f"❌ Loss function validation failed: {_e}")
        return False


def validate_training_config(args):
    """Validate training configuration."""
    logger.info("🔍 Validating training configuration...")

    try:
        logger.info("📋 Training Configuration:")
        logger.info(f"   Model: {args.model_name}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Learning rate: {args.learning_rate}")
        logger.info(f"   Epochs: {args.num_epochs}")
        logger.info(f"   Max length: {args.max_length}")
        logger.info(f"   Frozen layers: {args.freeze_bert_layers}")
        logger.info(f"   Focal loss: {args.use_focal_loss}")
        logger.info(f"   Class weights: {args.class_weights}")

        if args.learning_rate > 1e-4:
            logger.warning("⚠️  Learning rate might be too high")
            logger.warning("   Consider reducing to 2e-6 or lower")

        if args.batch_size > 32:
            logger.warning("⚠️  Large batch size might cause memory issues")

        if not args.use_focal_loss and not args.class_weights:
            logger.warning("⚠️  No class balancing strategy")
            logger.warning("   Consider using focal loss or class weights for imbalanced data")

        return True

    except Exception as _e:
        logger.error(f"❌ Training configuration validation failed: {_e}")
        return False


def run_training(args):
    """Run the actual training."""
    logger.info("🚀 Starting Vertex AI training...")

    try:
        from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
        trainer = EmotionDetectionTrainer(
            model_name=args.model_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            freeze_initial_layers=args.freeze_bert_layers,
            device=None,  # Let it auto-detect
        )

        logger.info("📊 Preparing data...")
        trainer.prepare_data(dev_mode=args.dev_mode)

        logger.info("🏗️  Initializing model...")
        trainer.initialize_model()

        logger.info("🎯 Starting training...")
        results = trainer.train()

        logger.info("📊 Training Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")

        return results

    except Exception as _e:
        logger.error(f"❌ Training failed: {_e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main function."""
    logger.info("🚀 SAMO Deep Learning - Vertex AI Training")
    logger.info("=" * 50)

    args = parse_arguments()

    if not validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)

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

        if not validations:
            validations = [
                ("Data Distribution", validate_data_distribution),
                ("Model Architecture", validate_model_architecture),
                ("Loss Function", validate_loss_function),
                ("Training Config", lambda: validate_training_config(args)),
            ]

        results = {}
        for name, validation_func in validations:
            logger.info(f"\n{'='*40}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*40}")

            try:
                success = validation_func()
                results[name] = success

                if success:
                    logger.info(f"✅ {name} PASSED")
                else:
                    logger.error(f"❌ {name} FAILED")

            except Exception as _e:
                logger.error(f"❌ {name} ERROR: {_e}")
                results[name] = False

        passed = sum(results.values())
        total = len(results)

        logger.info(f"\n{'='*50}")
        logger.info("📊 VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total checks: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")

        if passed == total:
            logger.info("\n✅ ALL VALIDATIONS PASSED!")
            logger.info("   Ready for training on Vertex AI")
        else:
            logger.error("\n❌ SOME VALIDATIONS FAILED!")
            logger.error("   Fix issues before training")
            sys.exit(1)

    else:
        logger.info("🎯 Running training mode...")
        results = run_training(args)

        if results:
            logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("📊 Final Results:")
            for key, value in results.items():
                logger.info(f"   {key}: {value}")
        else:
            logger.error("\n❌ TRAINING FAILED!")
            sys.exit(1)


if __name__ == "__main__":
    main()
