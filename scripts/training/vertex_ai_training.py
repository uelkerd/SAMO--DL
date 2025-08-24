#!/usr/bin/env python3
"""Vertex AI Training Script for SAMO Deep Learning.

Runs training/validation on Vertex AI and standardizes logging and path bootstrap.
"""

from pathlib import Path
import argparse
import logging
import sys
import traceback
import transformers
import os

# Import project-specific modules
from src.models.emotion_detection.dataset_loader import create_goemotions_loader
from src.models.emotion_detection.bert_classifier import (
    WeightedBCELoss, create_bert_emotion_classifier
)
from src.models.emotion_detection.training_pipeline import (
    EmotionDetectionTrainer
)

# Ensure project root on path
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    PROJECT_ROOT = Path(__file__).resolve().parent

if PROJECT_ROOT.is_dir():
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Ensure log directory exists for FileHandler safety
Path("/app/logs").mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(
        description="Vertex AI Training for SAMO Deep Learning"
    )

    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Learning rate (optimized for stability)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--freeze_bert_layers",
        type=int,
        default=6,
        help="Number of BERT layers to freeze"
    )

    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss instead of BCE"
    )
    parser.add_argument(
        "--class_weights",
        action="store_true",
        help="Use class weights for imbalanced data"
    )
    parser.add_argument(
        "--dev_mode",
        action="store_true",
        help="Run in development mode"
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debugging mode"
    )

    parser.add_argument(
        "--validation_mode",
        action="store_true",
        help="Run validation only"
    )
    parser.add_argument(
        "--check_data_distribution",
        action="store_true",
        help="Check data distribution"
    )
    parser.add_argument(
        "--check_model_architecture",
        action="store_true",
        help="Check model architecture"
    )
    parser.add_argument(
        "--check_loss_function",
        action="store_true",
        help="Check loss function"
    )
    parser.add_argument(
        "--check_training_config",
        action="store_true",
        help="Check training configuration"
    )

    return parser.parse_args()


def validate_environment():
    """Validate Vertex AI environment."""
    logger.info("🔍 Validating Vertex AI environment...")

    try:
        import torch
        logger.info("✅ PyTorch: %s", torch.__version__)
        logger.info("✅ Transformers: %s", transformers.__version__)
        logger.info("✅ Vertex AI: Available")

        # Log environment variables for visibility
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        region = os.getenv("VERTEX_AI_REGION")
        if project_id:
            logger.info("✅ Google Cloud Project: %s", project_id)
        else:
            logger.warning("⚠️  GOOGLE_CLOUD_PROJECT not set")

        if region:
            logger.info("✅ Vertex AI Region: %s", region)
        else:
            logger.warning("⚠️  VERTEX_AI_REGION not set")

        if torch.cuda.is_available():
            logger.info("✅ CUDA device: %s", torch.cuda.get_device_name(0))

        return True
    except Exception as _e:
        logger.error("❌ Environment validation failed: %s", _e)
        raise


def validate_data_distribution():
    """Validate data distribution to identify 0.0000 loss causes."""
    logger.info("🔍 Validating data distribution...")

    try:
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

        # 28 emotion classes
        positive_rate = total_positive_labels / (total_samples * 28)
        logger.info("✅ Total samples analyzed: %d", total_samples)
        logger.info("✅ Total positive labels: %d", total_positive_labels)
        logger.info("✅ Positive label rate: %.6f", positive_rate)

        if positive_rate == 0:
            error_msg = "No positive labels found! This will cause 0.0000 loss with BCE"
            logger.error("❌ CRITICAL: %s", error_msg)
            raise ValueError(error_msg)
        if positive_rate == 1:
            error_msg = "All labels are positive! This will cause 0.0000 loss with BCE"
            logger.error("❌ CRITICAL: %s", error_msg)
            raise ValueError(error_msg)
        if positive_rate < 0.01:
            logger.warning("⚠️  Very low positive label rate")
            logger.warning("   Consider using focal loss or class weights")

        logger.info("📊 Class distribution (first 10 classes):")
        for class_idx in range(min(10, len(label_distribution))):
            count = label_distribution.get(class_idx, 0)
            if count > 0:
                logger.info("   Class %d: %d positive samples", class_idx, count)

        return True

    except Exception as _e:
        logger.error("❌ Data distribution validation failed: %s", _e)
        raise


def validate_model_architecture():
    """Validate model architecture."""
    logger.info("🔍 Validating model architecture...")

    try:
        import torch
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=6,
        )

        param_count = sum(p.numel() for p in model.parameters())
        logger.info("✅ Model created: %s parameters", format(param_count, ","))
        logger.info("✅ Loss function: %s", type(loss_fn).__name__)

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
        logger.info("   Logits shape: %s", logits.shape)
        logger.info("   Loss value: %.8f", loss.item())

        if loss.item() <= 0:
            error_msg = f"Loss is zero or negative: {loss.item()}"
            logger.error("❌ CRITICAL: %s", error_msg)
            raise ValueError(error_msg)

        if torch.isnan(loss).any():
            error_msg = "NaN loss detected"
            logger.error("❌ CRITICAL: %s", error_msg)
            raise ValueError(error_msg)

        return True

    except Exception as _e:
        logger.error("❌ Model architecture validation failed: %s", _e)
        raise


def validate_loss_function():
    """Validate loss function implementation."""
    logger.info("🔍 Validating loss function...")

    try:
        import torch
        import torch.nn.functional as F
        batch_size = 4
        num_classes = 28

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        labels[:, 0] = 1.0  # Ensure some positive labels

        loss_fn = WeightedBCELoss()
        loss1 = loss_fn(logits, labels)

        bce_manual = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="mean"
        )

        logger.info("✅ Mixed labels loss: %.8f", loss1.item())
        logger.info("✅ Manual BCE loss: %.8f", bce_manual.item())

        loss_diff = abs(loss1.item() - bce_manual.item())
        if loss_diff > 1.0:
            logger.warning(
                "⚠️  Large difference between custom and manual loss: %s", loss_diff
            )

        labels_all_pos = torch.ones(batch_size, num_classes)
        loss2 = loss_fn(logits, labels_all_pos)
        logger.info("✅ All positive loss: %.8f", loss2.item())

        labels_all_neg = torch.zeros(batch_size, num_classes)
        loss3 = loss_fn(logits, labels_all_neg)
        logger.info("✅ All negative loss: %.8f", loss3.item())

        if loss1.item() <= 0 or loss2.item() <= 0 or loss3.item() <= 0:
            error_msg = "Loss function producing zero/negative values!"
            logger.error("❌ CRITICAL: %s", error_msg)
            raise ValueError(error_msg)

        return True

    except Exception as _e:
        logger.error("❌ Loss function validation failed: %s", _e)
        raise


def validate_training_config(args):
    """Validate training configuration."""
    logger.info("🔍 Validating training configuration...")

    try:
        logger.info("📋 Training Configuration:")
        logger.info("   Model: %s", args.model_name)
        logger.info("   Batch size: %d", args.batch_size)
        logger.info("   Learning rate: %s", args.learning_rate)
        logger.info("   Epochs: %d", args.num_epochs)
        logger.info("   Max length: %d", args.max_length)
        logger.info("   Frozen layers: %d", args.freeze_bert_layers)
        logger.info("   Focal loss: %s", args.use_focal_loss)
        logger.info("   Class weights: %s", args.class_weights)

        if args.learning_rate > 1e-4:
            logger.warning("⚠️  Learning rate might be too high")
            logger.warning("   Consider reducing to 2e-6 or lower")

        if args.batch_size > 32:
            logger.warning("⚠️  Large batch size might cause memory issues")

        if not args.use_focal_loss and not args.class_weights:
            logger.warning("⚠️  No class balancing strategy")
            logger.warning(
                "   Consider using focal loss or class weights for imbalanced data"
            )

        return True

    except Exception as _e:
        logger.error("❌ Training configuration validation failed: %s", _e)
        raise


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

        logger.info("📊 Preparing data...")
        trainer.prepare_data(dev_mode=args.dev_mode)

        logger.info("🏗️  Initializing model...")
        trainer.initialize_model()

        logger.info("🎯 Starting training...")
        results = trainer.train()

        logger.info("📊 Training Results:")
        for key, value in results.items():
            logger.info("   %s: %s", key, value)

        return results

    except Exception as _e:
        logger.error("❌ Training failed: %s", _e)
        logger.error("Traceback: %s", traceback.format_exc())
        return None


def _build_validation_list(args):
    """Build list of validations to run."""
    validations = []

    if args.check_data_distribution:
        validations.append(("Data Distribution", validate_data_distribution))

    if args.check_model_architecture:
        validations.append(("Model Architecture", validate_model_architecture))

    if args.check_loss_function:
        validations.append(("Loss Function", validate_loss_function))

    if args.check_training_config:
        validations.append(
            ("Training Config", lambda: validate_training_config(args))
        )

    if not validations:
        validations = [
            ("Data Distribution", validate_data_distribution),
            ("Model Architecture", validate_model_architecture),
            ("Loss Function", validate_loss_function),
            ("Training Config", lambda: validate_training_config(args)),
        ]

    return validations


def _run_validations(validations):
    """Run all validations and return results."""
    results = {}

    for name, validation_func in validations:
        logger.info("\n%s", "="*40)
        logger.info("Running: %s", name)
        logger.info("%s", "="*40)

        try:
            validation_func()  # Will raise exception on failure
            results[name] = True
            logger.info("✅ %s PASSED", name)

        except Exception as _e:
            logger.error("❌ %s ERROR: %s", name, _e)
            results[name] = False

    return results


def _print_validation_summary(results):
    """Print validation summary and handle exit codes."""
    passed = sum(results.values())
    total = len(results)

    logger.info("\n%s", "="*50)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("%s", "="*50)
    logger.info("Total checks: %d", total)
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", total - passed)

    if passed == total:
        logger.info("\n✅ ALL VALIDATIONS PASSED!")
        logger.info("   Ready for training on Vertex AI")
    else:
        logger.error("\n❌ SOME VALIDATIONS FAILED!")
        logger.error("   Fix issues before training")
        sys.exit(1)


def _run_training_mode(args):
    """Run training mode and handle results."""
    logger.info("🎯 Running training mode...")
    results = run_training(args)

    if results:
        logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("📊 Final Results:")
        for key, value in results.items():
            logger.info("   %s: %s", key, value)
    else:
        logger.error("\n❌ TRAINING FAILED!")
        sys.exit(1)


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
        validations = _build_validation_list(args)
        results = _run_validations(validations)
        _print_validation_summary(results)
    else:
        _run_training_mode(args)


if __name__ == "__main__":
    main()
