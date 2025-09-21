#!/usr/bin/env python3
"""Validation and Training Script.

Validates environment and starts model training.
"""

import logging
import sys
import time
from pathlib import Path

from pre_training_validation import PreTrainingValidator

from src.models.emotion_detection.training_pipeline import train_emotion_detection_model

"""
Validate and Train Script for SAMO Deep Learning.

This script runs comprehensive pre-training validation and only starts training
if all critical checks pass. This prevents wasting 4+ hours on failed training.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_session.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_pre_training_validation():
    """Run comprehensive pre-training validation."""
    logger.info("🔍 Running pre-training validation...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        validator = PreTrainingValidator()
        all_passed = validator.run_all_validations()
        validator.generate_report()

        return all_passed, validator.critical_issues, validator.warnings

    except Exception:
        logger.exception("❌ Pre-training validation failed: {e}")
        return False, ["Validation error: {e}"], []


def run_training_with_debugging():
    """Run training with comprehensive debugging enabled."""
    logger.info("🚀 Starting training with debugging...")

    try:
        config = {
            "model_name": "bert-base-uncased",
            "cache_dir": "./data/cache",
            "output_dir": "./models/emotion_detection",
            "batch_size": 8,  # Smaller batch for debugging
            "learning_rate": 2e-6,  # Reduced learning rate
            "num_epochs": 2,  # Fewer epochs for debugging
            "dev_mode": True,
            "debug_mode": True,
        }

        logger.info("📋 Training Configuration:")
        for _key, _value in config.items():
            logger.info("   {key}: {value}")

        start_time = time.time()
        results = train_emotion_detection_model(**config)
        time.time() - start_time

        logger.info("✅ Training completed in {training_time/60:.1f} minutes!")
        logger.info("📊 Final results: {results}")

        return True, results

    except Exception:
        logger.exception("❌ Training failed: {e}")
        logger.exception("Traceback: {traceback.format_exc()}")
        return False, None


def main():
    """Main function that validates and then trains."""
    logger.info("🚀 SAMO Deep Learning - Validate and Train")
    logger.info("=" * 60)

    logger.info("\n📋 STEP 1: Pre-Training Validation")
    logger.info("-" * 40)

    validation_passed, critical_issues, warnings = run_pre_training_validation()

    if not validation_passed:
        logger.error("\n❌ VALIDATION FAILED - Training blocked!")
        logger.error("Critical issues found:")
        for _i, _issue in enumerate(critical_issues, 1):
            logger.error("   {i}. {issue}")

        if warnings:
            logger.warning("\nWarnings (non-blocking):")
            for _i, _warning in enumerate(warnings, 1):
                logger.warning("   {i}. {warning}")

        logger.error(
            "\n🔧 Please fix all critical issues before running training again.",
        )
        return False

    logger.info("\n✅ VALIDATION PASSED!")

    if warnings:
        logger.warning("\n⚠️  {len(warnings)} warnings detected:")
        for _i, _warning in enumerate(warnings, 1):
            logger.warning("   {i}. {warning}")

        logger.warning("\nConsider addressing these warnings before proceeding.")

    logger.info("\n📋 STEP 2: Training Confirmation")
    logger.info("-" * 40)
    logger.info("Training will take approximately 4+ hours.")
    logger.info("Configuration:")
    logger.info("   • Model: BERT-base-uncased")
    logger.info("   • Batch size: 8")
    logger.info("   • Learning rate: 2e-6")
    logger.info("   • Epochs: 2")
    logger.info("   • Debug mode: Enabled")

    try:
        response = input("\n🤔 Proceed with training? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            logger.info("❌ Training cancelled by user.")
            return False
    except KeyboardInterrupt:
        logger.info("\n❌ Training cancelled by user.")
        return False

    logger.info("\n📋 STEP 3: Training Execution")
    logger.info("-" * 40)

    training_success, results = run_training_with_debugging()

    if training_success:
        logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("📊 Results summary:")
        if results:
            for _key, _value in results.items():
                logger.info("   {key}: {value}")

        logger.info("\n📁 Check the following files for details:")
        logger.info("   • training_session.log - Complete training log")
        logger.info("   • debug_training.log - Debug information")
        logger.info("   • models/emotion_detection/ - Model checkpoints")

        return True
    logger.error("\n❌ TRAINING FAILED!")
    logger.error("Check training_session.log for detailed error information.")
    return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
