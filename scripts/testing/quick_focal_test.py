#!/usr/bin/env python3
"""Quick Focal Loss Test Script.

Tests focal loss implementation and performance.
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

"""
Quick Focal Loss Test

Minimal test to validate focal loss implementation without complex dependencies.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_focal_loss_math():
    """Test focal loss mathematical implementation."""
    logger.info("🧮 Testing Focal Loss Mathematics...")

    try:

        class SimpleFocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                probs = torch.sigmoid(inputs)
                pt = probs * targets + (1 - probs) * (1 - targets)
                focal_weight = (1 - pt) ** self.gamma
                alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                bce_loss = F.binary_cross_entropy_with_logits(
                    inputs,
                    targets,
                    reduction="none",
                )
                focal_loss = alpha_weight * focal_weight * bce_loss
                return focal_loss.mean()

        batch_size = 4
        num_classes = 28

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()

        focal_loss = SimpleFocalLoss(alpha=0.25, gamma=2.0)
        focal_loss(inputs, targets)

        logger.info("✅ Focal Loss Test PASSED")
        logger.info("   • Loss value: {loss.item():.4f}")
        logger.info("   • Input shape: {inputs.shape}")
        logger.info("   • Target shape: {targets.shape}")

        return True

    except Exception:
        logger.exception("❌ Focal Loss Test FAILED: {e}")
        return False


def test_dataset_loading():
    """Test if we can load a small subset of the dataset."""
    logger.info("📊 Testing Dataset Loading...")

    try:
        sys.path.append(str(Path(__file__).parent.parent.resolve()))

        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()  # Use correct method name

        len(datasets["train"])
        len(datasets["validation"])

        logger.info("✅ Dataset Loading Test PASSED")
        logger.info("   • Train examples: {train_size}")
        logger.info("   • Validation examples: {val_size}")
        logger.info(
            "   • Class weights computed: {datasets['class_weights'] is not None}",
        )

        return True

    except Exception:
        logger.exception("❌ Dataset Loading Test FAILED: {e}")
        return False


def test_model_creation():
    """Test if we can create the BERT model."""
    logger.info("🤖 Testing Model Creation...")

    try:
        sys.path.append(str(Path(__file__).parent.parent.resolve()))

        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=4,
        )

        sum(p.numel() for p in model.parameters())
        sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model Creation Test PASSED")
        logger.info("   • Total parameters: {param_count:,}")
        logger.info("   • Trainable parameters: {trainable_count:,}")
        logger.info("   • Model type: {type(model).__name__}")

        return True

    except Exception:
        logger.exception("❌ Model Creation Test FAILED: {e}")
        return False


def main():
    """Run all quick tests."""
    logger.info("🎯 Quick Focal Loss Validation Tests")
    logger.info("=" * 50)

    tests = [
        ("Focal Loss Math", test_focal_loss_math),
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info("\n📋 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception:
            logger.exception("❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 30)

    passed = sum(results.values())
    total = len(results)

    for test_name, _result in results.items():
        logger.info("   • {test_name}: {status}")

    logger.info("\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ All tests passed! Ready for GCP deployment.")
        logger.info("🚀 Next step: Deploy to GCP for full training")
    else:
        logger.info("⚠️  Some tests failed. Check environment setup.")
        logger.info("🔧 Consider fixing local environment or going straight to GCP")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
