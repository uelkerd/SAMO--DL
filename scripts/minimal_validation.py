#!/usr/bin/env python3
"""
Minimal Validation for Core Components

Quick validation of essential components before GCP deployment.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test basic imports."""
    logger.info("📦 Testing Basic Imports...")

    try:
        import torch

        logger.info(f"   ✅ PyTorch: {torch.__version__}")

        import transformers

        logger.info(f"   ✅ Transformers: {transformers.__version__}")

        import numpy as np

        logger.info(f"   ✅ NumPy: {np.__version__}")

        import sklearn

        logger.info(f"   ✅ Scikit-learn: {sklearn.__version__}")

        logger.info("✅ Basic Imports: PASSED")
        return True

    except ImportError as e:
        logger.error(f"❌ Basic Imports: FAILED - {e}")
        return False


def test_focal_loss():
    """Test focal loss implementation."""
    logger.info("🧮 Testing Focal Loss...")

    try:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                probs = torch.sigmoid(inputs)
                pt = probs * targets + (1 - probs) * (1 - targets)
                focal_weight = (1 - pt) ** self.gamma
                alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
                focal_loss = alpha_weight * focal_weight * bce_loss
                return focal_loss.mean()

        # Test with dummy data
        inputs = torch.randn(4, 28)
        targets = torch.randint(0, 2, (4, 28)).float()

        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss(inputs, targets)

        logger.info(f"   ✅ Focal Loss: {loss.item():.4f}")
        logger.info("✅ Focal Loss: PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Focal Loss: FAILED - {e}")
        return False


def test_file_structure():
    """Test that required files exist."""
    logger.info("📁 Testing File Structure...")

    required_files = [
        "src/models/emotion_detection/bert_classifier.py",
        "src/models/emotion_detection/dataset_loader.py",
        "src/models/emotion_detection/training_pipeline.py",
        "scripts/focal_loss_training.py",
        "scripts/threshold_optimization.py",
        "docs/gcp_deployment_guide.md",
    ]

    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"   ✅ {file_path}")
        else:
            logger.error(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)

    if missing_files:
        logger.error(f"❌ File Structure: FAILED - {len(missing_files)} files missing")
        return False
    else:
        logger.info(f"✅ File Structure: PASSED - All {len(required_files)} files found")
        return True


def test_model_creation():
    """Test model creation without dataset loading."""
    logger.info("🤖 Testing Model Creation...")

    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent.parent.resolve()))

        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

        # Create model
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
        )

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"   ✅ Model created: {param_count:,} total params")
        logger.info(f"   ✅ Trainable: {trainable_count:,} params")
        logger.info("✅ Model Creation: PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Model Creation: FAILED - {e}")
        return False


def main():
    """Run minimal validations."""
    logger.info("🎯 Minimal Validation for GCP Deployment")
    logger.info("=" * 50)

    validations = [
        ("Basic Imports", test_imports),
        ("Focal Loss", test_focal_loss),
        ("File Structure", test_file_structure),
        ("Model Creation", test_model_creation),
    ]

    results = {}

    for name, validation_func in validations:
        logger.info(f"\n📋 Running {name}...")
        try:
            results[name] = validation_func()
        except Exception as e:
            logger.error(f"❌ {name} failed with exception: {e}")
            results[name] = False

    # Summary
    logger.info("\n📊 Validation Results:")
    logger.info("=" * 30)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   • {name}: {status}")

    logger.info(f"\n🎯 Overall: {passed}/{total} validations passed")

    if passed >= 3:
        logger.info("✅ Ready for GCP deployment!")
        logger.info("🚀 Core components are working correctly.")
        logger.info("📋 Next: Follow docs/gcp_deployment_guide.md")
        return True
    else:
        logger.info("⚠️ Some validations failed.")
        logger.info("🔧 Check environment setup before GCP deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
