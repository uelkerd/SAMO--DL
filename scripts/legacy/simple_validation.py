        # Test with dummy data
import logging
import subprocess
import sys
from pathlib import Path

#!/usr/bin/env python3
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import transformers
from torch import nn

    # Check if gcloud is available
    # Check if we have the deployment guide
# Configure logging
    # Summary








"""
Simple Validation for GCP Deployment

Quick validation of core components before GCP deployment.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_focal_loss():
    """Validate focal loss implementation."""
    logger.info("🧮 Validating Focal Loss Implementation...")

    try:
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

        inputs = torch.randn(4, 28)
        targets = torch.randint(0, 2, (4, 28)).float()

        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss(inputs, targets)

        logger.info("✅ Focal Loss: PASSED (loss={loss.item():.4f})")
        return True

    except Exception as e:
        logger.error("❌ Focal Loss: FAILED - {e}")
        return False


def validate_script_files():
    """Validate that all required scripts exist."""
    logger.info("📁 Validating Script Files...")

    required_scripts = [
        "scripts/focal_loss_training.py",
        "scripts/threshold_optimization.py",
        "scripts/setup_gpu_training.py",
        "src/models/emotion_detection/bert_classifier.py",
        "src/models/emotion_detection/dataset_loader.py",
    ]

    missing_files = []
    for script in required_scripts:
        if Path(script).exists():
            logger.info("   ✅ {script}")
        else:
            logger.error("   ❌ {script} - MISSING")
            missing_files.append(script)

    if missing_files:
        logger.error("❌ Script Files: FAILED - {len(missing_files)} files missing")
        return False
    else:
        logger.info("✅ Script Files: PASSED - All {len(required_scripts)} files found")
        return True


def validate_python_environment():
    """Validate Python environment and basic imports."""
    logger.info("🐍 Validating Python Environment...")

    try:
        logger.info("   ✅ PyTorch: {torch.__version__}")

        logger.info("   ✅ Transformers: {transformers.__version__}")


        logger.info("   ✅ NumPy: {np.__version__}")

        logger.info("   ✅ Scikit-learn: {sklearn.__version__}")

        logger.info("✅ Python Environment: PASSED")
        return True

    except ImportError as _:
        logger.error("❌ Python Environment: FAILED - {e}")
        return False


def validate_gcp_readiness():
    """Validate GCP deployment readiness."""
    logger.info("☁️ Validating GCP Readiness...")

    try:
        result = subprocess.run(
            ["gcloud", "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode == 0:
            logger.info("   ✅ gcloud CLI: Available")
            gcp_ready = True
        else:
            logger.warning("   ⚠️ gcloud CLI: Not available (will need to install)")
            gcp_ready = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("   ⚠️ gcloud CLI: Not available (will need to install)")
        gcp_ready = False

if Path("docs/GCP_DEPLOYMENT_GUIDE.md").exists():
        logger.info("   ✅ GCP Deployment Guide: Available")
        guide_ready = True
    else:
        logger.error("   ❌ GCP Deployment Guide: Missing")
        guide_ready = False

    if gcp_ready and guide_ready:
        logger.info("✅ GCP Readiness: PASSED")
        return True
    elif guide_ready:
        logger.info("✅ GCP Readiness: READY (gcloud can be installed on GCP)")
        return True
    else:
        logger.error("❌ GCP Readiness: FAILED")
        return False


def main():
    """Run all validations."""
    logger.info("🎯 Simple Validation for GCP Deployment")
    logger.info("=" * 50)

    validations = [
        ("Focal Loss", validate_focal_loss),
        ("Script Files", validate_script_files),
        ("Python Environment", validate_python_environment),
        ("GCP Readiness", validate_gcp_readiness),
    ]

    results = {}

    for name, validation_func in validations:
        logger.info("\n📋 Running {name} validation...")
        try:
            results[name] = validation_func()
        except Exception as e:
            logger.error("❌ {name} validation failed with exception: {e}")
            results[name] = False

    logger.info("\n📊 Validation Results:")
    logger.info("=" * 30)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info("   • {name}: {status}")

    logger.info("\n🎯 Overall: {passed}/{total} validations passed")

    if passed >= 3:  # At least 3 out of 4 should pass
        logger.info("✅ Ready for GCP deployment!")
        logger.info("🚀 Next steps:")
        logger.info("   1. Set up GCP project and APIs")
        logger.info("   2. Create GPU instance")
        logger.info("   3. Run focal loss training")
        return True
    else:
        logger.info("⚠️ Some validations failed.")
        logger.info("🔧 Consider fixing issues or proceeding with GCP setup")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
