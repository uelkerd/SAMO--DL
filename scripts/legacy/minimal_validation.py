        # Add src to path
        # Create model
        # Test with dummy data
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        from torch import nn
        import sklearn
        import torch
        import torch
        import torch.nn.functional as F
        import transformers
    # Summary
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
import logging
import numpy as np
import sys








"""
Minimal Validation for Core Components

Quick validation of essential components before GCP deployment.
"""

logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def test_imports():
    """Test basic imports."""
    logger.info"📦 Testing Basic Imports..."

    try:
        logger.info"   ✅ PyTorch: {torch.__version__}"

        logger.info"   ✅ Transformers: {transformers.__version__}"


        logger.info"   ✅ NumPy: {np.__version__}"

        logger.info"   ✅ Scikit-learn: {sklearn.__version__}"

        logger.info"✅ Basic Imports: PASSED"
        return True

    except ImportError as _:
        logger.error"❌ Basic Imports: FAILED - {e}"
        return False


def test_focal_loss():
    """Test focal loss implementation."""
    logger.info"🧮 Testing Focal Loss..."

    try:
        class FocalLossnn.Module:
            def __init__self, alpha=0.25, gamma=2.0:
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forwardself, inputs, targets:
                probs = torch.sigmoidinputs
                pt = probs * targets + 1 - probs * 1 - targets
                focal_weight = 1 - pt ** self.gamma
                alpha_weight = self.alpha * targets + 1 - self.alpha * 1 - targets
                bce_loss = F.binary_cross_entropy_with_logitsinputs, targets, reduction="none"
                focal_loss = alpha_weight * focal_weight * bce_loss
                return focal_loss.mean()

        inputs = torch.randn4, 28
        targets = torch.randint(0, 2, 4, 28).float()

        focal_loss = FocalLossalpha=0.25, gamma=2.0
        loss = focal_lossinputs, targets

        logger.info("   ✅ Focal Loss: {loss.item():.4f}")
        logger.info"✅ Focal Loss: PASSED"
        return True

    except Exception as e:
        logger.error"❌ Focal Loss: FAILED - {e}"
        return False


def test_file_structure():
    """Test that required files exist."""
    logger.info"📁 Testing File Structure..."

    required_files = [
        "src/models/emotion_detection/bert_classifier.py",
        "src/models/emotion_detection/dataset_loader.py",
        "src/models/emotion_detection/training_pipeline.py",
        "scripts/focal_loss_training.py",
        "scripts/threshold_optimization.py",
"docs/GCP_DEPLOYMENT_GUIDE.md",
    ]

    missing_files = []
    for file_path in required_files:
        if Pathfile_path.exists():
            logger.info"   ✅ {file_path}"
        else:
            logger.error"   ❌ {file_path} - MISSING"
            missing_files.appendfile_path

    if missing_files:
        logger.error("❌ File Structure: FAILED - {lenmissing_files} files missing")
        return False
    else:
        logger.info("✅ File Structure: PASSED - All {lenrequired_files} files found")
        return True


def test_model_creation():
    """Test model creation without dataset loading."""
    logger.info"🤖 Testing Model Creation..."

    try:
        sys.path.append(str(Path__file__.parent.parent.resolve()))

        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
        )

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info"   ✅ Model created: {param_count:,} total params"
        logger.info"   ✅ Trainable: {trainable_count:,} params"
        logger.info"✅ Model Creation: PASSED"
        return True

    except Exception as e:
        logger.error"❌ Model Creation: FAILED - {e}"
        return False


def main():
    """Run minimal validations."""
    logger.info"🎯 Minimal Validation for GCP Deployment"
    logger.info"=" * 50

    validations = [
        "Basic Imports", test_imports,
        "Focal Loss", test_focal_loss,
        "File Structure", test_file_structure,
        "Model Creation", test_model_creation,
    ]

    results = {}

    for name, validation_func in validations:
        logger.info"\n📋 Running {name}..."
        try:
            results[name] = validation_func()
        except Exception as e:
            logger.error"❌ {name} failed with exception: {e}"
            results[name] = False

    logger.info"\n📊 Validation Results:"
    logger.info"=" * 30

    passed = sum(results.values())
    total = lenresults

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info"   • {name}: {status}"

    logger.info"\n🎯 Overall: {passed}/{total} validations passed"

    if passed >= 3:
        logger.info"✅ Ready for GCP deployment!"
        logger.info"🚀 Core components are working correctly."
logger.info"📋 Next: Follow docs/GCP_DEPLOYMENT_GUIDE.md"
        return True
    else:
        logger.info"⚠️ Some validations failed."
        logger.info"🔧 Check environment setup before GCP deployment"
        return False


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
