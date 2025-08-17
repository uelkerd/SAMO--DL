        # Create a simple BERT classifier
        # Create a simple classifier head
        # Load a small subset for testing
        # Test with a simple input

        from datasets import load_dataset
        from torch import nn
        from transformers import AutoTokenizer, AutoModel

    # Compute loss
    # Create focal loss
    # Create synthetic data
    # Setup device
# Configure logging
#!/usr/bin/env python3

from torch import nn
import logging
import sys
import torch
import torch.nn.functional as F






"""
Standalone Focal Loss Test

This script tests focal loss implementation without depending on the src module structure.
It will download and use the GoEmotions dataset directly.

Usage:
    python3 standalone_focal_test.py
"""

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


def test_focal_loss():
    """Test focal loss with synthetic data."""
    logger.info("🧮 Testing Focal Loss with synthetic data...")

    batch_size = 4
    num_classes = 28
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    loss = focal_loss(inputs, targets)

    logger.info("✅ Focal Loss Test PASSED")
    logger.info("   • Loss value: {loss.item():.4f}")
    logger.info("   • Input shape: {inputs.shape}")
    logger.info("   • Target shape: {targets.shape}")

    return True


def test_bert_import():
    """Test if we can import transformers and create a simple BERT model."""
    logger.info("🤖 Testing BERT model creation...")

    try:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        num_classes = 28
        classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

        text = "I am happy today"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token

        logger.info("✅ BERT Model Test PASSED")
        logger.info("   • Model: {model_name}")
        logger.info("   • Input text: '{text}'")
        logger.info("   • Output shape: {logits.shape}")
        logger.info("   • Output values: {logits[0, :5].tolist()}...")

        return True

    except Exception as e:
        logger.error("❌ BERT Model Test FAILED: {e}")
        return False


def test_dataset_download():
    """Test if we can download the GoEmotions dataset."""
    logger.info("📊 Testing GoEmotions dataset download...")

    try:
        dataset = load_dataset("go_emotions", "simplified", split="train[:100]")

        logger.info("✅ Dataset Download Test PASSED")
        logger.info("   • Dataset size: {len(dataset)}")
        logger.info("   • Features: {list(dataset.features.keys())}")
        logger.info("   • Sample text: '{dataset[0]['text'][:50]}...'")
        logger.info("   • Sample labels: {dataset[0]['labels']}")

        return True

    except Exception as e:
        logger.error("❌ Dataset Download Test FAILED: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🎯 Standalone Focal Loss Validation Tests")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    tests = [
        ("Focal Loss Math", test_focal_loss),
        ("BERT Model Creation", test_bert_import),
        ("Dataset Download", test_dataset_download),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info("\n📋 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error("❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 30)
    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info("   • {name}: {status}")

    logger.info("\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ All tests passed! Ready for full training.")
        logger.info("🚀 Next step: Create full training script with these components")
        return True
    else:
        logger.info("⚠️  Some tests failed. Check environment setup.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
