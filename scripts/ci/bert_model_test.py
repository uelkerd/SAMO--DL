import sys

#!/usr/bin/env python3
import logging
from pathlib import Path

# Add src to path
import torch
from models.emotion_detection.bert_classifier import BERTEmotionClassifier

# Configure logging

"""
BERT Model Loading Test for CI/CD Pipeline.

This script validates that the BERT emotion detection model
can be loaded and initialized correctly.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bert_model_loading():
    """Test BERT model initialization and basic inference."""
    try:
        logger.info("🤖 Testing BERT emotion detection model loading...")

        # Set device
        device = torch.device("cpu")  # Use CPU for CI

        # Initialize model - removed 'device' parameter as it's not in the constructor
        model = BERTEmotionClassifier(
            model_name="bert-base-uncased",
            num_emotions=28,
        )

        # Move model to device after initialization
        model.to(device)

        logger.info("✅ Model initialized with {model.count_parameters():,} parameters")

        # Test model forward pass with dummy data
        batch_size = 2
        seq_length = 32

        # Create dummy input tensors and move to device
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        logger.info("✅ Forward pass successful, output shape: {outputs.shape}")

        # Verify output dimensions
        expected_shape = (batch_size, 28)  # 28 emotions
        if outputs.shape != expected_shape:
            raise ValueError("Expected output shape {expected_shape}, got {outputs.shape}")

        logger.info("✅ Output shape validation passed")

        # Test that outputs are reasonable (not NaN, finite)
        if torch.isnan(outputs).any():
            raise ValueError("Model outputs contain NaN values")

        if not torch.isfinite(outputs).all():
            raise ValueError("Model outputs contain infinite values")

        logger.info("✅ Output sanity checks passed")

        return True

    except Exception as _:
        logger.error("❌ BERT model test failed: {e}")
        return False


def main():
    """Run BERT model tests."""
    logger.info("🚀 Starting BERT Model Tests...")

    if test_bert_model_loading():
        logger.info("🎉 All BERT model tests passed!")
        return True
    else:
        logger.error("💥 BERT model tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
