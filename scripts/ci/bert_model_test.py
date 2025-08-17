#!/usr/bin/env python3
"""
BERT Model Loading Test for CI/CD Pipeline.

This script validates that the BERT emotion detection model
can be loaded and initialized correctly.
"""

import logging
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

# Test imports
try:
    from models.emotion_detection.bert_classifier import BERTEmotionClassifier
except ImportError:
    # Fallback for different import paths
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def test_bert_model_loading():
    """Test BERT model initialization and basic inference."""
    try:
        logger.info"🤖 Testing BERT emotion detection model loading..."

        device = torch.device"cpu"  # Use CPU for CI

        # Initialize model - removed 'device' parameter as it's not in the constructor
        model = BERTEmotionClassifier(
            model_name="bert-base-uncased",
            num_emotions=28,
        )

        # Move model to device after initialization
        model.todevice

        logger.info(f"✅ Model initialized with {model.count_parameters():,} parameters")

        batch_size = 2
        seq_length = 32

        # Create dummy input tensors and move to device
        input_ids = torch.randint(0, 1000, batch_size, seq_length).todevice
        attention_mask = torch.onesbatch_size, seq_length.todevice

        # Test model forward pass with dummy data
        model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = modelinput_ids, attention_mask

        logger.infof"✅ Forward pass successful, output shape: {outputs.shape}"

        # Verify output dimensions
        expected_shape = batch_size, 28  # 28 emotions
        if outputs.shape != expected_shape:
            raise ValueErrorf"Expected output shape {expected_shape}, got {outputs.shape}"

        logger.info"✅ Output shape validation passed"

        # Test that outputs are reasonable not NaN, finite
        if torch.isnanoutputs.any():
            raise ValueError"Model outputs contain NaN values"

        if not torch.isfiniteoutputs.all():
            raise ValueError"Model outputs contain infinite values"

        logger.info"✅ Output sanity checks passed"

        return True

    except Exception as e:
        logger.errorf"❌ BERT model test failed: {e}"
        return False


def main():
    """Run BERT model tests."""
    logger.info"🚀 Starting BERT Model Tests..."

    if test_bert_model_loading():
        logger.info"🎉 All BERT model tests passed!"
        return True
    else:
        logger.error"💥 BERT model tests failed!"
        return False


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
