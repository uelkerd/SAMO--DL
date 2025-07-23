#!/usr/bin/env python3
"""BERT Model Loading Test for CircleCI.

This script tests that the BERT emotion detection model can be loaded
and performs basic inference without errors.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from models.emotion_detection.bert_classifier import BertEmotionClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bert_model_loading():
    """Test BERT model initialization and basic inference."""
    try:
        logger.info("🤖 Testing BERT emotion detection model loading...")
        
        # Initialize model
        model = BertEmotionClassifier(
            model_name="bert-base-uncased",
            num_emotions=28,
            device="cpu"  # Use CPU for CI
        )
        
        logger.info(f"✅ Model initialized with {model.count_parameters():,} parameters")
        
        # Test forward pass with dummy input
        dummy_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        attention_mask = torch.ones_like(dummy_input)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, attention_mask=attention_mask)
        
        # Verify output shape
        expected_shape = (2, 28)  # batch_size, num_emotions
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Verify output is probabilities (after sigmoid)
        assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be probabilities [0,1]"
        
        logger.info(f"✅ Forward pass successful - Output shape: {output.shape}")
        logger.info(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BERT model test failed: {e}")
        return False


def main():
    """Run BERT model tests."""
    logger.info("🧪 Starting BERT Model Loading Tests")
    
    success = test_bert_model_loading()
    
    if success:
        logger.info("🎉 All BERT model tests passed!")
        return 0
    else:
        logger.error("❌ BERT model tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 