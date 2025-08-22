#!/usr/bin/env python3
"""
Minimal Test Script

This script provides a minimal test setup for the SAMO-DL project.
"""

import logging
import sys
from pathlib import Path

import torch

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def minimal_test():
    """Run minimal test to verify basic functionality."""
    logger.info("🚀 Starting Minimal Test")

    try:
        # Test model creation
        model, tokenizer = create_bert_emotion_classifier()
        logger.info("✅ Model creation successful")

        # Test basic forward pass
        test_text = "I am feeling happy today!"
        inputs = tokenizer(
            test_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            logger.info(f"✅ Forward pass successful, output shape: {outputs.shape}")

        logger.info("✅ Minimal test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Minimal test failed: {e}")
        raise


if __name__ == "__main__":
    minimal_test()
