#!/usr/bin/env python3
"""
Simple Temperature Test Script

This script tests temperature scaling on the emotion detection model.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier, evaluate_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test data for temperature scaling."""
    logger.info("Creating test data...")

    test_texts = [
        "I am feeling happy today!",
        "This makes me sad.",
        "I'm really angry about this!",
        "I'm scared of what might happen.",
        "I feel great about everything!",
    ]

    test_labels = [
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
        [1, 0, 0, 0],  # joy
    ]

    return test_texts, test_labels


def simple_temperature_test():
    """Run simple temperature scaling test."""
    logger.info("🚀 Starting Simple Temperature Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model and tokenizer
    model, tokenizer = create_bert_emotion_classifier()
    model.to(device)

    # Create test data
    test_texts, test_labels = create_test_data()

    # Test different temperatures
    temperatures = [0.5, 1.0, 1.5, 2.0]
    
    for temp in temperatures:
        logger.info(f"📊 Testing temperature: {temp}")
        
        # Set model temperature
        model.temperature = temp
        
        # Evaluate model
        try:
            results = evaluate_emotion_classifier(
                model=model,
                tokenizer=tokenizer,
                texts=test_texts,
                labels=test_labels,
                device=device
            )
            
            logger.info(f"   Temperature {temp}: F1 = {results.get('f1_score', 'N/A'):.4f}")
            
        except Exception as e:
            logger.warning(f"   Temperature {temp}: Error - {e}")

    logger.info("✅ Simple temperature test completed!")


if __name__ == "__main__":
    simple_temperature_test()
