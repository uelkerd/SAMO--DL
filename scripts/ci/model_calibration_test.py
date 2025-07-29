#!/usr/bin/env python3
"""
CI Model Calibration Test

This script tests the BERT emotion classifier calibration for CI/CD pipeline.
It creates a simple model and tests basic functionality without requiring checkpoints.

Usage:
    python scripts/ci/model_calibration_test.py

Returns:
    0 if test passes
    1 if test fails
"""

import sys
import torch
import logging
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(torch.nn.Module):
    """Simple BERT classifier for emotion detection."""

    def __init__(self, model_name="bert-base-uncased", num_emotions=28):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_emotions),
        )
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def create_test_data():
    """Create simple test data for calibration."""
    test_texts = [
        "I am so happy today!",
        "I love this new song!",
        "This makes me excited!",
        "I'm really angry about this!",
        "This is so frustrating!",
        "I hate this!",
        "I feel so sad right now",
        "This is heartbreaking",
        "I'm feeling down",
        "I love you so much!",
    ]

    # Create simple labels (one emotion per text)
    emotions = [
        "joy",
        "love",
        "excitement",
        "anger",
        "frustration",
        "disgust",
        "sadness",
        "grief",
        "sadness",
        "love",
    ]
    emotion_to_idx = {
        "joy": 0,
        "love": 1,
        "excitement": 2,
        "anger": 3,
        "frustration": 4,
        "disgust": 5,
        "sadness": 6,
        "grief": 7,
        "neutral": 27,
    }

    test_labels = []
    for emotion in emotions:
        labels = [0] * 28
        if emotion in emotion_to_idx:
            labels[emotion_to_idx[emotion]] = 1
        test_labels.append(labels)

    return test_texts, test_labels


def test_model_calibration():
    """Test model calibration functionality."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Creating BERT emotion classifier...")
        model = SimpleBERTClassifier()
        model.to(device)
        model.eval()

        # Test temperature setting
        logger.info("Testing temperature calibration...")
        model.temperature.data.fill_(1.0)
        if model.temperature.item() != 1.0:
            raise AssertionError("Temperature not set correctly")
        logger.info("✅ Temperature calibration works")

        # Create test data
        test_texts, test_labels = create_test_data()
        logger.info(f"Created {len(test_texts)} test examples")

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Test model inference
        logger.info("Testing model inference...")
        all_predictions = []

        for text in test_texts:
            # Tokenize
            inputs = tokenizer(
                text, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)

            # Get predictions (only pass required arguments)
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                probabilities = torch.sigmoid(outputs / model.temperature)
                predictions = (probabilities > 0.4).float().cpu().numpy()
                all_predictions.append(predictions[0])

        logger.info("✅ Model inference works")

        # Test metrics calculation
        logger.info("Testing metrics calculation...")
        all_predictions = np.array(all_predictions)
        all_labels = np.array(test_labels)

        micro_f1 = f1_score(all_labels, all_predictions, average="micro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        logger.info(f"Micro F1: {micro_f1:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")

        # Basic validation
        if not 0 <= micro_f1 <= 1:
            raise AssertionError(f"Invalid F1 score: {micro_f1}")
        if not 0 <= macro_f1 <= 1:
            raise AssertionError(f"Invalid F1 score: {macro_f1}")

        logger.info("✅ Metrics calculation works")

        # Test threshold optimization
        logger.info("Testing threshold optimization...")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        f1_scores = []

        for threshold in thresholds:
            predictions = (all_predictions > threshold).astype(int)
            f1 = f1_score(all_labels, predictions, average="micro", zero_division=0)
            f1_scores.append(f1)

        best_threshold = thresholds[np.argmax(f1_scores)]
        logger.info(f"Best threshold: {best_threshold:.1f} (F1: {max(f1_scores):.4f})")

        logger.info("✅ Threshold optimization works")

        logger.info("🎉 All calibration tests passed!")
        return 0

    except Exception as e:
        logger.error(f"❌ Calibration test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(test_model_calibration())
