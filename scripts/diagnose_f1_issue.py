import numpy as np

#!/usr/bin/env python3
"""
Diagnose F1 Score Issue

This script investigates why F1 scores are 0% despite good training loss.
It checks label formats, prediction outputs, and evaluation logic.

Usage:
    python3 diagnose_f1_issue.py
"""

import logging
import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for emotion detection."""

    def __init__(self, model_name="bert-base-uncased", num_classes=28):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
        return logits


def load_trained_model(model_path):
    """Load the trained model."""
    logger.info("📂 Loading trained model from {model_path}")

    model = SimpleBERTClassifier(model_name="bert-base-uncased", num_classes=28)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("✅ Model loaded successfully")
    return model


def create_test_data():
    """Create test data with proper emotion labels."""
    logger.info("📊 Creating test data with proper emotion labels...")

    # GoEmotions emotion names (28 classes)
    emotion_names = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grie",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relie",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]

    # Test examples with proper emotion labels
    test_data = [
        {
            "text": "I am extremely happy today!",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # joy
        },
        {
            "text": "This is absolutely disgusting!",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # disgust
        },
        {
            "text": "I'm feeling really sad and depressed",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ],  # sadness
        },
        {
            "text": "This makes me so angry!",
            "labels": [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # anger
        },
        {
            "text": "I love this so much!",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # love
        },
        {
            "text": "This is really frustrating",
            "labels": [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # annoyance
        },
        {
            "text": "I'm confused about this situation",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # confusion
        },
        {
            "text": "This is amazing and wonderful!",
            "labels": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # excitement
        },
    ]

    return test_data, emotion_names


def diagnose_predictions(model, test_data, device):
    """Diagnose prediction outputs and label formats."""
    logger.info("🔍 Diagnosing predictions and labels...")

    model.eval()

    for i, item in enumerate(test_data):
        text = item["text"]
        true_labels = np.array(item["labels"])

        logger.info("\n📝 Example {i+1}: '{text}'")
        logger.info("   True labels: {true_labels}")
        logger.info("   Sum of labels: {np.sum(true_labels)}")
        logger.info("   Non-zero indices: {np.where(true_labels > 0)[0]}")

        # Tokenize
        encoding = model.tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )

        # Get predictions
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            raw_logits = outputs.cpu().numpy().squeeze()
            probabilities = torch.sigmoid(outputs).cpu().numpy().squeeze()

            # Test different thresholds
            for threshold in [0.1, 0.3, 0.5, 0.7]:
                predictions = (probabilities > threshold).astype(float)

                logger.info("   Threshold {threshold}:")
                logger.info("     Raw logits (first 5): {raw_logits[:5]}")
                logger.info("     Probabilities (first 5): {probabilities[:5]}")
                logger.info("     Predictions: {predictions}")
                logger.info("     Sum of predictions: {np.sum(predictions)}")
                logger.info("     Non-zero indices: {np.where(predictions > 0)[0]}")

                # Calculate metrics
                f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
                precision = precision_score(
                    true_labels, predictions, average="macro", zero_division=0
                )
                recall = recall_score(true_labels, predictions, average="macro", zero_division=0)

                logger.info("     F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


def test_evaluation_logic():
    """Test the evaluation logic with synthetic data."""
    logger.info("🧮 Testing evaluation logic with synthetic data...")

    # Create synthetic data
    true_labels = np.array(
        [
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # class 0
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # class 1
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # class 2
        ]
    )

    # Perfect predictions
    perfect_predictions = true_labels.copy()

    # Random predictions
    rng = np.random.default_rng()
    random_predictions = rng.integers(0, 2, size=true_labels.shape)

    # All zeros predictions
    zero_predictions = np.zeros_like(true_labels)

    # All ones predictions
    ones_predictions = np.ones_like(true_labels)

    test_cases = [
        ("Perfect", perfect_predictions),
        ("Random", random_predictions),
        ("All Zeros", zero_predictions),
        ("All Ones", ones_predictions),
    ]

    for name, predictions in test_cases:
        f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
        f1_micro = f1_score(true_labels, predictions, average="micro", zero_division=0)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=0)

        logger.info("   {name}:")
        logger.info("     F1 Macro: {f1_macro:.4f}")
        logger.info("     F1 Micro: {f1_micro:.4f}")
        logger.info("     Precision: {precision:.4f}")
        logger.info("     Recall: {recall:.4f}")


def main():
    """Main diagnostic function."""
    logger.info("🔍 Starting F1 Score Issue Diagnosis")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    # Test evaluation logic first
    logger.info("=" * 60)
    test_evaluation_logic()

    # Load trained model
    logger.info("=" * 60)
    model_path = Path("models/emotion_detection/full_scale_focal_loss_model.pt")
    if not model_path.exists():
        logger.error("❌ Model not found at {model_path}")
        return

    model = load_trained_model(model_path)
    model = model.to(device)

    # Create test data
    test_data, emotion_names = create_test_data()
    logger.info("✅ Test data created with {len(test_data)} examples")
    logger.info("✅ Emotion names: {emotion_names}")

    # Diagnose predictions
    logger.info("=" * 60)
    diagnose_predictions(model, test_data, device)

    logger.info("=" * 60)
    logger.info("🎯 Diagnosis completed!")
    logger.info("📋 Check the output above to identify the issue with F1 scores")


if __name__ == "__main__":
    main()
