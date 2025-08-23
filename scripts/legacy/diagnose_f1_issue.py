#!/usr/bin/env python3
"""Diagnose F1 Score Issue.

This script investigates why F1 scores are 0% despite good training loss.
It checks label formats, prediction outputs, and evaluation logic.

Usage:
    python3 diagnose_f1_issue.py
"""

import logging
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for emotion detection."""

    def __init__(self, model_name="bert-base-uncased", num_classes=28):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
        return logits


def load_trained_model(model_path):
    """Load the trained model."""
    logger.info(f"📂 Loading trained model from {model_path}")

    model = SimpleBERTClassifier(model_name="bert-base-uncased", num_classes=28)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("✅ Model loaded successfully")
    return model


def create_test_data():
    """Create test data with proper emotion labels."""
    logger.info("📊 Creating test data with proper emotion labels...")

    # Create test examples with proper emotion labels
    test_data = [
        {
            "text": "I am so happy today!",
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # joy
        },
        {
            "text": "This makes me very angry!",
            "labels": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # anger
        },
        {
            "text": "I feel sad and disappointed.",
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # disappointment, sadness
        },
        {
            "text": "This is amazing and exciting!",
            "labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # admiration, excitement
        },
        {
            "text": "I'm neutral about this.",
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # neutral
        }
    ]

    logger.info(f"✅ Created {len(test_data)} test examples")
    return test_data


def diagnose_predictions(model, test_data, device):
    """Diagnose predictions and evaluation logic."""
    logger.info("🔍 Diagnosing predictions...")

    model.eval()
    results = []

    for i, example in enumerate(test_data):
        text = example["text"]
        true_labels = example["labels"]

        # Tokenize
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get predictions
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(logits)

        # Convert to numpy
        pred_np = predictions.cpu().numpy()[0]
        true_np = np.array(true_labels)

        # Calculate metrics
        f1_macro = f1_score(true_np, pred_np > 0.5, average='macro', zero_division=0)
        f1_micro = f1_score(true_np, pred_np > 0.5, average='micro', zero_division=0)
        precision = precision_score(true_np, pred_np > 0.5, average='macro', zero_division=0)
        recall = recall_score(true_np, pred_np > 0.5, average='macro', zero_division=0)

        results.append({
            "text": text,
            "true_labels": true_labels,
            "predictions": pred_np.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision": precision,
            "recall": recall
        })

        logger.info(f"📊 Example {i+1}:")
        logger.info(f"   Text: {text}")
        logger.info(f"   True labels: {true_labels}")
        logger.info(f"   Predictions: {pred_np.tolist()}")
        logger.info(f"   F1 Macro: {f1_macro:.4f}")
        logger.info(f"   F1 Micro: {f1_micro:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")

    return results


def test_evaluation_logic():
    """Test evaluation logic with synthetic data."""
    logger.info("🧪 Testing evaluation logic with synthetic data...")

    # Create synthetic data
    num_samples = 100
    num_classes = 28
    rng = np.random.default_rng()

    # Perfect predictions
    perfect_true = rng.integers(0, 2, (num_samples, num_classes))
    perfect_pred = perfect_true.copy()
    perfect_f1 = f1_score(perfect_true, perfect_pred, average='macro', zero_division=0)
    logger.info(f"✅ Perfect predictions F1: {perfect_f1:.4f}")

    # Random predictions
    random_pred = rng.integers(0, 2, (num_samples, num_classes))
    random_f1 = f1_score(perfect_true, random_pred, average='macro', zero_division=0)
    logger.info(f"📊 Random predictions F1: {random_f1:.4f}")

    # All ones predictions
    all_ones_pred = np.ones((num_samples, num_classes))
    all_ones_f1 = f1_score(perfect_true, all_ones_pred, average='macro', zero_division=0)
    logger.info(f"📊 All ones predictions F1: {all_ones_f1:.4f}")

    # All zeros predictions
    all_zeros_pred = np.zeros((num_samples, num_classes))
    all_zeros_f1 = f1_score(perfect_true, all_zeros_pred, average='macro', zero_division=0)
    logger.info(f"📊 All zeros predictions F1: {all_zeros_f1:.4f}")

    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        threshold_pred = (perfect_pred > threshold).astype(int)
        threshold_f1 = f1_score(perfect_true, threshold_pred, average='macro', zero_division=0)
        logger.info(f"📊 Threshold {threshold} F1: {threshold_f1:.4f}")

    return True


def main():
    """Main function."""
    logger.info("🚀 Starting F1 Score Diagnosis...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")

    # Test evaluation logic first
    if not test_evaluation_logic():
        logger.error("❌ Evaluation logic test failed")
        return False

    # Check if model file exists
    model_path = Path("test_checkpoints/best_model.pt")
    if not model_path.exists():
        logger.warning(f"⚠️ Model file not found: {model_path}")
        logger.info("📊 Running diagnosis with synthetic data only")
        return True

    try:
        # Load trained model
        model = load_trained_model(model_path)
        model.to(device)

        # Create test data
        test_data = create_test_data()

        # Diagnose predictions
        results = diagnose_predictions(model, test_data, device)

        # Summary
        avg_f1_macro = np.mean([r["f1_macro"] for r in results])
        avg_f1_micro = np.mean([r["f1_micro"] for r in results])
        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])

        logger.info("📋 Summary:")
        logger.info(f"   Average F1 Macro: {avg_f1_macro:.4f}")
        logger.info(f"   Average F1 Micro: {avg_f1_micro:.4f}")
        logger.info(f"   Average Precision: {avg_precision:.4f}")
        logger.info(f"   Average Recall: {avg_recall:.4f}")

        if avg_f1_macro < 0.1:
            logger.warning("⚠️ Very low F1 scores detected!")
            logger.info("   Possible issues:")
            logger.info("   - Label format mismatch")
            logger.info("   - Threshold too high/low")
            logger.info("   - Model not trained properly")
            logger.info("   - Evaluation logic error")

        logger.info("🎉 F1 Score Diagnosis Complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
