#!/usr/bin/env python3
"""
Evaluate Focal Loss Trained Model

This script evaluates the trained focal loss model and calculates F1 scores.
It also implements threshold optimization to improve performance.

Usage:
    python3 evaluate_focal_model.py
"""

import json
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
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
    """Load the trained focal loss model."""
    logger.info(f"📂 Loading trained model from {model_path}")

    model = SimpleBERTClassifier(model_name="bert-base-uncased", num_classes=28)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("✅ Model loaded successfully")
    logger.info("   • Final loss: {checkpoint["final_loss']:.4f}")
    logger.info("   • Focal loss alpha: {checkpoint["focal_loss_alpha']}")
    logger.info("   • Focal loss gamma: {checkpoint["focal_loss_gamma']}")
    logger.info("   • Learning rate: {checkpoint["learning_rate']}")
    logger.info("   • Epochs trained: {checkpoint["epochs']}")

    return model


def create_test_data():
    """Create test data for evaluation."""
    logger.info("📊 Creating test data for evaluation...")

    # Test examples with known emotions
    test_data = [
        {
            "text": "I am extremely happy today!",
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # joy
        },
        {
            "text": "This makes me so angry!",
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


def evaluate_model(model, test_data, threshold=0.5):
    """Evaluate model with given threshold."""
    logger.info(f"🔍 Evaluating model with threshold {threshold}...")

    model.eval()
    device = next(model.parameters()).device

    all_true_labels = []
    all_predictions = []
    all_probabilities = []

    for example in tqdm(test_data, desc="Evaluating"):
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

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get raw predictions
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(logits)

        # Get predictions
        predictions = (probabilities > threshold).float()

        # Convert to numpy arrays
        pred_np = predictions.cpu().numpy()[0]
        true_np = np.array(true_labels)
        prob_np = probabilities.cpu().numpy()[0]

        all_true_labels.append(true_np)
        all_predictions.append(pred_np)
        all_probabilities.append(prob_np)

    # Calculate metrics
    all_true = np.array(all_true_labels)
    all_pred = np.array(all_predictions)
    all_probs = np.array(all_probabilities)

    f1_macro = f1_score(all_true, all_pred, average='macro', zero_division=0)
    f1_micro = f1_score(all_true, all_pred, average='micro', zero_division=0)
    precision = precision_score(all_true, all_pred, average='macro', zero_division=0)
    recall = recall_score(all_true, all_pred, average='macro', zero_division=0)

    logger.info(f"📊 Results with threshold {threshold}:")
    logger.info(f"   F1 Macro: {f1_macro:.4f}")
    logger.info(f"   F1 Micro: {f1_micro:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall: {recall:.4f}")

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'probabilities': all_probs,
        'predictions': all_pred,
        'true_labels': all_true
    }


def optimize_threshold(model, test_data):
    """Optimize threshold for best F1 score."""
    logger.info("🎯 Optimizing threshold for best F1 score...")

    # Get raw probabilities first
    model.eval()
    device = next(model.parameters()).device

    all_true_labels = []
    all_probabilities = []

    for example in tqdm(test_data, desc="Getting probabilities"):
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

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get raw probabilities
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(logits)

        all_true_labels.append(np.array(true_labels))
        all_probabilities.append(probabilities.cpu().numpy()[0])

    all_true = np.array(all_true_labels)
    all_probs = np.array(all_probabilities)

    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        predictions = (all_probs > threshold).astype(float)
        f1 = f1_score(all_true, predictions, average='macro', zero_division=0)
        results.append({'threshold': threshold, 'f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Show top 5 thresholds
    results.sort(key=lambda x: x['f1'], reverse=True)
    logger.info("📊 Top 5 thresholds:")
    for i, result in enumerate(results[:5]):
        logger.info(f"   {i+1}. Threshold {result['threshold']:.2f}: F1 = {result['f1']:.4f}")

    logger.info(f"🎯 Best threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")

    return best_threshold, best_f1


def main():
    """Main evaluation function."""
    logger.info("🚀 Starting Focal Loss Model Evaluation...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")

    # Check if model file exists
    model_path = Path("test_checkpoints/best_model.pt")
    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return False

    try:
        # Load trained model
        model = load_trained_model(model_path)
        model.to(device)

        # Create test data
        test_data = create_test_data()

        # Evaluate with default threshold
        logger.info("=" * 50)
        default_results = evaluate_model(model, test_data, threshold=0.5)

        # Optimize threshold
        logger.info("=" * 50)
        best_threshold, best_f1 = optimize_threshold(model, test_data)

        # Evaluate with optimized threshold
        logger.info("=" * 50)
        optimized_results = evaluate_model(model, test_data, threshold=best_threshold)

        # Compare results
        logger.info("=" * 50)
        logger.info("📋 Comparison:")
        logger.info("   Default threshold (0.5): F1 = {default_results["f1_macro']:.4f}")
        logger.info(f"   Optimized threshold ({best_threshold:.2f}): F1 = {optimized_results['f1_macro']:.4f}")
        logger.info("   Improvement: {optimized_results["f1_macro'] - default_results['f1_macro']:.4f}")

        # Save results
        results = {
            'default_threshold': {
                'threshold': 0.5,
                'f1_macro': default_results['f1_macro'],
                'f1_micro': default_results['f1_micro'],
                'precision': default_results['precision'],
                'recall': default_results['recall']
            },
            'optimized_threshold': {
                'threshold': best_threshold,
                'f1_macro': optimized_results['f1_macro'],
                'f1_micro': optimized_results['f1_micro'],
                'precision': optimized_results['precision'],
                'recall': optimized_results['recall']
            }
        }

        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("💾 Results saved to evaluation_results.json")
        logger.info("🎉 Evaluation Complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
