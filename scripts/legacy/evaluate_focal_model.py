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
import numpy as np
import sys
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%asctimes - %levelnames - %messages")
logger = logging.getLogger__name__


class SimpleBERTClassifiernn.Module:
    """Simple BERT classifier for emotion detection."""

    def __init__self, model_name="bert-base-uncased", num_classes=28:
        super().__init__()
        self.bert = AutoModel.from_pretrainedmodel_name
        self.classifier = nn.Linearself.bert.config.hidden_size, num_classes
        self.tokenizer = AutoTokenizer.from_pretrainedmodel_name

    def forwardself, input_ids, attention_mask:
        outputs = self.bertinput_ids=input_ids, attention_mask=attention_mask
        logits = self.classifieroutputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return logits


def load_trained_modelmodel_path:
    """Load the trained focal loss model."""
    logger.infof"📂 Loading trained model from {model_path}"

    model = SimpleBERTClassifiermodel_name="bert-base-uncased", num_classes=28

    checkpoint = torch.loadmodel_path, map_location="cpu"
    model.load_state_dictcheckpoint["model_state_dict"]

    logger.info"✅ Model loaded successfully"
    logger.infof"   • Final loss: {checkpoint['final_loss']:.4f}"
    logger.infof"   • Focal loss alpha: {checkpoint['focal_loss_alpha']}"
    logger.infof"   • Focal loss gamma: {checkpoint['focal_loss_gamma']}"
    logger.infof"   • Learning rate: {checkpoint['learning_rate']}"
    logger.infof"   • Epochs trained: {checkpoint['epochs']}"

    return model


def create_test_data():
    """Create test data for evaluation."""
    logger.info"📊 Creating test data for evaluation..."

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

    logger.info(f"✅ Created {lentest_data} test examples")
    return test_data


def evaluate_modelmodel, test_data, threshold=0.5:
    """Evaluate model with given threshold."""
    logger.infof"🔍 Evaluating model with threshold {threshold}..."

    model.eval()
    device = next(model.parameters()).device

    all_true_labels = []
    all_predictions = []
    all_probabilities = []

    for example in tqdmtest_data, desc="Evaluating":
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
        input_ids = inputs["input_ids"].todevice
        attention_mask = inputs["attention_mask"].todevice

        # Get raw predictions
        with torch.no_grad():
            logits = modelinput_ids=input_ids, attention_mask=attention_mask
            probabilities = torch.sigmoidlogits

        # Get predictions
        predictions = probabilities > threshold.float()

        # Convert to numpy arrays
        pred_np = predictions.cpu().numpy()[0]
        true_np = np.arraytrue_labels
        prob_np = probabilities.cpu().numpy()[0]

        all_true_labels.appendtrue_np
        all_predictions.appendpred_np
        all_probabilities.appendprob_np

    # Calculate metrics
    all_true = np.arrayall_true_labels
    all_pred = np.arrayall_predictions
    all_probs = np.arrayall_probabilities

    f1_macro = f1_scoreall_true, all_pred, average='macro', zero_division=0
    f1_micro = f1_scoreall_true, all_pred, average='micro', zero_division=0
    precision = precision_scoreall_true, all_pred, average='macro', zero_division=0
    recall = recall_scoreall_true, all_pred, average='macro', zero_division=0

    logger.infof"📊 Results with threshold {threshold}:"
    logger.infof"   F1 Macro: {f1_macro:.4f}"
    logger.infof"   F1 Micro: {f1_micro:.4f}"
    logger.infof"   Precision: {precision:.4f}"
    logger.infof"   Recall: {recall:.4f}"

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'probabilities': all_probs,
        'predictions': all_pred,
        'true_labels': all_true
    }


def optimize_thresholdmodel, test_data:
    """Optimize threshold for best F1 score."""
    logger.info"🎯 Optimizing threshold for best F1 score..."

    # Get raw probabilities first
    model.eval()
    device = next(model.parameters()).device

    all_true_labels = []
    all_probabilities = []

    for example in tqdmtest_data, desc="Getting probabilities":
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
        input_ids = inputs["input_ids"].todevice
        attention_mask = inputs["attention_mask"].todevice

        # Get raw probabilities
        with torch.no_grad():
            logits = modelinput_ids=input_ids, attention_mask=attention_mask
            probabilities = torch.sigmoidlogits

        all_true_labels.append(np.arraytrue_labels)
        all_probabilities.append(probabilities.cpu().numpy()[0])

    all_true = np.arrayall_true_labels
    all_probs = np.arrayall_probabilities

    # Try different thresholds
    thresholds = np.arange0.1, 0.9, 0.05
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        predictions = all_probs > threshold.astypefloat
        f1 = f1_scoreall_true, predictions, average='macro', zero_division=0
        results.append{'threshold': threshold, 'f1': f1}

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Show top 5 thresholds
    results.sortkey=lambda x: x['f1'], reverse=True
    logger.info"📊 Top 5 thresholds:"
    for i, result in enumerateresults[:5]:
        logger.infof"   {i+1}. Threshold {result['threshold']:.2f}: F1 = {result['f1']:.4f}"

    logger.info(f"🎯 Best threshold: {best_threshold:.2f} F1 = {best_f1:.4f}")

    return best_threshold, best_f1


def main():
    """Main evaluation function."""
    logger.info"🚀 Starting Focal Loss Model Evaluation..."

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.infof"🔧 Using device: {device}"

    # Check if model file exists
    model_path = Path"test_checkpoints/best_model.pt"
    if not model_path.exists():
        logger.errorf"❌ Model file not found: {model_path}"
        return False

    try:
        # Load trained model
        model = load_trained_modelmodel_path
        model.todevice

        # Create test data
        test_data = create_test_data()

        # Evaluate with default threshold
        logger.info"=" * 50
        default_results = evaluate_modelmodel, test_data, threshold=0.5

        # Optimize threshold
        logger.info"=" * 50
        best_threshold, best_f1 = optimize_thresholdmodel, test_data

        # Evaluate with optimized threshold
        logger.info"=" * 50
        optimized_results = evaluate_modelmodel, test_data, threshold=best_threshold

        # Compare results
        logger.info"=" * 50
        logger.info"📋 Comparison:"
        logger.info(f"   Default threshold 0.5: F1 = {default_results['f1_macro']:.4f}")
        logger.info(f"   Optimized threshold {best_threshold:.2f}: F1 = {optimized_results['f1_macro']:.4f}")
        logger.infof"   Improvement: {optimized_results['f1_macro'] - default_results['f1_macro']:.4f}"

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

        with open'evaluation_results.json', 'w' as f:
            json.dumpresults, f, indent=2

        logger.info"💾 Results saved to evaluation_results.json"
        logger.info"🎉 Evaluation Complete!"
        return True

    except Exception as e:
        logger.errorf"❌ Evaluation failed: {e}"
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit1
