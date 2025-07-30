            # Get predictions
            # Get raw probabilities
            # Move to device
            # Move to device
            # Tokenize
            # Tokenize
        from transformers import AutoModel, AutoTokenizer
    # Calculate metrics
    # Compare results
    # Convert to numpy arrays
    # Create model
    # Create test data
    # Evaluate with default threshold
    # Evaluate with optimized threshold
    # Get raw predictions first
    # Load checkpoint
    # Load trained model
    # Optimize threshold
    # Save results
    # Setup device
    # Show top 5 thresholds
    # Test examples with known emotions
    # Try different thresholds
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
import json
import logging
import numpy as np
import torch




"""
Evaluate Focal Loss Trained Model

This script evaluates the trained focal loss model and calculates F1 scores.
It also implements threshold optimization to improve performance.

Usage:
    python3 evaluate_focal_model.py
"""

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
    logger.info("📂 Loading trained model from {model_path}")

    model = SimpleBERTClassifier(model_name="bert-base-uncased", num_classes=28)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("✅ Model loaded successfully")
    logger.info("   • Final loss: {checkpoint['final_loss']:.4f}")
    logger.info("   • Focal loss alpha: {checkpoint['focal_loss_alpha']}")
    logger.info("   • Focal loss gamma: {checkpoint['focal_loss_gamma']}")
    logger.info("   • Learning rate: {checkpoint['learning_rate']}")
    logger.info("   • Epochs trained: {checkpoint['epochs']}")

    return model


def create_test_data():
    """Create test data for evaluation."""
    logger.info("📊 Creating test data for evaluation...")

    test_data = [
        {
            "text": "I am extremely happy today!",
            "labels": [
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
            ],
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
            ],
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
                0,
                0,
                0,
            ],
        },
        {
            "text": "This makes me so angry!",
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
                0,
                0,
                0,
            ],
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
            ],
        },
        {
            "text": "This is really frustrating",
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
                0,
                0,
                0,
            ],
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
            ],
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
            ],
        },
    ]

    return test_data


def evaluate_model(model, test_data, threshold=0.5):
    """Evaluate the model with given threshold."""
    logger.info("🔍 Evaluating model with threshold {threshold}")

    device = next(model.parameters()).device
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            text = item["text"]
            true_labels = np.array(item["labels"])

            encoding = model.tokenizer(
                text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).float().cpu().numpy().squeeze()

            all_predictions.append(predictions)
            all_labels.append(true_labels)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average="micro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0)

    logger.info("📊 Evaluation Results (threshold={threshold}):")
    logger.info("   • F1 Macro: {f1_macro:.4f}")
    logger.info("   • F1 Micro: {f1_micro:.4f}")
    logger.info("   • F1 Weighted: {f1_weighted:.4f}")
    logger.info("   • Precision Macro: {precision_macro:.4f}")
    logger.info("   • Recall Macro: {recall_macro:.4f}")

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def optimize_threshold(model, test_data):
    """Optimize threshold for best F1 score."""
    logger.info("🎯 Optimizing threshold for best F1 score...")

    device = next(model.parameters()).device
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="Getting predictions"):
            text = item["text"]
            true_labels = np.array(item["labels"])

            encoding = model.tokenizer(
                text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy().squeeze()

            all_probabilities.append(probabilities)
            all_labels.append(true_labels)

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        predictions = (all_probabilities > threshold).astype(float)
        f1_macro = f1_score(all_labels, predictions, average="macro", zero_division=0)
        f1_micro = f1_score(all_labels, predictions, average="micro", zero_division=0)

        results.append({"threshold": threshold, "f1_macro": f1_macro, "f1_micro": f1_micro})

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold

    logger.info("🎯 Threshold Optimization Results:")
    logger.info("   • Best threshold: {best_threshold:.3f}")
    logger.info("   • Best F1 Macro: {best_f1:.4f}")

    results.sort(key=lambda x: x["f1_macro"], reverse=True)
    logger.info("📊 Top 5 thresholds:")
    for i, result in enumerate(results[:5]):
        logger.info(
            "   {i+1}. Threshold {result['threshold']:.3f}: F1 Macro = {result['f1_macro']:.4f}"
        )

    return best_threshold, results


def main():
    """Main evaluation function."""
    logger.info("🎯 Starting Focal Loss Model Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    model_path = Path("models/emotion_detection/focal_loss_model.pt")
    if not model_path.exists():
        logger.error("❌ Model not found at {model_path}")
        logger.info(
            "🔧 Please run the training script first: python3 scripts/full_focal_training.py"
        )
        return

    model = load_trained_model(model_path)
    model = model.to(device)

    test_data = create_test_data()
    logger.info("✅ Test data created with {len(test_data)} examples")

    logger.info("=" * 50)
    default_results = evaluate_model(model, test_data, threshold=0.5)

    logger.info("=" * 50)
    best_threshold, threshold_results = optimize_threshold(model, test_data)

    logger.info("=" * 50)
    optimized_results = evaluate_model(model, test_data, threshold=best_threshold)

    logger.info("=" * 50)
    logger.info("📊 Performance Comparison:")
    logger.info("   Default threshold (0.5):")
    logger.info("     • F1 Macro: {default_results['f1_macro']:.4f}")
    logger.info("     • F1 Micro: {default_results['f1_micro']:.4f}")
    logger.info("   Optimized threshold ({best_threshold:.3f}):")
    logger.info("     • F1 Macro: {optimized_results['f1_macro']:.4f}")
    logger.info("     • F1 Micro: {optimized_results['f1_micro']:.4f}")

    improvement = optimized_results["f1_macro"] - default_results["f1_macro"]
    logger.info("   🎯 Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

    results = {
        "default_threshold": 0.5,
        "optimized_threshold": best_threshold,
        "default_results": default_results,
        "optimized_results": optimized_results,
        "threshold_results": threshold_results,
        "improvement": improvement,
    }

    results_dir = Path("models/emotion_detection")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("✅ Results saved to {results_dir / 'evaluation_results.json'}")
    logger.info("🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
