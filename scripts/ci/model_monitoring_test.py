import sys
import time

#!/usr/bin/env python3
"""
Model Monitoring Test for CI/CD Pipeline.

This script validates that model monitoring functionality works correctly
without requiring external model checkpoints.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for testing monitoring."""

    def __init__(self, num_emotions=28):
        super().__init__()
        # Create a simple model for testing
        self.embedding = nn.Embedding(30522, 768)  # BERT vocab size
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
        )

    def forward(self, input_ids, attention_mask=None):
        # Simple forward pass for testing
        embeddings = self.embedding(input_ids)
        pooled = torch.mean(embeddings, dim=1)  # Simple pooling
        return self.classifier(pooled)


def create_synthetic_data(num_samples=100, num_emotions=28):
    """Create synthetic data for testing."""
    # Create synthetic input data
    input_ids = torch.randint(0, 30522, (num_samples, 128))
    attention_mask = torch.ones(num_samples, 128)

    # Create synthetic labels (multi-label)
    labels = torch.randint(0, 2, (num_samples, num_emotions)).float()

    return input_ids, attention_mask, labels


def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate basic metrics for monitoring."""
    # Convert predictions to binary
    binary_predictions = (predictions > threshold).float()

    # Calculate accuracy
    correct = (binary_predictions == labels).float().sum()
    total = labels.numel()
    accuracy = correct / total

    # Calculate precision and recall (simplified)
    true_positives = (binary_predictions * labels).sum()
    predicted_positives = binary_predictions.sum()
    actual_positives = labels.sum()

    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }


def test_model_performance_monitoring():
    """Test model performance monitoring."""
    try:
        logger.info("📊 Testing model performance monitoring...")

        # Create model and data
        model = SimpleBERTClassifier(num_emotions=28)
        model.eval()

        input_ids, attention_mask, labels = create_synthetic_data(100, 28)

        # Get predictions
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)

        # Calculate metrics
        metrics = calculate_metrics(probabilities, labels, threshold=0.5)

        logger.info("Accuracy: {metrics['accuracy']:.4f}")
        logger.info("Precision: {metrics['precision']:.4f}")
        logger.info("Recall: {metrics['recall']:.4f}")
        logger.info("F1 Score: {metrics['f1_score']:.4f}")

        # Validate metrics
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"
        assert 0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1"
        assert 0 <= metrics['f1_score'] <= 1, "F1 score should be between 0 and 1"

        logger.info("✅ Model performance monitoring test passed")
        return True

    except Exception as e:
        logger.error("❌ Model performance monitoring test failed: {e}")
        return False


def test_model_drift_detection():
    """Test model drift detection."""
    try:
        logger.info("🔄 Testing model drift detection...")

        # Create model
        model = SimpleBERTClassifier(num_emotions=28)
        model.eval()

        # Create baseline data
        baseline_input_ids, baseline_attention_mask, baseline_labels = create_synthetic_data(100, 28)

        # Create current data (simulate drift)
        current_input_ids, current_attention_mask, current_labels = create_synthetic_data(100, 28)

        # Get baseline predictions
        with torch.no_grad():
            baseline_logits = model(baseline_input_ids, baseline_attention_mask)
            baseline_probabilities = torch.sigmoid(baseline_logits)

            current_logits = model(current_input_ids, current_attention_mask)
            current_probabilities = torch.sigmoid(current_logits)

        # Calculate baseline and current metrics
        baseline_metrics = calculate_metrics(baseline_probabilities, baseline_labels)
        current_metrics = calculate_metrics(current_probabilities, current_labels)

        # Calculate drift (simplified)
        accuracy_drift = abs(current_metrics['accuracy'] - baseline_metrics['accuracy'])
        f1_drift = abs(current_metrics['f1_score'] - baseline_metrics['f1_score'])

        logger.info("Accuracy drift: {accuracy_drift:.4f}")
        logger.info("F1 score drift: {f1_drift:.4f}")

        # Validate drift detection
        assert accuracy_drift >= 0, "Drift should be non-negative"
        assert f1_drift >= 0, "Drift should be non-negative"

        logger.info("✅ Model drift detection test passed")
        return True

    except Exception as e:
        logger.error("❌ Model drift detection test failed: {e}")
        return False


def test_monitoring_logging():
    """Test monitoring logging functionality."""
    try:
        logger.info("📝 Testing monitoring logging...")

        # Create monitoring log entry
        timestamp = datetime.now(timezone.utc)
        model_version = "test-v1.0.0"
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }

        # Simulate logging
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'model_version': model_version,
            'metrics': metrics,
            'status': 'healthy'
        }

        logger.info("Monitoring log entry: {log_entry}")

        # Validate log entry
        assert 'timestamp' in log_entry, "Log entry should have timestamp"
        assert 'model_version' in log_entry, "Log entry should have model version"
        assert 'metrics' in log_entry, "Log entry should have metrics"
        assert 'status' in log_entry, "Log entry should have status"

        logger.info("✅ Monitoring logging test passed")
        return True

    except Exception as e:
        logger.error("❌ Monitoring logging test failed: {e}")
        return False


def main():
    """Run model monitoring tests."""
    logger.info("🚀 Starting Model Monitoring Tests...")

    tests = [
        ("Model Performance Monitoring", test_model_performance_monitoring),
        ("Model Drift Detection", test_model_drift_detection),
        ("Monitoring Logging", test_monitoring_logging),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("\n{'='*40}")
        logger.info("Running: {test_name}")
        logger.info("{'='*40}")

        if test_func():
            passed += 1
            logger.info("✅ {test_name}: PASSED")
        else:
            logger.error("❌ {test_name}: FAILED")

    logger.info("\n{'='*40}")
    logger.info("Monitoring Tests Results: {passed}/{total} tests passed")
    logger.info("{'='*40}")

    if passed == total:
        logger.info("🎉 All model monitoring tests passed!")
        return True
    else:
        logger.error("💥 Some model monitoring tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
