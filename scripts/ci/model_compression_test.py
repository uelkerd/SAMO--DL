# Calculate compression ratio
# Create a simple model for testing
# Create dummy input
# Create simple model
# Get compressed model size and performance
# Get original model size and performance
# Simple forward pass for testing
# Test quantization
# Test saving compressed model
# Validate compression
# Add src to path
# Configure logging
#!/usr/bin/env python3
import logging
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

"""
Model Compression Test for CI/CD Pipeline.

This script validates that model compression (quantization) works correctly
without requiring external model checkpoints.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from .validation_utils import ensure


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for testing compression."""

    def __init__(self, num_emotions=28):
        super().__init__()
        self.embedding = nn.Embedding(30522, 768)  # BERT vocab size
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
        )

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        pooled = torch.mean(embeddings, dim=1)  # Simple pooling
        return self.classifier(pooled)


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def benchmark_inference(model, input_tensor, num_runs=100):
    """Benchmark model inference time."""
    model.eval()
    start_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )
    end_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )

    if start_time and end_time:
        start_time.record()
    else:
        start_time = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        avg_time = start_time.elapsed_time(end_time) / num_runs
    else:
        avg_time = 0.1  # Fallback for CPU

    return avg_time


def test_model_compression():
    """Test model compression functionality."""
    try:
        logger.info("📦 Testing model compression...")

        model = SimpleBERTClassifier(num_emotions=28)
        model.eval()

        batch_size = 1
        sequence_length = 128
        dummy_input = torch.randint(0, 30522, (batch_size, sequence_length))

        original_size = get_model_size(model)
        benchmark_inference(model, dummy_input)

        logger.info("Original model size: {original_size:.2f} MB")
        logger.info("Original inference time: {original_time:.2f} ms")

        logger.info("Testing quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        compressed_size = get_model_size(quantized_model)
        benchmark_inference(quantized_model, dummy_input)

        logger.info("Compressed model size: {compressed_size:.2f} MB")
        logger.info("Compressed inference time: {compressed_time:.2f} ms")

        compression_ratio = original_size / compressed_size
        logger.info("Compression ratio: {compression_ratio:.2f}x")

        ensure(
            compressed_size < original_size, "Model should be smaller after compression"
        )
        ensure(compression_ratio > 1.0, "Compression ratio should be greater than 1")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as temp_file:
            torch.save(quantized_model.state_dict(), temp_file.name)
            logger.info("✅ Compressed model saved to {temp_file.name}")

        logger.info("✅ Model compression test passed")
        return True

    except Exception:
        logger.error("❌ Model compression test failed: {e}")
        return False


def main():
    """Run model compression tests."""
    logger.info("🚀 Starting Model Compression Tests...")

    tests = [
        ("Model Compression", test_model_compression),
    ]

    passed = 0
    total = len(tests)

    for _test_name, test_func in tests:
        logger.info("\n{'='*40}")
        logger.info("Running: {test_name}")
        logger.info("{'='*40}")

        if test_func():
            passed += 1
            logger.info("✅ {test_name}: PASSED")
        else:
            logger.error("❌ {test_name}: FAILED")

    logger.info("\n{'='*40}")
    logger.info("Compression Tests Results: {passed}/{total} tests passed")
    logger.info("{'='*40}")

    if passed == total:
        logger.info("🎉 All model compression tests passed!")
        return True
    else:
        logger.error("💥 Some model compression tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
