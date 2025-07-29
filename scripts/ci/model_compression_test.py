#!/usr/bin/env python3
"""
CI Model Compression Test

This script tests model compression functionality for CI/CD pipeline.
It creates a simple model and tests compression without requiring checkpoints.

Usage:
    python scripts/ci/model_compression_test.py

Returns:
    0 if test passes
    1 if test fails
"""

import os
import sys
import torch
import logging
import time
import numpy as np
from pathlib import Path
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
            torch.nn.Linear(256, num_emotions)
        )
        self.temperature = torch.nn.Parameter(torch.ones(1))
        self.prediction_threshold = 0.5
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def set_temperature(self, temperature):
        """Set temperature for calibration."""
        self.temperature.data.fill_(temperature)

def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    temp_file = Path("temp_model.pt")
    torch.save(model.state_dict(), temp_file)
    size_bytes = temp_file.stat().st_size
    temp_file.unlink()
    return size_bytes / (1024 * 1024)  # Convert to MB

def benchmark_inference(model: torch.nn.Module, num_runs: int = 10) -> float:
    """Benchmark model inference time."""
    # Create dummy input (batch_size=1, seq_len=128)
    dummy_input = {
        "input_ids": torch.randint(0, 30522, (1, 128)),
        "attention_mask": torch.ones(1, 128),
    }

    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(**dummy_input)

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**dummy_input)
    end_time = time.time()

    return (end_time - start_time) / num_runs

def test_model_compression():
    """Test model compression functionality."""
    try:
        device = torch.device("cpu")  # Quantization requires CPU
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Creating BERT emotion classifier...")
        model = SimpleBERTClassifier()
        model.to(device)
        model.eval()

        # Set optimal parameters
        model.set_temperature(1.0)
        model.prediction_threshold = 0.6

        # Measure original model size
        original_size = get_model_size(model)
        logger.info(f"Original model size: {original_size:.2f} MB")

        # Benchmark original model
        logger.info("Benchmarking original model...")
        original_inference_time = benchmark_inference(model)
        logger.info(f"Original inference time: {original_inference_time*1000:.2f} ms")

        # Test quantization
        logger.info("Testing dynamic quantization...")
        
        # Quantize model
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Measure quantized model size
        quantized_size = get_model_size(quantized_model)
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        
        # Calculate size reduction
        size_reduction = (1 - quantized_size/original_size) * 100
        logger.info(f"Size reduction: {size_reduction:.1f}%")

        # Benchmark quantized model
        logger.info("Benchmarking quantized model...")
        quantized_inference_time = benchmark_inference(quantized_model)
        logger.info(f"Quantized inference time: {quantized_inference_time*1000:.2f} ms")

        # Calculate speedup
        speedup = original_inference_time / quantized_inference_time
        logger.info(f"Inference speedup: {speedup:.2f}x")

        # Basic validation
        if quantized_size >= original_size:
            raise AssertionError(f"Quantized model should be smaller: {quantized_size} >= {original_size}")
        if size_reduction <= 0:
            raise AssertionError(f"Size reduction should be positive: {size_reduction}")
        if speedup <= 0:
            raise AssertionError(f"Speedup should be positive: {speedup}")

        # Test model saving
        logger.info("Testing model saving...")
        output_path = Path("/tmp/compressed_model_test.pt")
        
        # Save quantized model
        torch.save({
            "model_state_dict": quantized_model.state_dict(),
            "quantized": True,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_percent": size_reduction,
            "speedup": speedup
        }, output_path)
        
        # Verify file was created
        if not output_path.exists():
            raise AssertionError(f"Compressed model file not created: {output_path}")
        logger.info(f"✅ Compressed model saved to {output_path}")

        # Clean up
        output_path.unlink()
        
        logger.info("🎉 All compression tests passed!")
        return 0

    except Exception as e:
        logger.error(f"❌ Compression test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_model_compression())