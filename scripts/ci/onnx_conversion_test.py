#!/usr/bin/env python3
"""
CI ONNX Conversion Test

This script tests ONNX conversion functionality for CI/CD pipeline.
It creates a simple model and tests ONNX conversion without requiring checkpoints.

Usage:
    python scripts/ci/onnx_conversion_test.py

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

def benchmark_pytorch_inference(model, input_ids, attention_mask, num_runs=10):
    """Benchmark PyTorch model inference time."""
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()

    return (end_time - start_time) / num_runs

def test_onnx_conversion():
    """Test ONNX conversion functionality."""
    try:
        device = torch.device("cpu")  # ONNX conversion requires CPU
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Creating BERT emotion classifier...")
        model = SimpleBERTClassifier()
        model.to(device)
        model.eval()

        # Set optimal parameters
        model.set_temperature(1.0)
        model.prediction_threshold = 0.6

        # Create dummy input for ONNX export
        batch_size = 1
        sequence_length = 128
        dummy_input_ids = torch.randint(0, 30522, (batch_size, sequence_length), dtype=torch.long)
        dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)

        # Benchmark PyTorch model
        logger.info("Benchmarking PyTorch model...")
        pytorch_inference_time = benchmark_pytorch_inference(
            model, dummy_input_ids, dummy_attention_mask
        )
        logger.info(f"PyTorch inference time: {pytorch_inference_time*1000:.2f} ms")

        # Test ONNX conversion
        logger.info("Testing ONNX conversion...")
        output_path = Path("/tmp/bert_emotion_classifier_test.onnx")
        
        # Define input names and output names
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        output_names = ["logits"]

        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=12,
            verbose=False,
        )

        # Verify ONNX file was created
        assert output_path.exists(), f"ONNX model file not created: {output_path}"
        logger.info(f"✅ ONNX model created: {output_path}")

        # Test ONNX model loading and inference
        logger.info("Testing ONNX model inference...")
        
        # Check if onnxruntime is available
        try:
            import onnxruntime as ort
            
            # Create ONNX session
            session = ort.InferenceSession(str(output_path))
            
            # Prepare inputs
            ort_inputs = {
                "input_ids": dummy_input_ids.numpy(),
                "attention_mask": dummy_attention_mask.numpy(),
                "token_type_ids": dummy_token_type_ids.numpy(),
            }

            # Test inference
            outputs = session.run(None, ort_inputs)
            logger.info(f"✅ ONNX inference successful, output shape: {outputs[0].shape}")
            
            # Basic validation
            assert outputs[0].shape[0] == batch_size, f"Expected batch size {batch_size}, got {outputs[0].shape[0]}"
            assert outputs[0].shape[1] == 28, f"Expected 28 emotions, got {outputs[0].shape[1]}"
            
            # Benchmark ONNX model
            logger.info("Benchmarking ONNX model...")
            
            # Warm up
            for _ in range(5):
                _ = session.run(None, ort_inputs)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                _ = session.run(None, ort_inputs)
            end_time = time.time()
            
            onnx_inference_time = (end_time - start_time) / 10
            logger.info(f"ONNX inference time: {onnx_inference_time*1000:.2f} ms")

            # Calculate speedup
            speedup = pytorch_inference_time / onnx_inference_time
            logger.info(f"ONNX inference speedup: {speedup:.2f}x")

            # Basic validation
            assert speedup > 0, f"Speedup should be positive: {speedup}"
            
        except ImportError:
            logger.warning("onnxruntime not found. Skipping ONNX benchmarking.")
            logger.info("To install: pip install onnxruntime")

        # Clean up
        output_path.unlink()
        
        logger.info("🎉 All ONNX conversion tests passed!")
        return 0

    except Exception as e:
        logger.error(f"❌ ONNX conversion test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_onnx_conversion())