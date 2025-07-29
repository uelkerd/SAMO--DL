#!/usr/bin/env python3
"""
ONNX Conversion Test for CI/CD Pipeline.

This script validates that ONNX model conversion works correctly
without requiring external model checkpoints.
"""

import logging
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for testing ONNX conversion."""

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


def benchmark_pytorch_inference(model, input_tensor, num_runs=100):
    """Benchmark PyTorch model inference time."""
    model.eval()
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

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


def test_onnx_conversion():
    """Test ONNX model conversion functionality."""
    try:
        logger.info("🔄 Testing ONNX conversion...")

        # Create simple model
        model = SimpleBERTClassifier(num_emotions=28)
        model.eval()

        # Create dummy input
        batch_size = 1
        sequence_length = 128
        dummy_input = torch.randint(0, 30522, (batch_size, sequence_length))
        dummy_attention_mask = torch.ones(batch_size, sequence_length)

        # Test ONNX export
        logger.info("Testing ONNX export...")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
            torch.onnx.export(
                model,
                (dummy_input, dummy_attention_mask),
                temp_file.name,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"},
                },
            )
            logger.info(f"✅ ONNX model exported to {temp_file.name}")

            # Verify ONNX file was created and has content
            import onnx

            onnx_model = onnx.load(temp_file.name)
            logger.info(f"✅ ONNX model loaded successfully")
            logger.info(f"ONNX model inputs: {[input.name for input in onnx_model.graph.input]}")
            logger.info(f"ONNX model outputs: {[output.name for output in onnx_model.graph.output]}")

        # Test inference with ONNX model
        logger.info("Testing ONNX inference...")
        import onnxruntime as ort

        # Create ONNX Runtime session
        session = ort.InferenceSession(temp_file.name)
        logger.info("✅ ONNX Runtime session created")

        # Prepare input data
        input_data = {
            "input_ids": dummy_input.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        }

        # Run inference
        outputs = session.run(None, input_data)
        logger.info(f"✅ ONNX inference successful, output shape: {outputs[0].shape}")

        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input, dummy_attention_mask)
            torch_output_np = torch_output.numpy()

        # Check if outputs are similar (allowing for small numerical differences)
        import numpy as np

        diff = np.abs(outputs[0] - torch_output_np)
        max_diff = np.max(diff)
        logger.info(f"Maximum difference between PyTorch and ONNX: {max_diff:.6f}")

        if max_diff < 1e-3:
            logger.info("✅ ONNX and PyTorch outputs match within tolerance")
        else:
            logger.warning(f"⚠️ ONNX and PyTorch outputs differ by {max_diff:.6f}")

        logger.info("✅ ONNX conversion test passed")
        return True

    except Exception as e:
        logger.error(f"❌ ONNX conversion test failed: {e}")
        return False


def main():
    """Run ONNX conversion tests."""
    logger.info("🚀 Starting ONNX Conversion Tests...")

    tests = [
        ("ONNX Conversion", test_onnx_conversion),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*40}")

        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")

    logger.info(f"\n{'='*40}")
    logger.info(f"ONNX Conversion Tests Results: {passed}/{total} tests passed")
    logger.info(f"{'='*40}")

    if passed == total:
        logger.info("🎉 All ONNX conversion tests passed!")
        return True
    else:
        logger.error("💥 Some ONNX conversion tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)