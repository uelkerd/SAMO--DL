        # Benchmark original model
        # Benchmark quantized model
        # Calculate speedup
        # Check if input model exists
        # Create model
        # Create output directory if it doesn't exist
        # Define quantization configuration
        # Load checkpoint
        # Load state dict
        # Measure original model size
        # Measure quantized model size
        # Prepare model for quantization
        # Quantize
        # Quantize model
        # Save compression metrics
        # Save quantized model
        # Set model to evaluation mode
        # Set optimal temperature and threshold
    # Benchmark
    # Create dummy input (batch_size=1, seq_len=128)
    # Warm up
# Add src to path
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
import argparse
import logging
import sys
import time
import torch
import torch.quantization

"""
Compress Model

This script applies quantization to the BERT emotion classifier model
to reduce its size and improve inference speed.

Usage:
    python scripts/compress_model.py [--input_model PATH] [--output_model PATH]

Arguments:
    --input_model: Path to input model (default: test_checkpoints/best_model.pt)
    --output_model: Path to save compressed model (default: models/checkpoints/bert_emotion_classifier_quantized.pt)
"""

sys.path.append(str(Path(__file__).parent.parent.resolve()))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_MODEL = "test_checkpoints/best_model.pt"
DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier_quantized.pt"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6


def compress_model(input_model: str, output_model: str) -> bool:
    """Compress model using dynamic quantization.

    Args:
        input_model: Path to input model
        output_model: Path to save compressed model

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        device = torch.device("cpu")  # Quantization requires CPU

        input_path = Path(input_model)
        if not input_path.exists():
            logger.error("Input model not found: {input_path}")
            return False

        output_path = Path(output_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model from {input_path}...")

        checkpoint = torch.load(input_path, map_location=device, weights_only=False)

        model, _ = create_bert_emotion_classifier()

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            logger.error("Unexpected checkpoint format: {type(checkpoint)}")
            return False

        model.set_temperature(OPTIMAL_TEMPERATURE)
        model.prediction_threshold = OPTIMAL_THRESHOLD

        original_size = get_model_size(model)
        logger.info("Original model size: {original_size:.2f} MB")

        logger.info("Benchmarking original model...")
        original_inference_time = benchmark_inference(model)

        logger.info("Applying dynamic quantization...")

        model.eval()

        torch.quantization.get_default_qconfig("fbgemm")

        torch.quantization.prepare(model, inplace=True)

        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        quantized_size = get_model_size(quantized_model)
        logger.info("Quantized model size: {quantized_size:.2f} MB")
        logger.info("Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

        logger.info("Benchmarking quantized model...")
        quantized_inference_time = benchmark_inference(quantized_model)

        speedup = original_inference_time / quantized_inference_time
        logger.info("Inference speedup: {speedup:.2f}x")

        logger.info("Saving quantized model to {output_path}...")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint["model_state_dict"] = quantized_model.state_dict()
            checkpoint["quantized"] = True
            torch.save(checkpoint, output_path)
        else:
            torch.save(quantized_model.state_dict(), output_path)

        logger.info("✅ Model compression complete!")

        metrics = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_percent": (1 - quantized_size / original_size) * 100,
            "original_inference_ms": original_inference_time * 1000,
            "quantized_inference_ms": quantized_inference_time * 1000,
            "speedup": speedup,
        }

        logger.info("📊 Compression metrics:")
        for _key, _value in metrics.items():
            logger.info("  {key}: {value:.2f}")

        return True

    except Exception:
        logger.error("Error compressing model: {e}")
        return False


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in MB
    """
    temp_file = Path("temp_model.pt")
    torch.save(model.state_dict(), temp_file)
    size_bytes = temp_file.stat().st_size
    temp_file.unlink()
    return size_bytes / (1024 * 1024)  # Convert to MB


def benchmark_inference(model: torch.nn.Module, num_runs: int = 50) -> float:
    """Benchmark model inference time.

    Args:
        model: PyTorch model
        num_runs: Number of inference runs to average

    Returns:
        float: Average inference time in seconds
    """
    dummy_input = {
        "input_ids": torch.randint(0, 30522, (1, 128)),
        "attention_mask": torch.ones(1, 128),
    }

    for _ in range(10):
        with torch.no_grad():
            _ = model(**dummy_input)

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**dummy_input)
    end_time = time.time()

    return (end_time - start_time) / num_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress BERT emotion classifier model")
    parser.add_argument(
        "--input_model",
        type=str,
        default=DEFAULT_INPUT_MODEL,
        help="Path to input model (default: {DEFAULT_INPUT_MODEL})",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save compressed model (default: {DEFAULT_OUTPUT_MODEL})",
    )

    args = parser.parse_args()
    success = compress_model(args.input_model, args.output_model)
    sys.exit(0 if success else 1)
