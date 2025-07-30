            # Benchmark ONNX model
            # Calculate speedup
            # Save metrics
        # Benchmark PyTorch model
        # Check if input model exists
        # Check if onnxruntime is available
        # Create a wrapper function for export
        # Create dummy input for ONNX export
        # Create model
        # Create output directory if it doesn't exist
        # Define input names and output names
        # Export to ONNX
        # Export to ONNX
        # Load checkpoint
        # Load state dict
        # Set model to evaluation mode
        # Set optimal temperature and threshold
    # Benchmark
    # Benchmark
    # Create ONNX session
    # Prepare inputs
    # Warm up
    # Warm up
    import onnxruntime as ort
# Add src to path
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
import argparse
import importlib.util
import logging
import sys
import time
import torch





"""
Convert Model to ONNX

This script converts the BERT emotion classifier model to ONNX format
for faster inference and easier deployment.

Usage:
    python scripts/convert_to_onnx.py [--input_model PATH] [--output_model PATH]

Arguments:
    --input_model: Path to input model (default: models/checkpoints/bert_emotion_classifier_quantized.pt)
    --output_model: Path to save ONNX model (default: models/checkpoints/bert_emotion_classifier.onnx)
"""

sys.path.append(str(Path(__file__).parent.parent.resolve()))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_MODEL = "models/checkpoints/bert_emotion_classifier_quantized.pt"
DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier.onnx"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6


def convert_to_onnx(input_model: str, output_model: str) -> bool:
    """Convert model to ONNX format.

    Args:
        input_model: Path to input model
        output_model: Path to save ONNX model

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        device = torch.device("cpu")  # ONNX conversion requires CPU

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

        model.eval()

        batch_size = 1
        sequence_length = 128
        dummy_input_ids = torch.randint(0, 30522, (batch_size, sequence_length))
        dummy_attention_mask = torch.ones(batch_size, sequence_length)
        dummy_token_type_ids = torch.zeros(batch_size, sequence_length)

        logger.info("Benchmarking PyTorch model...")
        pytorch_inference_time = benchmark_pytorch_inference(
            model, dummy_input_ids, dummy_attention_mask
        )

        logger.info("Converting model to ONNX format...")

        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        output_names = ["logits"]

        def wrapper_function(input_ids, attention_mask, token_type_ids):
            return model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

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

        logger.info("✅ Model converted to ONNX format: {output_path}")

        if importlib.util.find_spec("onnxruntime") is not None:
            logger.info("Benchmarking ONNX model...")
            onnx_inference_time = benchmark_onnx_inference(
                output_path,
                dummy_input_ids.numpy(),
                dummy_attention_mask.numpy(),
                dummy_token_type_ids.numpy(),
            )

            speedup = pytorch_inference_time / onnx_inference_time
            logger.info("ONNX inference speedup: {speedup:.2f}x")

            metrics = {
                "pytorch_inference_ms": pytorch_inference_time * 1000,
                "onnx_inference_ms": onnx_inference_time * 1000,
                "speedup": speedup,
            }

            logger.info("📊 Conversion metrics:")
            for key, value in metrics.items():
                logger.info("  {key}: {value:.2f}")

        else:
            logger.warning("onnxruntime not found. Skipping ONNX benchmarking.")
            logger.info("To install: pip install onnxruntime")

        return True

    except Exception as e:
        logger.error("Error converting model to ONNX: {e}")
        return False


def benchmark_pytorch_inference(model, input_ids, attention_mask, num_runs=50):
    """Benchmark PyTorch model inference time.

    Args:
        model: PyTorch model
        input_ids: Input IDs tensor
        attention_mask: Attention mask tensor
        num_runs: Number of inference runs to average

    Returns:
        float: Average inference time in seconds
    """
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()

    return (end_time - start_time) / num_runs


def benchmark_onnx_inference(model_path, input_ids, attention_mask, token_type_ids, num_runs=50):
    """Benchmark ONNX model inference time.

    Args:
        model_path: Path to ONNX model
        input_ids: Input IDs numpy array
        attention_mask: Attention mask numpy array
        token_type_ids: Token type IDs numpy array
        num_runs: Number of inference runs to average

    Returns:
        float: Average inference time in seconds
    """
    session = ort.InferenceSession(model_path)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    for _ in range(10):
        _ = session.run(None, ort_inputs)

    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, ort_inputs)
    end_time = time.time()

    return (end_time - start_time) / num_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BERT emotion classifier model to ONNX format"
    )
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
        help="Path to save ONNX model (default: {DEFAULT_OUTPUT_MODEL})",
    )

    args = parser.parse_args()
    success = convert_to_onnx(args.input_model, args.output_model)
    sys.exit(0 if success else 1)
