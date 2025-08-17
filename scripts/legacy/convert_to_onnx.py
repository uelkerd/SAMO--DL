#!/usr/bin/env python3
"""
Convert Model to ONNX

This script converts the BERT emotion classifier model to ONNX format
for faster inference and easier deployment.

Usage:
    python scripts/convert_to_onnx.py [--input_model PATH] [--output_model PATH]

Arguments:
    --input_model: Path to input model default: models/checkpoints/bert_emotion_classifier_quantized.pt
    --output_model: Path to save ONNX model default: models/checkpoints/bert_emotion_classifier.onnx
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Add src to path
sys.path.append(str(Path__file__.parent.parent.resolve()))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__

# Constants
DEFAULT_INPUT_MODEL = "models/checkpoints/bert_emotion_classifier_quantized.pt"
DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier.onnx"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6


def convert_to_onnxinput_model: str, output_model: str -> bool:
    """Convert model to ONNX format.

    Args:
        input_model: Path to input model
        output_model: Path to save ONNX model

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        device = torch.device"cpu"  # ONNX conversion requires CPU

        input_path = Pathinput_model
        if not input_path.exists():
            logger.errorf"Input model not found: {input_path}"
            return False

        output_path = Pathoutput_model
        output_path.parent.mkdirparents=True, exist_ok=True

        logger.infof"Loading model from {input_path}..."

        checkpoint = torch.loadinput_path, map_location=device, weights_only=False

        model, _ = create_bert_emotion_classifier()

        if isinstancecheckpoint, dict and "model_state_dict" in checkpoint:
            model.load_state_dictcheckpoint["model_state_dict"]
        elif isinstancecheckpoint, dict:
            model.load_state_dictcheckpoint
        else:
            logger.error(f"Unexpected checkpoint format: {typecheckpoint}")
            return False

        model.set_temperatureOPTIMAL_TEMPERATURE
        model.prediction_threshold = OPTIMAL_THRESHOLD

        model.eval()

        batch_size = 1
        sequence_length = 128
        dummy_input_ids = torch.randint(0, 30522, batch_size, sequence_length)
        dummy_attention_mask = torch.onesbatch_size, sequence_length
        dummy_token_type_ids = torch.zerosbatch_size, sequence_length

        logger.info"Benchmarking PyTorch model..."
        pytorch_inference_time = benchmark_pytorch_inference(
            model, dummy_input_ids, dummy_attention_mask
        )

        logger.info"Converting model to ONNX format..."

        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        output_names = ["logits"]

        # Create a wrapper function for the model
        def wrapper_functioninput_ids, attention_mask, token_type_ids:
            return modelinput_ids, attention_mask, token_type_ids

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input_ids, dummy_attention_mask, dummy_token_type_ids,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "token_type_ids": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        logger.infof"Model converted and saved to {output_path}"

        # Benchmark ONNX model
        logger.info"Benchmarking ONNX model..."
        onnx_inference_time = benchmark_onnx_inference(
            output_path, dummy_input_ids, dummy_attention_mask, dummy_token_type_ids
        )

        # Compare performance
        speedup = pytorch_inference_time / onnx_inference_time
        logger.infof"PyTorch inference time: {pytorch_inference_time:.4f}s"
        logger.infof"ONNX inference time: {onnx_inference_time:.4f}s"
        logger.infof"Speedup: {speedup:.2f}x"

        return True

    except Exception as e:
        logger.errorf"ONNX conversion failed: {e}"
        return False


def benchmark_pytorch_inferencemodel, input_ids, attention_mask, num_runs=50:
    """Benchmark PyTorch model inference time."""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range10:
            _ = modelinput_ids, attention_mask

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in rangenum_runs:
            _ = modelinput_ids, attention_mask
    end_time = time.time()

    return end_time - start_time / num_runs


def benchmark_onnx_inferencemodel_path, input_ids, attention_mask, token_type_ids, num_runs=50:
    """Benchmark ONNX model inference time."""
    import onnxruntime as ort

    # Create ONNX session
    session = ort.InferenceSessionmodel_path
    
    # Prepare inputs
    input_feed = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }

    # Warm up
    for _ in range10:
        _ = session.runNone, input_feed

    # Benchmark
    start_time = time.time()
    for _ in rangenum_runs:
        _ = session.runNone, input_feed
    end_time = time.time()

    return end_time - start_time / num_runs


def main():
    """Main function."""
    parser = argparse.ArgumentParserdescription="Convert BERT emotion classifier to ONNX"
    parser.add_argument(
        "--input_model",
        type=str,
        default=DEFAULT_INPUT_MODEL,
        help="Path to input model",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save ONNX model",
    )

    args = parser.parse_args()

    logger.info"🚀 Starting ONNX conversion..."
    logger.infof"Input model: {args.input_model}"
    logger.infof"Output model: {args.output_model}"

    success = convert_to_onnxargs.input_model, args.output_model

    if success:
        logger.info"✅ ONNX conversion completed successfully!"
        return 0
    else:
        logger.error"❌ ONNX conversion failed!"
        return 1


if __name__ == "__main__":
    sys.exit(main())
