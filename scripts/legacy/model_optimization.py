            # Check if target speedup is achieved with ONNX
            # Prepare ONNX inputs
        # Benchmark ONNX model
        # Benchmark original PyTorch model
        # Benchmark quantized PyTorch model
        # Calculate statistics
        # Check if outputs are close
        # Create dummy input
        # Generate random input texts
        # Log results
        # Move model back to CPU
        # Move outputs to CPU for comparison
        # Save benchmark results
        # Test on CPU
        # Test on GPU
        # Tokenize
        import onnx
        import onnxruntime as ort
    # Apply dynamic quantization to linear layers
    # Apply optimizations
    # Apply quantization
    # Benchmark for different batch sizes
    # Calculate size reduction
    # Check if CUDA is available
    # Check if ONNX Runtime is available
    # Check if all requirements are met
    # Check if model exists
    # Check if target size is achieved
    # Collect all metrics
    # Convert to ONNX
    # Create dummy input for ONNX export
    # Create model
    # Create output directory
    # Create tokenizer
    # Define dynamic axes for variable batch size and sequence length
    # Define input and output names
    # Define output paths
    # Exit with success code
    # Export to ONNX
    # Initialize results dictionary
    # Load checkpoint
    # Load model
    # Load quantized model
    # Load state dict
    # Log summary
    # Measure original model size
    # Measure quantized model size
    # Return metrics
    # Run benchmarks if requested
    # Save quantized model
    # Save results
    # Set model to evaluation mode
    # Set model to evaluation mode
    # Set model to evaluation mode
    # Set models to evaluation mode
    # Verify GPU compatibility
    # Verify ONNX model
# Add src to path
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, Union, Optional
import argparse
import json
import logging
import numpy as np
import sys
import time
import torch







"""
Model Optimization Script for REQ-DL-008

This script implements comprehensive model optimization techniques for SAMO Deep
Learning:
1. ONNX Runtime integration for 2x inference speedup
2. Model compression achieving <100MB total model size
3. GPU/CPU compatibility for high availability
4. Performance benchmarking and validation

Usage:
    python scripts/model_optimization
    .py [--model_path PATH] [--output_dir PATH] [--benchmark]

Arguments:
    --model_path: Path to input model (
                                       default: models/checkpoints/bert_emotion_classifier_final.pt
                                      )
    --output_dir: Directory to save optimized models (default: models/optimized)
    --benchmark: Run performance benchmarks on optimized models
"""

sys.path.append(str(Path(__file__).parent.parent.resolve()))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/checkpoints/bert_emotion_classifier_final.pt"
DEFAULT_OUTPUT_DIR = "models/optimized"
TARGET_SIZE_MB = 100  # Maximum model size in MB
TARGET_SPEEDUP = 2.0  # Target inference speedup


def apply_dynamic_quantization(
    model: torch.nn.Module, model_path: str, output_path: str
) -> dict[str, float]:
    """Apply dynamic quantization to reduce model size.

    Args:
        model: PyTorch model
        model_path: Path to original model
        output_path: Path to save quantized model

    Returns:
        Dictionary with optimization metrics
    """
    logger.info("Applying dynamic quantization...")

    model.eval()

    original_size = get_model_size_mb(model_path)
    logger.info("Original model size: {original_size:.2f} MB")

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    torch.save(
        {
            "model_state_dict": quantized_model.state_dict(),
            "quantized": True,
            "quantization_type": "dynamic",
            "original_size_mb": original_size,
        },
        output_path,
    )

    quantized_size = get_model_size_mb(output_path)
    logger.info("Quantized model size: {quantized_size:.2f} MB")

    size_reduction = (original_size - quantized_size) / original_size * 100
    logger.info("Size reduction: {size_reduction:.2f}%")

    if quantized_size <= TARGET_SIZE_MB:
        logger.info(
                    "✅ Target size achieved: {quantized_size:.2f} MB <= {TARGET_SIZE_MB} MB"
                   )
    else:
        logger.warning(
                       "⚠️ Target size not achieved: {quantized_size:.2f} MB > {TARGET_SIZE_MB} MB"
                      )

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_percent": size_reduction,
    }


def convert_to_onnx(
                    model: torch.nn.Module,
                    output_path: str,
                    opset_version: int = 12) -> str:
    """Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    logger.info("Converting model to ONNX format...")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    dummy_text = "This is a test sentence for ONNX conversion."
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (
            dummy_inputs["input_ids"],
            dummy_inputs["attention_mask"],
            dummy_inputs.get(
                             "token_type_ids",
                             torch.zeros_like(dummy_inputs["input_ids"])),
                             
        ),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    logger.info("Model converted to ONNX format: {output_path}")

    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verified successfully")
    except ImportError:
        logger.warning("⚠️ ONNX package not installed, skipping verification")
        logger.info("To install: pip install onnx")
    except Exception as e:
        logger.error("❌ ONNX model verification failed: {e}")

    return output_path


def benchmark_models(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    onnx_path: str,
    num_runs: int = 100,
    batch_sizes: Optional[list] = None,
) -> dict[str, Any]:
    """Benchmark original, quantized, and ONNX models.

    Args:
        original_model: Original PyTorch model
        quantized_model: Quantized PyTorch model
        onnx_path: Path to ONNX model
        num_runs: Number of inference runs for benchmarking
        batch_sizes: List of batch sizes to benchmark

    Returns:
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    logger.info("Running performance benchmarks...")

    original_model.eval()
    quantized_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(original_model.model_name)

    results = {"pytorch_original": {}, "pytorch_quantized": {}, "onnx": {}}

    onnx_available = False
    try:
        onnx_session = ort.InferenceSession(onnx_path)
        onnx_available = True
    except ImportError:
        logger.warning("⚠️ ONNX Runtime not installed, skipping ONNX benchmarks")
        logger.info("To install: pip install onnxruntime")
    except Exception as e:
        logger.error("❌ Error loading ONNX model: {e}")

    for batch_size in batch_sizes:
        logger.info("Benchmarking with batch_size={batch_size}...")

        texts = ["This is test sentence {i} for benchmarking." for i in range(
                                                                              batch_size)]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        original_times = []
        for _ in tqdm(range(num_runs), desc="Original PyTorch"):
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(**inputs)
            original_times.append(time.time() - start_time)

        quantized_times = []
        for _ in tqdm(range(num_runs), desc="Quantized PyTorch"):
            start_time = time.time()
            with torch.no_grad():
                _ = quantized_model(**inputs)
            quantized_times.append(time.time() - start_time)

        onnx_times = []
        if onnx_available:
            onnx_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
                "token_type_ids": inputs.get(
                    "token_type_ids", torch.zeros_like(inputs["input_ids"])
                ).numpy(),
            }

            for _ in tqdm(range(num_runs), desc="ONNX Runtime"):
                start_time = time.time()
                _ = onnx_session.run(None, onnx_inputs)
                onnx_times.append(time.time() - start_time)

        results["pytorch_original"]["batch_{batch_size}"] = {
            "mean_ms": np.mean(original_times) * 1000,
            "median_ms": np.median(original_times) * 1000,
            "p95_ms": np.percentile(original_times, 95) * 1000,
            "p99_ms": np.percentile(original_times, 99) * 1000,
        }

        results["pytorch_quantized"]["batch_{batch_size}"] = {
            "mean_ms": np.mean(quantized_times) * 1000,
            "median_ms": np.median(quantized_times) * 1000,
            "p95_ms": np.percentile(quantized_times, 95) * 1000,
            "p99_ms": np.percentile(quantized_times, 99) * 1000,
            "speedup": np.mean(original_times) / np.mean(quantized_times),
        }

        if onnx_available:
            results["onnx"]["batch_{batch_size}"] = {
                "mean_ms": np.mean(onnx_times) * 1000,
                "median_ms": np.median(onnx_times) * 1000,
                "p95_ms": np.percentile(onnx_times, 95) * 1000,
                "p99_ms": np.percentile(onnx_times, 99) * 1000,
                "speedup": np.mean(original_times) / np.mean(onnx_times),
            }

        logger.info("Batch size: {batch_size}")
        logger.info(
"Original PyTorch: {results['pytorch_original']['batch_{batch_size}']['mean_ms']:.2f}
ms"
        )
        logger.info(
"Quantized PyTorch: {results['pytorch_quantized']['batch_{batch_size}']['mean_ms']:.2f}
ms "
            + "(
                speedup: {results['pytorch_quantized']['batch_{batch_size}']['speedup']:.2f}x)"
        )

        if onnx_available:
            logger.info(
"ONNX Runtime: {results['onnx']['batch_{batch_size}']['mean_ms']:.2f} ms "
                + "(speedup: {results['onnx']['batch_{batch_size}']['speedup']:.2f}x)"
            )

            if results["onnx"]["batch_{batch_size}"]["speedup"] >= TARGET_SPEEDUP:
                logger.info(
"✅ Target speedup achieved: {results['onnx']['batch_{batch_size}']['speedup']:.2f}x >=
{TARGET_SPEEDUP}x"
                )
            else:
                logger.warning(
"⚠️ Target speedup not achieved: {results['onnx']['batch_{batch_size}']['speedup']:.2f}x
< {TARGET_SPEEDUP}x"
                )

    return results


def get_model_size_mb(model_path: Union[str, Path]) -> float:
    """Get model file size in MB.

    Args:
        model_path: Path to model file

    Returns:
        Model size in MB
    """
    path = Path(model_path)
    return path.stat().st_size / (1024 * 1024)


def verify_gpu_compatibility(model: torch.nn.Module) -> bool:
    """Verify model compatibility with both CPU and GPU.

    Args:
        model: PyTorch model

    Returns:
        True if compatible with both CPU and GPU, False otherwise
    """
    logger.info("Verifying GPU compatibility...")

    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, skipping GPU compatibility check")
        return True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        dummy_text = "This is a test sentence for GPU compatibility."
        dummy_inputs = tokenizer(
dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        model.to("cpu")
        with torch.no_grad():
            cpu_output = model(**dummy_inputs)

        model.to("cuda")
        dummy_inputs_gpu = {k: v.to("cuda") for k, v in dummy_inputs.items()}
        with torch.no_grad():
            gpu_output = model(**dummy_inputs_gpu)

        gpu_output_cpu = gpu_output.cpu()

        if torch.allclose(cpu_output, gpu_output_cpu, rtol=1e-3, atol=1e-3):
            logger.info("✅ Model is compatible with both CPU and GPU")
            return True
        else:
            logger.error("❌ Model outputs differ between CPU and GPU")
            return False

    except Exception as e:
        logger.error("❌ Error during GPU compatibility check: {e}")
        return False
    finally:
        model.to("cpu")


def optimize_model(
                   model_path: str,
                   output_dir: str,
                   run_benchmark: bool = False) -> dict[str,
                   Any]:
    """Apply all optimization techniques to model.

    Args:
        model_path: Path to input model
        output_dir: Directory to save optimized models
        run_benchmark: Whether to run performance benchmarks

    Returns:
        Dictionary with optimization results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from {model_path}...")
    device = torch.device("cpu")  # Use CPU for optimization

    if not Path(model_path).exists():
        logger.error("Model not found: {model_path}")
        return {}

    checkpoint = torch.load(model_path, map_location=device)

    model, _ = create_bert_emotion_classifier()
    model.to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        logger.error("Unexpected checkpoint format: {type(checkpoint)}")
        return {}

    model.eval()

    quantized_path = output_dir / "bert_emotion_classifier_quantized.pt"
    onnx_path = output_dir / "bert_emotion_classifier.onnx"

    quantization_metrics = apply_dynamic_quantization(model, model_path, quantized_path)

    quantized_checkpoint = torch.load(quantized_path, map_location=device)
    quantized_model, _ = create_bert_emotion_classifier()
    quantized_model.to(device)
    quantized_model.load_state_dict(quantized_checkpoint["model_state_dict"])
    quantized_model.eval()

    onnx_model_path = convert_to_onnx(model, onnx_path)

    gpu_compatible = verify_gpu_compatibility(model)

    benchmark_results = {}
    if run_benchmark:
        benchmark_results = benchmark_models(model, quantized_model, onnx_model_path)

        benchmark_path = output_dir / "benchmark_results.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info("Benchmark results saved to {benchmark_path}")

    results = {
        "quantization": quantization_metrics,
        "onnx_conversion": {"path": str(onnx_path)},
        "gpu_compatible": gpu_compatible,
        "benchmark": benchmark_results if run_benchmark else "Not run",
    }

    results_path = output_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Optimization results saved to {results_path}")

    logger.info("\n=== Optimization Summary ===")
    logger.info(
                "Original model size: {quantization_metrics['original_size_mb']:.2f} MB"
               )
    logger.info(
                "Quantized model size: {quantization_metrics['quantized_size_mb']:.2f} MB"
               )
    logger.info("Size reduction: {quantization_metrics['size_reduction_percent']:.2f}%")
    logger.info("ONNX model path: {onnx_path}")
    logger.info("GPU compatible: {'Yes' if gpu_compatible else 'No'}")

if run_benchmark and "onnx" in benchmark_results and "batch_1" in
benchmark_results["onnx"]:
        logger.info(
                    "ONNX speedup: {benchmark_results['onnx']['batch_1']['speedup']:.2f}x"
                   )

    all_requirements_met = (
        quantization_metrics["quantized_size_mb"] <= TARGET_SIZE_MB
        and gpu_compatible
        and (
            not run_benchmark
            or (
                "onnx" in benchmark_results
                and "batch_1" in benchmark_results["onnx"]
                and benchmark_results["onnx"]["batch_1"]["speedup"] >= TARGET_SPEEDUP
            )
        )
    )

    if all_requirements_met:
        logger.info("✅ All optimization requirements met!")
    else:
        logger.warning(
                       "⚠️ Some optimization requirements not met. Check logs for details."
                      )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Optimization for REQ-DL-008")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to input model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save optimized models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
"--benchmark", action="store_true", help="Run performance benchmarks on optimized
models"
    )

    args = parser.parse_args()

    results = optimize_model(
model_path =
    args.model_path, output_dir=args.output_dir, run_benchmark=args.benchmark
    )

    sys.exit(0)
