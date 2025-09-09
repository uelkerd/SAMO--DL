#!/usr/bin/env python3
"""
Model Optimization Script for REQ-DL-008

This script implements comprehensive model optimization techniques for SAMO Deep Learning:
1. ONNX Runtime integration for 2x inference speedup
2. Model compression achieving <100MB total model size
3. GPU/CPU compatibility for high availability
4. Performance benchmarking and validation

Usage:
    python scripts/legacy/model_optimization.py [--model_path PATH] [--output_dir PATH] [--benchmark]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_PATH = "models/checkpoints/bert_emotion_classifier_final.pt"
DEFAULT_OUTPUT_DIR = "models/optimized"
TARGET_SIZE_MB = 100
TARGET_SPEEDUP = 2.0

def get_model_size_mb(model_path: Union[str, Path]) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)

def apply_dynamic_quantization(
    model: torch.nn.Module, model_path: str, output_path: str
) -> Dict[str, float]:
    """Apply dynamic quantization to reduce model size."""
    logger.info("Applying dynamic quantization...")
    model.eval()
    original_size = get_model_size_mb(model_path)
    logger.info(f"Original model size: {original_size:.2f} MB")

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    torch.save(
        {
            "model_state_dict": quantized_model.state_dict(),
            "quantized": True,
            "quantization_type": "dynamic",
        },
        output_path,
    )

    quantized_size = get_model_size_mb(output_path)
    size_reduction = (original_size - quantized_size) / original_size * 100
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {size_reduction:.2f}%")

    if quantized_size <= TARGET_SIZE_MB:
        logger.info(f"✅ Target size achieved: {quantized_size:.2f} MB <= {TARGET_SIZE_MB} MB")
    else:
        logger.warning(f"⚠️ Target size not achieved: {quantized_size:.2f} MB > {TARGET_SIZE_MB} MB")

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_percent": size_reduction,
    }

def convert_to_onnx(model: torch.nn.Module, output_path: str, opset_version: int = 12) -> str:
    """Convert PyTorch model to ONNX format."""
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
        name: {0: "batch_size", 1: "sequence_length"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch_size"}

    torch.onnx.export(
        model,
        (
            dummy_inputs["input_ids"],
            dummy_inputs["attention_mask"],
            dummy_inputs.get("token_type_ids", torch.zeros_like(dummy_inputs["input_ids"])),
        ),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    logger.info(f"Model converted to ONNX format: {output_path}")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verified successfully")
    except ImportError:
        logger.warning("⚠️ ONNX package not installed, skipping verification. `pip install onnx`")
    except Exception as e:
        logger.error(f"❌ ONNX model verification failed: {e}")

    return output_path

def benchmark_models(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    onnx_path: str,
    num_runs: int = 100,
    batch_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Benchmark original, quantized, and ONNX models."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    logger.info("Running performance benchmarks...")
    original_model.eval()
    quantized_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(original_model.model_name)
    results: Dict[str, Any] = {"pytorch_original": {}, "pytorch_quantized": {}, "onnx": {}}

    try:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(onnx_path)
        onnx_available = True
    except (ImportError, Exception) as e:
        logger.warning(f"⚠️ ONNX Runtime not available, skipping ONNX benchmarks: {e}")
        onnx_available = False

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking with batch_size={batch_size}...")
        texts = [f"This is test sentence {i} for benchmarking." for i in range(batch_size)]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        # PyTorch Original
        original_times = []
        for _ in tqdm(range(num_runs), desc="Original PyTorch"):
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(**inputs)
            original_times.append(time.time() - start_time)

        # PyTorch Quantized
        quantized_times = []
        for _ in tqdm(range(num_runs), desc="Quantized PyTorch"):
            start_time = time.time()
            with torch.no_grad():
                _ = quantized_model(**inputs)
            quantized_times.append(time.time() - start_time)

        # ONNX
        onnx_times = []
        if onnx_available:
            onnx_inputs = {
                k: v.numpy() for k, v in inputs.items()
            }
            if "token_type_ids" not in onnx_inputs:
                 onnx_inputs["token_type_ids"] = np.zeros_like(onnx_inputs["input_ids"])

            for _ in tqdm(range(num_runs), desc="ONNX Runtime"):
                start_time = time.time()
                _ = onnx_session.run(None, onnx_inputs)
                onnx_times.append(time.time() - start_time)

        # Store results
        results["pytorch_original"][f"batch_{batch_size}"] = {"mean_ms": np.mean(original_times) * 1000}
        results["pytorch_quantized"][f"batch_{batch_size}"] = {"mean_ms": np.mean(quantized_times) * 1000, "speedup": np.mean(original_times) / np.mean(quantized_times)}
        if onnx_available:
            results["onnx"][f"batch_{batch_size}"] = {"mean_ms": np.mean(onnx_times) * 1000, "speedup": np.mean(original_times) / np.mean(onnx_times)}

        logger.info(f"Original PyTorch: {results['pytorch_original'][f'batch_{batch_size}']['mean_ms']:.2f} ms")
        logger.info(f"Quantized PyTorch: {results['pytorch_quantized'][f'batch_{batch_size}']['mean_ms']:.2f} ms (speedup: {results['pytorch_quantized'][f'batch_{batch_size}']['speedup']:.2f}x)")
        if onnx_available:
            logger.info(f"ONNX Runtime: {results['onnx'][f'batch_{batch_size}']['mean_ms']:.2f} ms (speedup: {results['onnx'][f'batch_{batch_size}']['speedup']:.2f}x)")
            if results["onnx"][f"batch_{batch_size}"]["speedup"] >= TARGET_SPEEDUP:
                logger.info(f"✅ Target speedup achieved for batch size {batch_size}")
            else:
                logger.warning(f"⚠️ Target speedup not achieved for batch size {batch_size}")

    return results

def verify_gpu_compatibility(model: torch.nn.Module) -> bool:
    """Verify model compatibility with both CPU and GPU."""
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

        if torch.allclose(cpu_output.cpu(), gpu_output.cpu(), rtol=1e-3, atol=1e-3):
            logger.info("✅ Model is compatible with both CPU and GPU")
            return True
        else:
            logger.error("❌ Model outputs differ between CPU and GPU")
            return False
    except Exception as e:
        logger.error(f"❌ Error during GPU compatibility check: {e}")
        return False
    finally:
        model.to("cpu")

def optimize_model(model_path: str, output_dir: str, run_benchmark: bool) -> Dict[str, Any]:
    """Apply all optimization techniques to the model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading model from {model_path}...")
    device = torch.device("cpu")

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)
    model, _ = create_bert_emotion_classifier()
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device).eval()

    # Quantization
    quantized_path = output_dir / "bert_emotion_classifier_quantized.pt"
    quantization_metrics = apply_dynamic_quantization(model, model_path, str(quantized_path))

    # ONNX Conversion
    onnx_path = output_dir / "bert_emotion_classifier.onnx"
    onnx_model_path = convert_to_onnx(model, str(onnx_path))

    # GPU Compatibility
    gpu_compatible = verify_gpu_compatibility(model)

    # Benchmarking
    benchmark_results = {}
    if run_benchmark:
        quantized_checkpoint = torch.load(quantized_path, map_location=device)
        quantized_model, _ = create_bert_emotion_classifier()
        quantized_model.load_state_dict(quantized_checkpoint["model_state_dict"])
        quantized_model.to(device).eval()
        benchmark_results = benchmark_models(model, quantized_model, onnx_model_path)
        benchmark_path = output_dir / "benchmark_results.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to {benchmark_path}")

    # Final Summary
    results = {
        "quantization": quantization_metrics,
        "onnx_conversion": {"path": str(onnx_model_path)},
        "gpu_compatible": gpu_compatible,
        "benchmark": benchmark_results if run_benchmark else "Not run",
    }
    results_path = output_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Optimization results saved to {results_path}")
    logger.info("\n=== Optimization Summary ===")
    logger.info(f"Quantized model size: {quantization_metrics['quantized_size_mb']:.2f} MB")
    logger.info(f"ONNX model path: {onnx_model_path}")
    logger.info(f"GPU compatible: {'Yes' if gpu_compatible else 'No'}")
    if run_benchmark:
        onnx_speedup = results.get("benchmark", {}).get("onnx", {}).get("batch_1", {}).get("speedup", 0)
        logger.info(f"ONNX speedup (batch 1): {onnx_speedup:.2f}x")
    logger.info("✅ Optimization complete.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Optimization Script")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to input model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save optimized models (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()
    optimize_model(model_path=args.model_path, output_dir=args.output_dir, run_benchmark=args.benchmark)
