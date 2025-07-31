#!/usr/bin/env python3
"""
Performance Optimization Script for SAMO Deep Learning.

This script handles GPU setup verification, ONNX model conversion,
and comprehensive performance benchmarking to meet <500ms P95 targets.

Usage:
    python scripts/optimize_performance.py --check-gpu
    python scripts/optimize_performance.py --convert-onnx --model-path ./models/checkpoints/best_model.pt
    python scripts/optimize_performance.py --benchmark --target-latency 500
"""

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# Add project root to Python path - more robust for CI environments
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Debug logging for path resolution
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 entries

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_gpu_setup() -> dict[str, any]:
    """Check GPU availability and CUDA setup.

    Returns:
        Dictionary with GPU setup information

    """
    logger.info("🔍 Checking GPU Setup...")

    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": None,
        "memory_total": None,
        "memory_free": None,
        "recommendations": [],
    }

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_free = torch.cuda.memory_reserved(0)

        gpu_info.update(
            {
                "device_name": device_name,
                "memory_total": f"{memory_total / 1e9:.1f} GB",
                "memory_free": f"{memory_free / 1e9:.1f} GB",
            }
        )

        logger.info(f"✅ GPU Available: {device_name}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   Memory: {memory_total / 1e9:.1f} GB total")

        if memory_total < 8e9:  # Less than 8GB
            gpu_info["recommendations"].append(
                "Consider using mixed precision training (fp16) to save memory"
            )
            gpu_info["recommendations"].append("Reduce batch size if encountering OOM errors")

        if "T4" in device_name or "V100" in device_name:
            gpu_info["recommendations"].append(
                "Tensor Core support available - use mixed precision for 2x speedup"
            )

    else:
        logger.warning("⚠️  No GPU available - training will use CPU")
        gpu_info["recommendations"].extend(
            [
                "Install CUDA-compatible PyTorch for GPU acceleration",
                "Consider using Google Colab or cloud GPU instances for faster training",
                "CPU training will be significantly slower for BERT models",
            ]
        )

    return gpu_info


def convert_to_onnx(
    model_path: str,
    output_path: str | None = None,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
) -> str:
    """Convert PyTorch model to ONNX format for inference optimization.

    Args:
        model_path: Path to the saved PyTorch model
        output_path: Path to save ONNX model (auto-generated if None)
        model_name: Tokenizer model name
        max_length: Maximum sequence length

    Returns:
        Path to the converted ONNX model

    """
    logger.info("🔄 Converting model to ONNX format...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model = BERTEmotionClassifier(model_name=model_name, num_emotions=28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dummy_text = "This is a sample text for ONNX conversion."
    dummy_encoding = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    if output_path is None:
        output_path = model_path.replace(".pt", ".onnx")

    torch.onnx.export(
        model,
        (dummy_encoding["input_ids"], dummy_encoding["attention_mask"]),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"✅ ONNX model saved to: {output_path}")
    return output_path


def benchmark_model_performance(
    model_path: str,
    onnx_path: str | None = None,
    num_samples: int = 100,
    target_latency: float = 500.0,
    model_name: str = "bert-base-uncased",
) -> dict[str, any]:
    """Benchmark model performance for PyTorch and ONNX versions.

    Args:
        model_path: Path to PyTorch model
        onnx_path: Path to ONNX model (optional)
        num_samples: Number of samples for benchmarking
        target_latency: Target P95 latency in milliseconds
        model_name: Tokenizer model name

    Returns:
        Dictionary with benchmark results

    """
    logger.info("📊 Starting performance benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_texts = [
        "I'm feeling great today, everything is going well!",
        "This situation is really frustrating and disappointing.",
        "I'm worried about the upcoming presentation at work.",
        "The sunset was absolutely beautiful and peaceful.",
        "I can't believe how angry this makes me feel.",
        "Feeling grateful for all the support from friends.",
        "This is the most boring meeting I've ever attended.",
        "I'm so excited about the weekend trip we planned!",
    ] * (num_samples // 8 + 1)
    sample_texts = sample_texts[:num_samples]

    results = {}

    if Path(model_path).exists():
        logger.info("Testing PyTorch model performance...")
        pytorch_latencies = benchmark_pytorch_model(model_path, sample_texts, tokenizer, device)
        results["pytorch"] = analyze_latencies(pytorch_latencies, "PyTorch")

    if onnx_path and Path(onnx_path).exists():
        logger.info("Testing ONNX model performance...")
        onnx_latencies = benchmark_onnx_model(onnx_path, sample_texts, tokenizer)
        results["onnx"] = analyze_latencies(onnx_latencies, "ONNX")

        if "pytorch" in results:
            speedup = results["pytorch"]["mean_latency"] / results["onnx"]["mean_latency"]
            results["onnx_speedup"] = f"{speedup:.2f}x"
            logger.info(f"🚀 ONNX Speedup: {speedup:.2f}x")

    results["target_latency"] = target_latency
    results["assessment"] = assess_performance(results, target_latency)

    return results


def benchmark_pytorch_model(
    model_path: str, texts: list[str], tokenizer, device: torch.device
) -> list[float]:
    """Benchmark PyTorch model inference times."""
    checkpoint = torch.load(model_path, map_location=device)

    model = BERTEmotionClassifier(model_name="bert-base-uncased", num_emotions=28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    latencies = []

    with torch.no_grad():
        for text in texts:
            start_time = time.time()

            encoding = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            model(encoding["input_ids"], encoding["attention_mask"])

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return latencies


def benchmark_onnx_model(model_path: str, texts: list[str], tokenizer) -> list[float]:
    """Benchmark ONNX model inference times."""
    session = ort.InferenceSession(model_path)

    latencies = []

    for text in texts:
        start_time = time.time()

        encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        inputs = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

        session.run(None, inputs)

        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return latencies


def analyze_latencies(latencies: list[float], model_type: str) -> dict[str, float]:
    """Analyze latency statistics."""
    latencies_sorted = sorted(latencies)

    stats = {
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "p95_latency": latencies_sorted[int(0.95 * len(latencies))],
        "p99_latency": latencies_sorted[int(0.99 * len(latencies))],
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }

    logger.info(f"{model_type} Performance:")
    logger.info(f"  Mean: {stats['mean_latency']:.1f}ms")
    logger.info(f"  P95:  {stats['p95_latency']:.1f}ms")
    logger.info(f"  P99:  {stats['p99_latency']:.1f}ms")

    return stats


def assess_performance(results: dict[str, any], target_latency: float) -> dict[str, str]:
    """Assess whether performance meets targets."""
    assessment = {}

    for model_type in ["pytorch", "onnx"]:
        if model_type in results:
            p95_latency = results[model_type]["p95_latency"]

            if p95_latency <= target_latency:
                assessment[model_type] = (
                    f"✅ MEETS TARGET ({p95_latency:.1f}ms ≤ {target_latency}ms)"
                )
            elif p95_latency <= target_latency * 1.2:  # Within 20%
                assessment[model_type] = (
                    f"⚠️  CLOSE TO TARGET ({p95_latency:.1f}ms vs {target_latency}ms)"
                )
            else:
                assessment[model_type] = (
                    f"❌ EXCEEDS TARGET ({p95_latency:.1f}ms > {target_latency}ms)"
                )

    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description="SAMO Deep Learning Performance Optimization")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU setup")
    parser.add_argument("--convert-onnx", action="store_true", help="Convert model to ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark model performance")
    parser.add_argument("--model-path", type=str, default="./models/checkpoints/best_model.pt")
    parser.add_argument("--onnx-path", type=str, default=None)
    parser.add_argument(
        "--target-latency", type=float, default=500.0, help="Target P95 latency (ms)"
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of benchmark samples")

    args = parser.parse_args()

    if args.check_gpu:
        gpu_info = check_gpu_setup()

        print("\n" + "=" * 50)
        print("🔍 GPU SETUP ASSESSMENT")
        print("=" * 50)

        for key, value in gpu_info.items():
            if key != "recommendations":
                print(f"{key}: {value}")

        if gpu_info["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in gpu_info["recommendations"]:
                print("   • {rec}")

    if args.convert_onnx:
        if not Path(args.model_path).exists():
            logger.error(f"Model not found: {args.model_path}")
            return

        onnx_path = convert_to_onnx(args.model_path, args.onnx_path)
        print(f"\n✅ ONNX conversion complete: {onnx_path}")

    if args.benchmark:
        if not Path(args.model_path).exists():
            logger.error(f"Model not found: {args.model_path}")
            return

        results = benchmark_model_performance(
            args.model_path, args.onnx_path, args.num_samples, args.target_latency
        )

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        for model_type, assessment in results["assessment"].items():
            print(f"\n{model_type.upper()}: {assessment}")

        if "onnx_speedup" in results:
            print(f"\n🚀 ONNX Optimization: {results['onnx_speedup']} faster")

        print(f"\nTarget: P95 ≤ {args.target_latency}ms")


if __name__ == "__main__":
    main()
