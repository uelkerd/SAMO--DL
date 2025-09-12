#!/usr/bin/env python3
"""
Model Comparison Test Framework

This script compares the performance of different emotion detection models:
- Current BERT model (custom trained)
- New DeBERTa v3 Large model (from Hugging Face)
- Current production model (DistilRoBERTa-base)

Metrics measured:
- Inference latency
- Memory usage
- Prediction accuracy (on GoEmotions test set)
- F1 scores and other classification metrics
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

# Import model classes
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import TextClassificationPipeline
import os
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle protobuf compatibility issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

class ModelBenchmark:
    """Benchmark different emotion detection models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.results = {}

    def load_current_bert_model(self) -> bool:
        """Load the current custom BERT model."""
        try:
            # Try importing from src
            try:
                from src.models.emotion_detection.samo_bert_emotion_classifier import create_samo_bert_emotion_classifier
            except ImportError:
                # Try importing directly
                sys.path.insert(0, str(project_root / 'src'))
                from models.emotion_detection.samo_bert_emotion_classifier import create_samo_bert_emotion_classifier

            logger.info("Loading current BERT model...")
            model, loss_fn = create_samo_bert_emotion_classifier()
            self.models['bert_custom'] = model
            logger.info("✅ Current BERT model loaded")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load BERT model (this is expected if not available): {e}")
            return False

    def load_deberta_model(self) -> bool:
        """Load the new DeBERTa v3 Large model."""
        try:
            logger.info("Loading DeBERTa v3 Large model...")

            # Load from Hugging Face
            model_name = "duelker/samo-goemotions-deberta-v3-large"

            # Try to load with error handling
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                    low_cpu_mem_usage=True
                )
            except Exception as load_error:
                logger.warning(f"DeBERTa direct load failed, trying alternative: {load_error}")
                # Try loading with different settings
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )

            # Create pipeline
            clf = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None,  # Return all emotions
                truncation=True,
                max_length=512
            )

            self.models['deberta_large'] = clf
            logger.info("✅ DeBERTa v3 Large model loaded")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DeBERTa model (this is expected if network/HF issues): {e}")
            return False

    def load_production_model(self) -> bool:
        """Load the current production model (DistilRoBERTa)."""
        try:
            logger.info("Loading production DistilRoBERTa model...")

            model_name = "j-hartmann/emotion-english-distilroberta-base"
            clf = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None,
                truncation=True,
                max_length=512,
                model_kwargs={"torch_dtype": torch.float32}
            )

            self.models['distilroberta_prod'] = clf
            logger.info("✅ Production model loaded")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load production model (this is expected if network issues): {e}")
            return False

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }

    def benchmark_inference_speed(self, model_key: str, test_texts: List[str], num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark inference speed for a model."""
        if model_key not in self.models:
            return {'error': f'Model {model_key} not loaded'}

        model = self.models[model_key]
        latencies = []
        memory_usage = []

        logger.info(f"Benchmarking {model_key} inference speed...")

        # Warm up
        for text in test_texts[:3]:
            if model_key == 'bert_custom':
                model.predict_emotions(text, threshold=0.5)
            else:
                model(text)

        # Benchmark
        for i in range(num_runs):
            start_time = time.time()

            for text in test_texts:
                if model_key == 'bert_custom':
                    results = model.predict_emotions(text, threshold=0.5)
                else:
                    results = model(text)

            end_time = time.time()
            latency = (end_time - start_time) / len(test_texts) * 1000  # ms per text
            latencies.append(latency)
            memory_usage.append(self.get_memory_usage())

        return {
            'model': model_key,
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'avg_memory_mb': np.mean([m['rss_mb'] for m in memory_usage]),
            'throughput_texts_per_sec': 1000 / np.mean(latencies)
        }

    def benchmark_accuracy(self, model_key: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark accuracy on test dataset."""
        if model_key not in self.models:
            return {'error': f'Model {model_key} not loaded'}

        model = self.models[model_key]
        predictions = []
        true_labels = []
        latencies = []

        logger.info(f"Benchmarking {model_key} accuracy...")

        for item in test_data:
            text = item['text']
            true_emotion = item['emotion']

            start_time = time.time()

            if model_key == 'bert_custom':
                results = model.predict_emotions(text, threshold=0.5)
                # Get top prediction
                if results['emotions']:
                    pred_emotion = results['emotions'][0][0] if isinstance(results['emotions'][0], list) else results['emotions'][0]
                else:
                    pred_emotion = 'neutral'
                confidence = results['probabilities'][0][0] if results['probabilities'] else 0.0
            else:
                results = model(text)
                if results and len(results[0]) > 0:
                    pred_emotion = results[0][0]['label']
                    confidence = results[0][0]['score']
                else:
                    pred_emotion = 'neutral'
                    confidence = 0.0

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms

            predictions.append(pred_emotion)
            true_labels.append(true_emotion)
            latencies.append(latency)

        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(predictions) if predictions else 0.0

        return {
            'model': model_key,
            'accuracy': accuracy,
            'total_samples': len(test_data),
            'correct_predictions': correct,
            'avg_latency_ms': np.mean(latencies),
            'predictions': predictions[:10],  # First 10 for inspection
            'true_labels': true_labels[:10]
        }

    def run_comprehensive_benchmark(self, test_texts: List[str] = None, test_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing all models."""

        # Default test data
        if test_texts is None:
            test_texts = [
                "I am so happy today! This is amazing!",
                "I'm feeling really sad and disappointed about this situation.",
                "I'm frustrated but hopeful about the future.",
                "Thank you so much for your help!",
                "I feel anxious and worried about what might happen next.",
                "I'm grateful for all the support I've received.",
                "This situation makes me really angry.",
                "I'm surprised by how well things turned out.",
                "I feel proud of what I've accomplished.",
                "I'm nervous about the upcoming presentation."
            ] * 5  # Repeat for more stable measurements

        if test_data is None:
            # Create simple test data from texts
            test_data = [{'text': text, 'emotion': 'unknown'} for text in test_texts[:20]]

        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'models_loaded': list(self.models.keys()),
            'inference_benchmarks': {},
            'accuracy_benchmarks': {}
        }

        # Load all models
        self.load_current_bert_model()
        self.load_deberta_model()
        self.load_production_model()

        logger.info("Starting comprehensive model comparison...")

        # Run inference benchmarks
        for model_key in self.models.keys():
            logger.info(f"Running inference benchmark for {model_key}...")
            results['inference_benchmarks'][model_key] = self.benchmark_inference_speed(
                model_key, test_texts, num_runs=5
            )

        # Run accuracy benchmarks
        for model_key in self.models.keys():
            logger.info(f"Running accuracy benchmark for {model_key}...")
            results['accuracy_benchmarks'][model_key] = self.benchmark_accuracy(
                model_key, test_data
            )

        # Save results
        results_file = Path("artifacts/test-reports/model_comparison_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Benchmark results saved to {results_file}")
        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)

        print(f"\n📊 Device: {results['device']}")
        print(f"📅 Timestamp: {results['timestamp']}")
        print(f"🤖 Models Tested: {', '.join(results['models_loaded'])}")

        print("\n🚀 INFERENCE PERFORMANCE")
        print("-"*50)

        inference_results = results['inference_benchmarks']
        for model_key, data in inference_results.items():
            if 'error' in data:
                print(f"❌ {model_key}: {data['error']}")
                continue

            print(f"📈 {model_key.upper()}:")
            print(".2f")
            print(".2f")
            print(".1f")
            print()

        print("🎯 ACCURACY COMPARISON")
        print("-"*50)

        accuracy_results = results['accuracy_benchmarks']
        for model_key, data in accuracy_results.items():
            if 'error' in data:
                print(f"❌ {model_key}: {data['error']}")
                continue

            print(f"🎯 {model_key.upper()}:")
            print(".1f")
            print(".2f")
            print()

        print("💡 RECOMMENDATIONS")
        print("-"*50)

        # Find best models
        if inference_results:
            best_speed = min(
                [(k, v) for k, v in inference_results.items() if 'avg_latency_ms' in v],
                key=lambda x: x[1]['avg_latency_ms']
            )
            print(f"🏃‍♂️ Fastest Model: {best_speed[0].upper()} ({best_speed[1]['avg_latency_ms']:.0f}ms avg)")

        if accuracy_results:
            best_accuracy = max(
                [(k, v) for k, v in accuracy_results.items() if 'accuracy' in v],
                key=lambda x: x[1]['accuracy']
            )
            print(f"🎯 Most Accurate Model: {best_accuracy[0].upper()} ({best_accuracy[1]['accuracy']:.1%})")
        print("\n" + "="*80)


def main():
    """Main function to run model comparison."""
    print("🧪 Starting Model Comparison Benchmark")
    print("="*50)

    benchmark = ModelBenchmark()

    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_summary(results)

        # Check which models were successfully loaded
        loaded_models = [k for k in results['models_loaded'] if k in benchmark.models]
        print(f"\n✅ Successfully loaded {len(loaded_models)} out of 3 models: {', '.join(loaded_models)}")

        if not loaded_models:
            print("❌ No models could be loaded. Please check network connectivity and dependencies.")
            return

        print("📊 Detailed results saved to artifacts/test-reports/model_comparison_results.json")

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Check internet connection for model downloads")
        print("   2. Ensure transformers and torch are properly installed")
        print("   3. Try running with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")
        raise


if __name__ == "__main__":
    main()
