#!/usr/bin/env python3
"""
Evaluate Whisper model performance using LibriSpeech test set.

This script downloads a portion of the LibriSpeech test-clean dataset
and evaluates the Word Error Rate (WER) of the Whisper transcription model.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import jiwer
import pandas as pd
import soundfile as sf
import torch
import tqdm
from datasets import load_dataset

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.models.voice_processing.transcription_api import TranscriptionAPI, create_transcription_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_librispeech_sample(
    output_dir: Optional[str] = None, max_samples: int = 50
) -> list[dict]:
    """Download LibriSpeech test-clean sample for evaluation.

    Args:
        output_dir: Directory to save audio files (uses temp dir if None)
        max_samples: Maximum number of samples to download

    Returns:
        List of dicts with audio path and reference text
    """
    logger.info(f"Loading LibriSpeech test-clean (max_samples={max_samples})...")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="librispeech_")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    try:
        dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

        samples = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            audio = sample["audio"]
            text = sample["text"]

            # Save audio to file
            audio_path = output_dir / f"sample_{i:04d}.wav"
            sf.write(audio_path, audio["array"], audio["sampling_rate"])

            # Store result
            samples.append({
                "audio_path": str(audio_path),
                "reference_text": text,
                "sample_id": i
            })

        logger.info(f"Downloaded {len(samples)} samples to {output_dir}")
        return samples

    except Exception as e:
        logger.error(f"Failed to download LibriSpeech samples: {e}")
        return []


def evaluate_wer(api: TranscriptionAPI, samples: list[dict], model_size: str) -> dict:
    """Evaluate WER on LibriSpeech samples.

    Args:
        api: Transcription API instance
        samples: List of sample dicts with audio_path and reference_text
        model_size: Model size identifier

    Returns:
        Dict with WER metrics and timing info
    """
    logger.info(f"Evaluating WER on {len(samples)} samples with {model_size} model...")

    results = []
    total_time = 0.0

    for sample in tqdm.tqdm(samples, desc="Processing samples"):
        audio_path = sample["audio_path"]
        reference_text = sample["reference_text"]

        # Transcribe
        start_time = time.time()
        try:
            transcription_result = api.transcribe(audio_path)
            processing_time = time.time() - start_time
            total_time += processing_time

            # Calculate WER
            hypothesis = transcription_result.text.lower()
            reference = reference_text.lower()
            wer_score = jiwer.wer(reference, hypothesis)

            # Store result
            results.append({
                "sample_id": sample["sample_id"],
                "reference": reference_text,
                "hypothesis": transcription_result.text,
                "wer": wer_score,
                "processing_time": processing_time,
                "language": transcription_result.language
            })

        except Exception as e:
            logger.warning(f"Failed to transcribe {audio_path}: {e}")
            results.append({
                "sample_id": sample["sample_id"],
                "reference": reference_text,
                "hypothesis": "",
                "wer": 1.0,
                "processing_time": 0.0,
                "language": "unknown",
                "error": str(e)
            })

    # Calculate metrics
    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_time = total_time / len(results)

        return {
            "model_size": model_size,
            "num_samples": len(results),
            "average_wer": avg_wer,
            "average_processing_time": avg_time,
            "total_processing_time": total_time,
            "detailed_results": results
        }
    else:
        return {
            "model_size": model_size,
            "num_samples": 0,
            "average_wer": 1.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "detailed_results": []
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER on LibriSpeech")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save results and audio files"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=50, 
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base", 
        help="Whisper model size (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true", 
        help="Save detailed results to JSON file"
    )

    args = parser.parse_args()

    # Create output directory if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None

    # Download or load LibriSpeech samples
    samples = download_librispeech_sample(
        output_dir=args.output_dir, 
        max_samples=args.max_samples
    )

    if not samples:
        logger.error("No samples available for evaluation")
        return

    # Create TranscriptionAPI
    api = create_transcription_api()

    # Run evaluation
    results = evaluate_wer(api, samples, args.model_size)

    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Model: {results['model_size']}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info(f"Average WER: {results['average_wer']:.4f}")
    logger.info(f"Average Processing Time: {results['average_processing_time']:.3f}s")
    logger.info(f"Total Processing Time: {results['total_processing_time']:.3f}s")

    # Save results if output directory provided
    if args.save_results and output_dir:
        results_file = output_dir / "wer_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {results_file}")

        # Save summary
        summary_file = output_dir / "wer_summary.csv"
        df = pd.DataFrame(results["detailed_results"])
        df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
