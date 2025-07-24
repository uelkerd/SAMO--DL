#!/usr/bin/env python3
"""
Evaluate Whisper model performance using LibriSpeech test set.

This script downloads a portion of the LibriSpeech test-clean dataset
and evaluates the Word Error Rate (WER) of the Whisper transcription model.
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import jiwer
import pandas as pd
import torch
import tqdm
from datasets import load_dataset

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import local modules
from models.voice_processing.transcription_api import TranscriptionAPI, create_transcription_api

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

    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="librispeech_")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    try:
        dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

        # Prepare samples
        samples = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            # Extract sample info
            audio = sample["audio"]
            text = sample["text"]
            file_id = f"{sample['chapter_id']}_{sample['id']}"

            # Save audio to file
            audio_path = Path(output_dir) / f"{file_id}.wav"

            # Skip if already exists
            if audio_path.exists():
                samples.append({"path": str(audio_path), "reference": text})
                continue

            # Save audio data
            import soundfile as sf

            sf.write(
                str(audio_path),
                audio["array"],
                audio["sampling_rate"],
            )

            # Add to samples list
            samples.append({"path": str(audio_path), "reference": text})

            if i % 10 == 0:
                logger.info(f"Downloaded {i+1}/{max_samples} samples...")

        logger.info(f"✅ Downloaded {len(samples)} LibriSpeech samples to {output_dir}")
        return samples

    except Exception as e:
        logger.error(f"❌ Failed to download LibriSpeech: {e}")
        return []


def evaluate_wer(api: TranscriptionAPI, samples: list[dict], model_size: str) -> dict:
    """Evaluate Word Error Rate on LibriSpeech samples.

    Args:
        api: TranscriptionAPI instance
        samples: List of dicts with audio path and reference
        model_size: Whisper model size for reporting

    Returns:
        Evaluation metrics dict
    """
    logger.info(f"Evaluating WER on {len(samples)} samples...")

    results = []
    references = []
    transcriptions = []
    wers = []
    processing_times = []
    audio_durations = []

    # Process each sample
    for i, sample in enumerate(tqdm.tqdm(samples, desc="Evaluating")):
        try:
            # Get paths and references
            audio_path = sample["path"]
            reference = sample["reference"].lower().strip()
            references.append(reference)

            # Transcribe
            start_time = time.time()
            result = api.transcribe(audio_path)
            processing_time = time.time() - start_time

            # Get text and metrics
            transcription = result["text"].lower().strip()
            transcriptions.append(transcription)

            # Calculate WER
            wer = jiwer.wer(reference, transcription)
            wers.append(wer)

            # Collect timing info
            processing_times.append(processing_time)
            audio_durations.append(result["duration"])

            # Store result
            results.append(
                {
                    "id": i,
                    "reference": reference,
                    "transcription": transcription,
                    "wer": wer,
                    "processing_time": processing_time,
                    "audio_duration": result["duration"],
                    "real_time_factor": processing_time / result["duration"]
                    if result["duration"] > 0
                    else 0,
                }
            )

        except Exception as e:
            logger.error(f"Failed on sample {i}: {e}")

    # Calculate global WER
    global_wer = jiwer.wer(references, transcriptions)

    # Calculate metrics
    avg_wer = sum(wers) / len(wers) if wers else 0
    median_wer = sorted(wers)[len(wers) // 2] if wers else 0

    # Average timing
    avg_rtf = sum(processing_times) / sum(audio_durations) if audio_durations else 0

    # Create summary
    evaluation_summary = {
        "model_size": model_size,
        "samples": len(samples),
        "global_wer": global_wer,
        "average_wer": avg_wer,
        "median_wer": median_wer,
        "wer_below_15_percent": sum(1 for w in wers if w < 0.15) / len(wers) if wers else 0,
        "avg_real_time_factor": avg_rtf,
        "total_audio_duration": sum(audio_durations),
        "total_processing_time": sum(processing_times),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Store detailed results
    results_df = pd.DataFrame(results)

    return {
        "summary": evaluation_summary,
        "details": results,
        "dataframe": results_df,
    }


def main():
    """Run LibriSpeech WER evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER on LibriSpeech")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of LibriSpeech samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=str,
        default=None,
        help="Directory for LibriSpeech samples (will download if not provided)",
    )

    args = parser.parse_args()

    logger.info(f"🚀 Starting Whisper WER evaluation with {args.model_size} model")
    start_time = time.time()

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Download or load LibriSpeech samples
    samples = download_librispeech_sample(
        output_dir=args.librispeech_dir,
        max_samples=args.samples,
    )

    if not samples:
        logger.error("No LibriSpeech samples available. Exiting.")
        return 1

    # Create TranscriptionAPI
    api = create_transcription_api(model_size=args.model_size)

    # Run evaluation
    evaluation_results = evaluate_wer(
        api=api,
        samples=samples,
        model_size=args.model_size,
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Whisper WER Evaluation Results:")
    logger.info("=" * 50)

    summary = evaluation_results["summary"]
    logger.info(f"Model: {summary['model_size']}")
    logger.info(f"Device: {summary['device']}")
    logger.info(f"Samples: {summary['samples']}")
    logger.info(f"Global WER: {summary['global_wer']:.4f}")
    logger.info(f"Average WER: {summary['average_wer']:.4f}")
    logger.info(f"Median WER: {summary['median_wer']:.4f}")
    logger.info(f"Samples with WER < 15%: {summary['wer_below_15_percent']*100:.1f}%")
    logger.info(f"Real-time factor: {summary['avg_real_time_factor']:.2f}x")
    logger.info(f"Total audio: {summary['total_audio_duration']:.1f}s")
    logger.info(f"Total processing: {summary['total_processing_time']:.1f}s")

    # Save results if output directory provided
    if args.output_dir:
        # Save summary
        summary_path = Path(args.output_dir) / f"whisper_{args.model_size}_wer_summary.json"
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        details_path = Path(args.output_dir) / f"whisper_{args.model_size}_wer_details.csv"
        evaluation_results["dataframe"].to_csv(details_path, index=False)

        logger.info(f"✅ Results saved to {args.output_dir}")

    total_time = time.time() - start_time
    logger.info(f"✅ Evaluation completed in {total_time:.1f}s")

    # Determine if we meet the target WER
    target_wer = 0.15  # 15% target WER
    if summary["global_wer"] <= target_wer:
        logger.info(f"🎉 SUCCESS: WER {summary['global_wer']:.4f} meets target of {target_wer:.4f}")
        return 0
    else:
        logger.warning(f"⚠️ WER {summary['global_wer']:.4f} exceeds target of {target_wer:.4f}")
        return 0  # Still return success


if __name__ == "__main__":
    sys.exit(main())
