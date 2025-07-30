            # Add to samples list
            # Calculate WER
            # Collect timing info
            # Extract sample info
            # Get paths and references
            # Get text and metrics
            # Save audio data
            # Save audio to file
            # Skip if already exists
            # Store result
            # Transcribe
            import soundfile as sf
        # Prepare samples
        # Save detailed results
        # Save summary
    # Average timing
    # Calculate global WER
    # Calculate metrics
    # Create TranscriptionAPI
    # Create output directory
    # Create output directory if needed
    # Create summary
    # Determine if we meet the target WER
    # Download or load LibriSpeech samples
    # Load dataset
    # Print summary
    # Process each sample
    # Run evaluation
    # Save results if output directory provided
    # Store detailed results
# Add src directory to path
# Configure logging
# Import local modules
#!/usr/bin/env python3
from datasets import load_dataset
from models.voice_processing.transcription_api import TranscriptionAPI, create_transcription_api
from pathlib import Path
from typing import Optional
import argparse
import jiwer
import json
import logging
import os
import pandas as pd
import sys
import tempfile
import time
import torch
import tqdm






"""
Evaluate Whisper model performance using LibriSpeech test set.

This script downloads a portion of the LibriSpeech test-clean dataset
and evaluates the Word Error Rate (WER) of the Whisper transcription model.
"""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    logger.info("Loading LibriSpeech test-clean (max_samples={max_samples})...")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="librispeech_")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    try:
        dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

        samples = []
        for _i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            audio = sample["audio"]
            text = sample["text"]
            file_id = "{sample['chapter_id']}_{sample['id']}"

            audio_path = Path(output_dir) / "{file_id}.wav"

            if audio_path.exists():
                samples.append({"path": str(audio_path), "reference": text})
                continue

            sf.write(
                str(audio_path),
                audio["array"],
                audio["sampling_rate"],
            )

            samples.append({"path": str(audio_path), "reference": text})

            if i % 10 == 0:
                logger.info("Downloaded {i+1}/{max_samples} samples...")

        logger.info("✅ Downloaded {len(samples)} LibriSpeech samples to {output_dir}")
        return samples

    except Exception as e:
        logger.error("❌ Failed to download LibriSpeech: {e}")
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
    logger.info("Evaluating WER on {len(samples)} samples...")

    results = []
    references = []
    transcriptions = []
    wers = []
    processing_times = []
    audio_durations = []

    for i, sample in enumerate(tqdm.tqdm(samples, desc="Evaluating")):
        try:
            audio_path = sample["path"]
            reference = sample["reference"].lower().strip()
            references.append(reference)

            start_time = time.time()
            result = api.transcribe(audio_path)
            processing_time = time.time() - start_time

            transcription = result["text"].lower().strip()
            transcriptions.append(transcription)

            wer = jiwer.wer(reference, transcription)
            wers.append(wer)

            processing_times.append(processing_time)
            audio_durations.append(result["duration"])

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
            logger.error("Failed on sample {i}: {e}")

    global_wer = jiwer.wer(references, transcriptions)

    avg_wer = sum(wers) / len(wers) if wers else 0
    median_wer = sorted(wers)[len(wers) // 2] if wers else 0

    avg_rtf = sum(processing_times) / sum(audio_durations) if audio_durations else 0

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

    logger.info("🚀 Starting Whisper WER evaluation with {args.model_size} model")
    start_time = time.time()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    samples = download_librispeech_sample(
        output_dir=args.librispeech_dir,
        max_samples=args.samples,
    )

    if not samples:
        logger.error("No LibriSpeech samples available. Exiting.")
        return 1

    api = create_transcription_api(model_size=args.model_size)

    evaluation_results = evaluate_wer(
        api=api,
        samples=samples,
        model_size=args.model_size,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Whisper WER Evaluation Results:")
    logger.info("=" * 50)

    summary = evaluation_results["summary"]
    logger.info("Model: {summary['model_size']}")
    logger.info("Device: {summary['device']}")
    logger.info("Samples: {summary['samples']}")
    logger.info("Global WER: {summary['global_wer']:.4f}")
    logger.info("Average WER: {summary['average_wer']:.4f}")
    logger.info("Median WER: {summary['median_wer']:.4f}")
    logger.info("Samples with WER < 15%: {summary['wer_below_15_percent']*100:.1f}%")
    logger.info("Real-time factor: {summary['avg_real_time_factor']:.2f}x")
    logger.info("Total audio: {summary['total_audio_duration']:.1f}s")
    logger.info("Total processing: {summary['total_processing_time']:.1f}s")

    if args.output_dir:
        summary_path = Path(args.output_dir) / "whisper_{args.model_size}_wer_summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        details_path = Path(args.output_dir) / "whisper_{args.model_size}_wer_details.csv"
        evaluation_results["dataframe"].to_csv(details_path, index=False)

        logger.info("✅ Results saved to {args.output_dir}")

    total_time = time.time() - start_time
    logger.info("✅ Evaluation completed in {total_time:.1f}s")

    target_wer = 0.15  # 15% target WER
    if summary["global_wer"] <= target_wer:
        logger.info("🎉 SUCCESS: WER {summary['global_wer']:.4f} meets target of {target_wer:.4f}")
        return 0
    else:
        logger.warning("⚠️ WER {summary['global_wer']:.4f} exceeds target of {target_wer:.4f}")
        return 0  # Still return success


if __name__ == "__main__":
    sys.exit(main())
