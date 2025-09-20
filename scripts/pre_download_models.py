#!/usr/bin/env python3
"""Pre-download models for Docker build optimization with parallel downloads and retries."""
import concurrent.futures
import logging
import os
import time
from typing import Callable, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Tuple[bool, Optional[Exception]]:
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            func()
            return True, None
        except Exception as e:
            if attempt == max_retries - 1:
                return False, e

            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    return False, None


def download_samo_emotion_model(model_dir: str) -> None:
    """Download SAMO emotion model with fallback."""
    logger.info("🚀 Starting SAMO emotion model download...")

    def _download_primary():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "duelker/samo-goemotions-deberta-v3-large"
        logger.info(f"Downloading primary model: {model_name}")

        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_dir
        )
        logger.info("✅ Primary SAMO model downloaded successfully")

    def _download_fallback():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        fallback_model = "j-hartmann/emotion-english-distilroberta-base"
        logger.info(f"Downloading fallback model: {fallback_model}")

        _tokenizer = AutoTokenizer.from_pretrained(fallback_model, cache_dir=model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model, cache_dir=model_dir
        )
        logger.info("✅ Fallback emotion model downloaded successfully")

    # Try primary model first
    success, error = retry_with_backoff(_download_primary)
    if success:
        return

    logger.warning(f"Primary SAMO model failed: {error}. Trying fallback...")
    success, error = retry_with_backoff(_download_fallback)
    if not success:
        raise Exception(f"Both primary and fallback emotion models failed: {error}")


def download_t5_model(model_dir: str) -> None:
    """Download T5 summarization model."""
    logger.info("🚀 Starting T5 summarization model download...")

    def _download():
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        t5_model = "t5-small"
        logger.info(f"Downloading T5 model: {t5_model}")

        _tokenizer = T5Tokenizer.from_pretrained(t5_model, cache_dir=model_dir)
        _model = T5ForConditionalGeneration.from_pretrained(t5_model, cache_dir=model_dir)
        logger.info("✅ T5 model downloaded successfully")

    success, error = retry_with_backoff(_download)
    if not success:
        logger.warning(f"T5 model download failed: {error}")
        logger.info("📝 T5 will be downloaded at runtime if needed")


def download_whisper_model(model_dir: str) -> None:
    """Download Whisper model."""
    logger.info("🚀 Starting Whisper model download...")

    def _download():
        try:
            import whisper
            whisper_model = "base"
            logger.info(f"Downloading Whisper model: {whisper_model}")
            whisper.load_model(whisper_model, download_root=model_dir)
            logger.info("✅ Whisper model downloaded successfully")
        except ImportError:
            logger.warning("⚠️ Whisper not available - skipping download")
            logger.info("📝 Whisper will be downloaded at runtime if needed")
            return

    success, error = retry_with_backoff(_download)
    if not success:
        logger.warning(f"Whisper model download failed: {error}")
        logger.info("📝 Whisper will be downloaded at runtime if needed")


def main():
    """Pre-download all required models in parallel."""
    start_time = time.time()

    # Get model directory from environment variable, fallback to /app/models
    model_dir = os.environ.get("MODEL_DIR", "/app/models")
    os.makedirs(model_dir, exist_ok=True)

    # Set cache environment variables to use the same directory
    os.environ["HF_HOME"] = model_dir
    os.environ["TRANSFORMERS_CACHE"] = model_dir

    logger.info(f"📁 Using model directory: {model_dir}")
    logger.info("🚀 Starting parallel model downloads...")

    # Download models in parallel using ThreadPoolExecutor
    download_functions = [
        ("SAMO Emotion", lambda: download_samo_emotion_model(model_dir)),
        ("T5 Summarization", lambda: download_t5_model(model_dir)),
        ("Whisper Voice", lambda: download_whisper_model(model_dir)),
    ]

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all download tasks
        future_to_model = {
            executor.submit(func): name
            for name, func in download_functions
        }

        # Wait for completion and collect results
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                future.result()
                results[model_name] = "✅ Success"
                logger.info(f"📦 {model_name} model download completed")
            except Exception as e:
                results[model_name] = f"❌ Failed: {e}"
                logger.error(f"💥 {model_name} model download failed: {e}")

    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n📊 Download Summary (completed in {elapsed_time:.1f}s):")
    for model_name, status in results.items():
        logger.info(f"   • {model_name}: {status}")

    # Check if at least emotion model succeeded (critical for app functionality)
    if "✅ Success" not in results.get("SAMO Emotion", ""):
        logger.error("❌ Critical: Emotion model download failed completely")
        raise Exception("SAMO emotion model is required but failed to download")

    logger.info("🎉 Model pre-download process completed!")
    logger.info(f"⏱️  Total time: {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()
