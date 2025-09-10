"""
SAMO Whisper Model Management Module

This module handles Whisper model loading, caching, and management
with robust error handling and cache corruption detection.
"""

import logging
import os
import shutil
from typing import Dict, Any, Optional
import whisper

from .whisper_audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class WhisperModelManager:
    """Manages Whisper model loading and caching with error handling."""

    def __init__(self, config, device):
        """Initialize model manager."""
        self.config = config
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """Load Whisper model with cache corruption handling."""
        logger.info(
            "Initializing SAMO Whisper %s model...",
            self.config.model_size
        )
        logger.info("Device: %s", self.device)

        try:
            # Use cache directory from environment or create a local one
            cache_dir = os.environ.get(
                'HF_HOME', os.path.expanduser('~/.cache/whisper')
            )
            os.makedirs(cache_dir, exist_ok=True)

            def is_model_corrupted(cache_dir, model_size):
                """Check for expected model files and their integrity."""
                model_file = os.path.join(cache_dir, f"{model_size}.pt")
                return not (
                    os.path.isfile(model_file) and os.path.getsize(model_file) > 0
                )

            try:
                if is_model_corrupted(cache_dir, self.config.model_size):
                    logger.warning(
                        "Detected corrupted or missing model files in cache. "
                        "Clearing cache directory: %s", cache_dir
                    )
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                self.model = whisper.load_model(
                    self.config.model_size,
                    device=self.device,
                    download_root=cache_dir
                )
            except Exception:
                logger.error(
                    "Model loading failed, possibly due to cache corruption. "
                    "Clearing cache and retrying..."
                )
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                self.model = whisper.load_model(
                    self.config.model_size,
                    device=self.device,
                    download_root=cache_dir
                )
            logger.info(
                "✅ SAMO Whisper %s model loaded successfully",
                self.config.model_size
            )

        except Exception as e:
            logger.error("❌ Failed to load Whisper model: %s", e)
            raise RuntimeError(f"Whisper model loading failed: {e}")

    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_size": self.config.model_size,
            "device": str(self.device),
            "language": self.config.language or "auto-detect",
            "task": self.config.task,
            "supported_formats": list(AudioPreprocessor.SUPPORTED_FORMATS),
            "max_duration": AudioPreprocessor.MAX_DURATION,
            "target_sample_rate": AudioPreprocessor.TARGET_SAMPLE_RATE,
        }
