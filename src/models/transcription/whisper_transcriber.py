"""
SAMO-optimized Whisper audio transcription model.

This module provides a high-performance Whisper-based audio transcription service
with SAMO-specific optimizations for journal audio processing.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def _load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to defaults."""
    default_config = {
        "model": {
            "name": "openai/whisper-base",
            "device": "auto",
            "torch_dtype": "float16"
        },
        "audio": {
            "sample_rate": 16000,
            "max_duration": 30.0,
            "chunk_length": 30.0,
            "stride_length": 5.0
        },
        "transcription": {
            "language": "auto",
            "task": "transcribe",
            "return_timestamps": True,
            "return_language": True,
            "chunk_length_s": 30.0,
            "stride_length_s": 5.0
        },
        "samo_optimizations": {
            "log_level": "INFO",
            "enable_chunking": True,
            "enable_vad": False,
            "confidence_threshold": 0.5,
            "max_retries": 3
        }
    }

    config = default_config.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
                config = _deep_merge(default_config.copy(), loaded)
        except Exception as e:
            logger.warning(
                "Failed to load config from %s: %s. Using default config.",
                config_path, e
            )

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries, with values from override taking precedence."""
    for key, value in override.items():
        if (key in base and
                isinstance(base[key], dict) and
                isinstance(value, dict)):
            base[key].update(value)
        else:
            base[key] = value
    return base


def _configure_logging(level_str: str) -> None:
    """Configure logging level from config."""
    level = getattr(logging, str(level_str).upper(), logging.INFO)
    logger.setLevel(level)


def _get_device(config: Dict[str, Any]) -> str:
    """Get the best available device, respecting user-specified device override."""
    # Check if user specified a device in config
    user_device = config.get("model", {}).get("device")
    if user_device and user_device != "auto":
        logger.info("Using user-specified device: %s", user_device)
        return user_device

    # Auto-detect best device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available, using GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS available, using Apple Silicon GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")

    return device


class SAMOWhisperTranscriber:
    """SAMO-optimized Whisper audio transcription model."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize the SAMO Whisper transcriber.

        Args:
            config_path: Path to configuration file
        """
        self.config = _load_config(config_path)
        log_level = self.config.get("samo_optimizations", {}).get("log_level", "INFO")
        _configure_logging(log_level)
        self.model = None
        self.processor = None
        self.device = _get_device(self.config)
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model and processor."""
        try:
            model_name = self.config["model"]["name"]
            logger.info("Loading Whisper model: %s", model_name)

            # Load processor
            self.processor = WhisperProcessor.from_pretrained(model_name)

            # Load model with appropriate dtype
            torch_dtype = getattr(torch, self.config["model"]["torch_dtype"])
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            )
            
            # Move to device after loading
            if self.device != "auto":
                self.model = self.model.to(self.device)

            logger.info("Whisper model loaded successfully on %s", self.device)

        except Exception as e:
            logger.exception("Failed to load Whisper model")
            raise RuntimeError("Model loading failed") from e

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'de', 'fr') or None for auto-detect
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Dictionary containing transcription results
        """
        start_time = time.time()

        try:
            # Load and preprocess audio
            audio_array, sample_rate = self._load_audio(audio_path)
            audio_array = self._preprocess_audio(audio_array, sample_rate)

            # Prepare inputs
            inputs = self.processor(
                audio_array,
                sampling_rate=self.config["audio"]["sample_rate"],
                return_tensors="pt"
            )
            
            # Move to device and ensure dtype consistency
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
                if hasattr(self.model, 'dtype'):
                    inputs[key] = inputs[key].to(self.model.dtype)

            # Transcribe
            with torch.inference_mode():
                # Handle language parameter - don't pass "auto" to Whisper
                whisper_language = language or self.config["transcription"]["language"]
                if whisper_language == "auto":
                    whisper_language = None  # Let Whisper auto-detect
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    language=whisper_language,
                    task=self.config["transcription"]["task"],
                    return_timestamps=return_timestamps
                )

            # Decode results
            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract timestamps if requested
            timestamps = None
            if return_timestamps:
                timestamps = self._extract_timestamps(generated_ids)

            processing_time = time.time() - start_time

            return {
                "text": transcription.strip(),
                "timestamps": timestamps,
                "language": language or "auto",
                "processing_time": processing_time,
                "audio_duration": len(audio_array) / self.config["audio"]["sample_rate"],
                "confidence": self._calculate_confidence(generated_ids)
            }

        except Exception as e:
            logger.exception("Transcription failed for %s", audio_path)
            raise RuntimeError(f"Transcription failed: {e}") from e

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        return_timestamps: bool = True
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files
            language: Language code or None for auto-detect
            return_timestamps: Whether to return word-level timestamps

        Returns:
            List of transcription results
        """
        start_time = time.time()
        results = []

        for i, audio_path in enumerate(audio_paths):
            try:
                logger.info("Transcribing file %d/%d: %s", i + 1, len(audio_paths), audio_path)
                result = self.transcribe_audio(audio_path, language, return_timestamps)
                results.append(result)
            except Exception as e:
                logger.error("Failed to transcribe %s: %s", audio_path, e)
                results.append({
                    "text": "",
                    "error": str(e),
                    "audio_path": str(audio_path)
                })

        total_time = time.time() - start_time
        logger.info("Batch transcription completed in %.2f seconds", total_time)

        return results

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Load audio file and return as tensor."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio using librosa
        waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)

        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()

        return waveform, sample_rate

    def _preprocess_audio(
        self, audio_array: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Preprocess audio for Whisper model."""
        target_sample_rate = self.config["audio"]["sample_rate"]

        # Resample if necessary using librosa
        if sample_rate != target_sample_rate:
            audio_array = librosa.resample(
                audio_array.numpy(),
                orig_sr=sample_rate,
                target_sr=target_sample_rate
            )
            audio_array = torch.from_numpy(audio_array).float()

        # Normalize audio
        audio_array = audio_array / torch.max(torch.abs(audio_array))

        return audio_array

    def _extract_timestamps(self, generated_ids: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from generated IDs."""
        # This is a simplified implementation
        # In practice, you'd need to decode the timestamp tokens
        return []

    def _calculate_confidence(self, generated_ids: torch.Tensor) -> float:
        """Calculate confidence score for transcription."""
        # Simplified confidence calculation
        # In practice, you'd use the model's logits
        return 0.85


def create_samo_whisper_transcriber(config_path: Union[str, Path]) -> SAMOWhisperTranscriber:
    """Create a SAMO Whisper transcriber instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured SAMOWhisperTranscriber instance
    """
    return SAMOWhisperTranscriber(config_path)
