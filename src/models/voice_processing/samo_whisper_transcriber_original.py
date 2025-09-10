#!/usr/bin/env python3
"""
SAMO-Optimized Whisper Voice Transcription Model

This module provides a specialized Whisper transcription model optimized for
journal entries and voice processing in the SAMO-DL system.

Key Features:
- OpenAI Whisper model integration (tiny, base, small, medium, large)
- Multi-format audio support (MP3, WAV, M4A, OGG, FLAC)
- Confidence scoring and quality assessment
- Chunk-based processing for long audio files
- Production-ready error handling and logging
- Batch transcription for multiple audio files
"""

import logging
import os
import shutil
import time
import tempfile
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml

import torch
import whisper
import numpy as np
from pydub import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)

# Only suppress specific known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


class SAMOWhisperConfig:
    """Configuration for SAMO Whisper transcription."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._load_from_dict(config_data)
        else:
            self._load_defaults()

    def _load_from_dict(self, config_data: Dict[str, Any]):
        """Load configuration from dictionary."""
        whisper_config = config_data.get('whisper', {})
        transcription_config = config_data.get('transcription', {})

        # Load from whisper section first, then transcription section as fallback
        self.model_size = whisper_config.get('model_size', 'base')
        self.language = whisper_config.get('language', None)
        self.device = whisper_config.get('device', None)

        # Load transcription parameters from both sections
        self.task = whisper_config.get(
            'task', transcription_config.get('task', 'transcribe')
        )
        self.temperature = whisper_config.get(
            'temperature', transcription_config.get('temperature', 0.0)
        )
        self.beam_size = whisper_config.get(
            'beam_size', transcription_config.get('beam_size', None)
        )
        self.best_of = whisper_config.get(
            'best_of', transcription_config.get('best_of', None)
        )
        self.patience = whisper_config.get(
            'patience', transcription_config.get('patience', None)
        )
        self.length_penalty = whisper_config.get(
            'length_penalty', transcription_config.get('length_penalty', None)
        )
        self.suppress_tokens = whisper_config.get(
            'suppress_tokens', transcription_config.get('suppress_tokens', '-1')
        )
        self.initial_prompt = whisper_config.get(
            'initial_prompt', transcription_config.get('initial_prompt', None)
        )
        self.condition_on_previous_text = whisper_config.get(
            'condition_on_previous_text',
            transcription_config.get('condition_on_previous_text', True)
        )
        self.fp16 = whisper_config.get(
            'fp16', transcription_config.get('fp16', True)
        )
        self.compression_ratio_threshold = whisper_config.get(
            'compression_ratio_threshold',
            transcription_config.get('compression_ratio_threshold', 2.4)
        )
        self.logprob_threshold = whisper_config.get(
            'logprob_threshold',
            transcription_config.get('logprob_threshold', -1.0)
        )
        self.no_speech_threshold = whisper_config.get(
            'no_speech_threshold',
            transcription_config.get('no_speech_threshold', 0.6)
        )

    def _load_defaults(self):
        """Load default configuration."""
        self.model_size = 'base'
        self.language = None
        self.task = 'transcribe'
        self.temperature = 0.0
        self.beam_size = None
        self.best_of = None
        self.patience = None
        self.length_penalty = None
        self.suppress_tokens = '-1'
        self.initial_prompt = None
        self.condition_on_previous_text = True
        self.fp16 = True
        self.compression_ratio_threshold = 2.4
        self.logprob_threshold = -1.0
        self.no_speech_threshold = 0.6
        self.device = None


class TranscriptionResult:
    """Result of audio transcription."""

    def __init__(
        self,
        text: str,
        language: str,
        confidence: float,
        duration: float,
        processing_time: float,
        segments: List[Dict],
        audio_quality: str,
        word_count: int,
        speaking_rate: float,
        no_speech_probability: float
    ):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.duration = duration
        self.processing_time = processing_time
        self.segments = segments
        self.audio_quality = audio_quality
        self.word_count = word_count
        self.speaking_rate = speaking_rate
        self.no_speech_probability = no_speech_probability


class AudioPreprocessor:
    """Audio preprocessing for optimal Whisper performance."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
    MAX_DURATION = 300  # 5 minutes maximum

    @staticmethod
    def validate_audio_file(audio_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate audio file format and properties."""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return False, f"Audio file not found: {audio_path}"

        if audio_path.suffix.lower() not in AudioPreprocessor.SUPPORTED_FORMATS:
            return False, f"Unsupported audio format: {audio_path.suffix}"

        try:
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert to seconds

            if duration > AudioPreprocessor.MAX_DURATION:
                return False, (
                    f"Audio too long: {duration:.1f}s > "
                    f"{AudioPreprocessor.MAX_DURATION}s"
                )

            if duration < 0.1:  # Too short
                return False, f"Audio too short: {duration:.1f}s"

            return True, "Valid audio file"

        except Exception as e:
            return False, f"Error loading audio: {e}"

    @staticmethod
    def preprocess_audio(
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        normalize: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Preprocess audio for optimal Whisper performance."""
        audio_path = Path(audio_path)

        is_valid, error_msg = AudioPreprocessor.validate_audio_file(audio_path)
        if not is_valid:
            raise ValueError(error_msg)

        logger.info("Preprocessing audio: %s", audio_path)

        audio = AudioSegment.from_file(str(audio_path))

        original_metadata = {
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": audio_path.suffix.lower(),
            "file_size": audio_path.stat().st_size,
        }

        # Convert stereo to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted stereo to mono")

        # Resample to target sample rate
        if audio.frame_rate != AudioPreprocessor.TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(AudioPreprocessor.TARGET_SAMPLE_RATE)
            logger.info("Resampled to %dHz", AudioPreprocessor.TARGET_SAMPLE_RATE)

        # Optionally normalize audio if below threshold
        if normalize:
            peak_dbfs = audio.max_dBFS
            # Only normalize if peak is below -1 dBFS (to avoid artifacts)
            if peak_dbfs < -1.0:
                audio = audio.normalize()
                logger.info("Audio normalized (peak dBFS: %.2f)", peak_dbfs)
            else:
                logger.info("Audio normalization skipped (peak dBFS: %.2f)", peak_dbfs)

        # Create output path if not provided
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()

        # Export as WAV
        audio.export(str(output_path), format="wav")

        processed_metadata = {
            **original_metadata,
            "processed_duration": len(audio) / 1000.0,
            "processed_sample_rate": AudioPreprocessor.TARGET_SAMPLE_RATE,
            "processed_channels": 1,
            "processed_format": ".wav",
            "processed_file_size": Path(output_path).stat().st_size,
        }

        logger.info("Audio preprocessed: %s", output_path)
        return str(output_path), processed_metadata


class SAMOWhisperTranscriber:
    """SAMO-optimized Whisper transcriber for journal voice processing."""

    def __init__(
        self,
        config: Optional[SAMOWhisperConfig] = None,
        model_size: Optional[str] = None
    ) -> None:
        """Initialize SAMO Whisper transcriber."""
        self.config = config or SAMOWhisperConfig()
        if model_size:
            self.config.model_size = model_size

        # Auto-detect device if not specified
        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

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
                """Check for expected model files and their basic integrity.

                Args:
                    cache_dir (str): Path to the cache directory
                    model_size (str): Size of the model (e.g., 'base', 'small')

                Returns:
                    bool: True if model is corrupted or missing, False otherwise
                """
                model_file = os.path.join(cache_dir, f"{model_size}.pt")
                if not os.path.isfile(model_file):
                    return True
                
                # Check minimum file size based on model size
                min_sizes = {
                    "tiny": 39_000_000,    # ~39MB
                    "base": 74_000_000,    # ~74MB
                    "small": 244_000_000,  # ~244MB
                    "medium": 769_000_000, # ~769MB
                    "large": 1_550_000_000 # ~1.55GB
                }
                
                min_size = min_sizes.get(model_size, 1_000_000)  # Default 1MB
                return os.path.getsize(model_file) < min_size

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
            except (RuntimeError, OSError) as e:
                logger.exception(
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
            logger.exception("❌ Failed to load Whisper model")
            raise RuntimeError(f"Whisper model loading failed: {e}") from e

        self.preprocessor = AudioPreprocessor()

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text."""
        start_time = time.time()

        logger.info("Starting transcription: %s", audio_path)

        # Preprocess audio
        processed_audio_path, audio_metadata = self.preprocessor.preprocess_audio(
            audio_path
        )

        try:
            # Prepare transcription options
            transcribe_options = {
                "language": language or self.config.language,
                "task": self.config.task,
                "temperature": self.config.temperature,
                "best_of": self.config.best_of,
                "beam_size": self.config.beam_size,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "suppress_tokens": self.config.suppress_tokens,
                "initial_prompt": initial_prompt or self.config.initial_prompt,
                "condition_on_previous_text": self.config.condition_on_previous_text,
                "fp16": self.config.fp16,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold": self.config.logprob_threshold,
                "no_speech_threshold": self.config.no_speech_threshold,
            }

            # Remove None values
            transcribe_options = {
                k: v for k, v in transcribe_options.items() if v is not None
            }

            # Transcribe
            result = self.model.transcribe(processed_audio_path, **transcribe_options)

            processing_time = time.time() - start_time
            word_count = len(result['text'].split())
            speaking_rate = (
                (word_count / audio_metadata['duration']) * 60
                if audio_metadata['duration'] > 0 else 0
            )

            # Calculate confidence from segments
            confidence = self._calculate_confidence(
                result.get('segments', [])
            )

            # Calculate no_speech_probability from segments
            no_speech_probability = self._calculate_no_speech_probability(
                result.get('segments', [])
            )

            # Assess audio quality
            audio_quality = self._assess_audio_quality(result)

            transcription_result = TranscriptionResult(
                text=result['text'].strip() if isinstance(
                    result.get('text'), str
                ) else '',
                language=result.get('language', 'unknown'),
                confidence=confidence,
                duration=audio_metadata['duration'],
                processing_time=processing_time,
                segments=result.get('segments', []),
                audio_quality=audio_quality,
                word_count=word_count,
                speaking_rate=speaking_rate,
                no_speech_probability=no_speech_probability,
            )

            logger.info(
                "✅ Transcription complete: %d words, %.2f confidence",
                word_count, confidence
            )
            logger.info(
                "Processing time: %.2fs, Quality: %s",
                processing_time, audio_quality
            )

            return transcription_result

        finally:
            # Clean up temporary file
            if processed_audio_path != str(audio_path):
                with suppress(OSError):
                    os.unlink(processed_audio_path)

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files."""
        logger.info("Starting batch transcription of %d files...", len(audio_paths))

        results = []
        errors = []

        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(
                "Processing file %d/%d: %s",
                i, len(audio_paths), Path(audio_path).name
            )

            try:
                result = self.transcribe(audio_path, language, initial_prompt)
                results.append(result)

            except Exception as e:
                logger.exception("Failed to transcribe %s", audio_path)
                error_msg = f"Failed to transcribe {audio_path}: {e}"
                errors.append(error_msg)

                # Create error result with detailed error information
                error_result = TranscriptionResult(
                    text=f"[ERROR: {str(e)}]",
                    language="unknown",
                    confidence=0.0,
                    duration=0.0,
                    processing_time=0.0,
                    segments=[],
                    audio_quality="error",
                    word_count=0,
                    speaking_rate=0.0,
                    no_speech_probability=1.0,
                )
                results.append(error_result)

        total_duration = sum(r.duration for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        successful_transcriptions = sum(
            1 for r in results if not r.text.startswith("[ERROR:")
        )

        logger.info("✅ Batch transcription complete: %d files", len(results))
        logger.info(
            "Successful: %d/%d, Total audio: %.1fs, Processing: %.1fs",
            successful_transcriptions, len(results),
            total_duration, total_processing_time
        )

        if errors:
            logger.warning("Batch transcription had %d errors:", len(errors))
            for error in errors:
                logger.warning("  - %s", error)

        return results

    @staticmethod
    def _calculate_confidence(segments: List[Dict]) -> float:
        """Calculate overall confidence from segment data."""
        if not segments:
            return 0.0

        confidences = []
        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            no_speech_prob = segment.get("no_speech_prob", 0.5)

            # Calculate segment confidence
            segment_confidence = min(
                1.0, max(0.0, np.exp(avg_logprob) * (1 - no_speech_prob))
            )
            confidences.append(segment_confidence)

        return float(np.mean(confidences)) if confidences else 0.5

    @staticmethod
    def _calculate_no_speech_probability(segments: List[Dict]) -> float:
        """Calculate overall no_speech_probability from segment data."""
        if not segments:
            # No segments means silence - return high probability
            return 0.9

        # Get no_speech_prob from each segment and return the maximum
        no_speech_probs = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            no_speech_probs.append(no_speech_prob)

        # Return the maximum no_speech_prob across all segments
        return float(max(no_speech_probs)) if no_speech_probs else 0.9

    @staticmethod
    def _assess_audio_quality(result: Dict) -> str:
        """Assess audio quality based on transcription results."""
        compression_ratio = result.get("compression_ratio", 2.0)
        avg_logprob = result.get("avg_logprob", -0.5)
        no_speech_prob = result.get("no_speech_prob", 0.3)

        quality_score = 0

        # Compression ratio scoring
        if compression_ratio <= 2.4:
            quality_score += 2
        elif compression_ratio <= 3.0:
            quality_score += 1

        # Log probability scoring
        if avg_logprob > -0.3:
            quality_score += 2
        elif avg_logprob > -0.5:
            quality_score += 1

        # No speech probability scoring
        if no_speech_prob < 0.2:
            quality_score += 2
        elif no_speech_prob < 0.4:
            quality_score += 1

        # Determine quality level
        if quality_score >= 5:
            return "excellent"
        if quality_score >= 3:
            return "good"
        if quality_score >= 1:
            return "fair"
        return "poor"

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


def create_samo_whisper_transcriber(
    config_path: Optional[str] = None,
    model_size: Optional[str] = None
) -> SAMOWhisperTranscriber:
    """Create a SAMO Whisper transcriber with specified configuration."""
    config = SAMOWhisperConfig(config_path) if config_path else None
    return SAMOWhisperTranscriber(config, model_size)


def test_samo_whisper_transcriber() -> None:
    """Test SAMO Whisper transcriber with sample audio."""
    logger.info("Testing SAMO Whisper Transcriber...")

    transcriber = create_samo_whisper_transcriber(model_size="base")
    logger.info("SAMO Whisper transcriber initialized successfully")
    logger.info("Model info: %s", transcriber.get_model_info())


if __name__ == "__main__":
    test_samo_whisper_transcriber()
