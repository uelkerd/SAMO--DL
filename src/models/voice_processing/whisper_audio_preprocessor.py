#!/usr/bin/env python3
"""
SAMO Whisper Audio Preprocessing Module

This module handles audio preprocessing for optimal Whisper performance,
including format validation, resampling, normalization, and quality assessment.
"""

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from pydub import AudioSegment

# Suppress warnings from audio processing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


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
                return False, f"Audio too long: {duration:.1f}s > {AudioPreprocessor.MAX_DURATION}s"

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

    @staticmethod
    def assess_audio_quality(result: Dict, metadata: Dict) -> str:
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
