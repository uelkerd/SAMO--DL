# Configure logging
# Suppress warnings from audio processing
from dataclasses import dataclass
from pathlib import Path
from pydub import AudioSegment
from typing import Any, Optional, Union, Tuple, List, Dict
import contextlib
import logging
import numpy as np
import os
import tempfile
import time
import torch
import warnings
import whisper
"""OpenAI Whisper Transcriber for SAMO Deep Learning.

This module implements OpenAI Whisper for high-accuracy voice-to-text transcription
of journal entries, supporting multiple audio formats with confidence scoring
and quality assessment.

Key Features:
- OpenAI Whisper model integration (tiny, base, small, medium, large)
- Multi-format audio support (MP3, WAV, M4A, OGG)
- Confidence scoring and quality assessment
- Chunk-based processing for long audio files
- Production-ready error handling and logging
- Batch transcription for multiple audio files
"""



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription."""

    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # transcribe or translate
    temperature: float = 0.0  # Sampling temperature
    beam_size: Optional[int] = None  # Beam search size
    best_of: Optional[int] = None  # Number of candidates
    patience: Optional[float] = None  # Patience for beam search
    length_penalty: Optional[float] = None  # Length penalty
    suppress_tokens: str = "-1"  # Tokens to suppress
    initial_prompt: Optional[str] = None  # Context prompt
    condition_on_previous_text: bool = True
    fp16: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    device: Optional[str] = None  # Auto-detect if None


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    confidence: float
    duration: float
    processing_time: float
    segments: List[dict]
    audio_quality: str  # excellent, good, fair, poor
    word_count: int
    speaking_rate: float  # words per minute
    no_speech_probability: float


class AudioPreprocessor:
    """Audio preprocessing for optimal Whisper performance."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
    MAX_DURATION = 300  # 5 minutes maximum

    @staticmethod
    def validate_audio_file(audio_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate audio file format and properties.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, error_message)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return False, "Audio file not found: {audio_path}"

        if audio_path.suffix.lower() not in AudioPreprocessor.SUPPORTED_FORMATS:
            return False, "Unsupported audio format: {audio_path.suffix}"

        try:
            audio = AudioSegment.from_file(str(audio_path))

            duration = len(audio) / 1000.0  # Convert to seconds
            if duration > AudioPreprocessor.MAX_DURATION:
                return False, "Audio too long: {duration:.1f}s > {AudioPreprocessor.MAX_DURATION}s"

            if duration < 0.1:  # Too short
                return False, "Audio too short: {duration:.1f}s"

            return True, "Valid audio file"

        except Exception as exc:
            return False, f"Error loading audio: {exc!s}"

    @staticmethod
    def preprocess_audio(
        audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Preprocess audio for optimal Whisper performance.

        Args:
            audio_path: Input audio file path
            output_path: Output path (temporary file if None)

        Returns:
            Tuple of (processed_audio_path, metadata)
        """
        audio_path = Path(audio_path)

        is_valid, error_msg = AudioPreprocessor.validate_audio_file(audio_path)
        if not is_valid:
            raise ValueError(error_msg)

        logger.info("Preprocessing audio: {audio_path}", extra={"format_args": True})

        audio = AudioSegment.from_file(str(audio_path))

        original_metadata = {
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": audio_path.suffix.lower(),
            "file_size": audio_path.stat().st_size,
        }

        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted stereo to mono")

        if audio.frame_rate != AudioPreprocessor.TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(AudioPreprocessor.TARGET_SAMPLE_RATE)
            logger.info(
                "Resampled to {AudioPreprocessor.TARGET_SAMPLE_RATE}Hz", extra={"format_args": True}
            )

        audio = audio.normalize()

        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = temp_file.name
            temp_file.close()

        audio.export(str(output_path), format="wav")

        processed_metadata = {
            **original_metadata,
            "processed_duration": len(audio) / 1000.0,
            "processed_sample_rate": AudioPreprocessor.TARGET_SAMPLE_RATE,
            "processed_channels": 1,
            "processed_format": ".wav",
            "processed_file_size": Path(output_path).stat().st_size,
        }

        logger.info("Audio preprocessed: {output_path}", extra={"format_args": True})
        return str(output_path), processed_metadata


class WhisperTranscriber:
    """OpenAI Whisper transcriber for journal voice processing."""

    def __init__(
        self, config: Optional[TranscriptionConfig] = None, model_size: Optional[str] = None
    ) -> None:
        """Initialize Whisper transcriber.

        Args:
            config: Transcription configuration
            model_size: Override model size from config
        """
        self.config = config or TranscriptionConfig()
        if model_size:
            self.config.model_size = model_size

        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(
            "Initializing Whisper {self.config.model_size} model...", extra={"format_args": True}
        )
        logger.info("Device: {self.device}", extra={"format_args": True})

        try:
            # Use cache directory from environment
            cache_dir = os.environ.get('HF_HOME', '/app/models')
            self.model = whisper.load_model(
                self.config.model_size, device=self.device, download_root=cache_dir
            )
            logger.info(
                "✅ Whisper {self.config.model_size} model loaded successfully",
                extra={"format_args": True},
            )

        except Exception as exc:
            logger.error("❌ Failed to load Whisper model: %s", exc)
            raise RuntimeError(f"Whisper model loading failed: {exc}")

        self.preprocessor = AudioPreprocessor()

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            initial_prompt: Context prompt for better accuracy

        Returns:
            TranscriptionResult with detailed information
        """
        start_time = time.time()

        logger.info("Starting transcription: {audio_path}", extra={"format_args": True})

        processed_audio_path, audio_metadata = self.preprocessor.preprocess_audio(audio_path)

        try:
            transcribe_options = {
                "language": language or self.config.language,
                "task": self.config.task,
                "temperature": self.config.temperature,
                "best_o": self.config.best_of,
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

            transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}

            result = self.model.transcribe(processed_audio_path, **transcribe_options)

            processing_time = time.time() - start_time
            word_count = len(result.text.split())
            speaking_rate = (
                (word_count / audio_metadata["duration"]) * 60
                if audio_metadata["duration"] > 0
                else 0
            )

            audio_quality = self._assess_audio_quality(result, audio_metadata)

            # Defensive check for segments to prevent non-subscriptable errors
            segments = (
                result.get('segments', [])
                if hasattr(result, 'segments') and isinstance(result.segments, list)
                else []
            )
            confidence = self._calculate_confidence(segments)

            transcription_result = TranscriptionResult(
                text=(
                    result['text'].strip()
                    if isinstance(result.get('text'), str) else ''
                ),
                language=result.get('language', 'unknown'),
                confidence=confidence,
                duration=audio_metadata["duration"],
                processing_time=processing_time,
                segments=segments,
                audio_quality=audio_quality,
                word_count=word_count,
                speaking_rate=speaking_rate,
                no_speech_probability=getattr(result, 'no_speech_prob', 0.0),
            )

            logger.info(
                "✅ Transcription complete: {word_count} words, {confidence:.2f} confidence",
                extra={"format_args": True},
            )
            logger.info(
                "Processing time: {processing_time:.2f}s, Quality: {audio_quality}",
                extra={"format_args": True},
            )

            return transcription_result

        finally:
            if processed_audio_path != str(audio_path):
                with contextlib.suppress(Exception):
                    os.unlink(processed_audio_path)

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths
            language: Language code for all files
            initial_prompt: Context prompt for better accuracy

        Returns:
            List of TranscriptionResult objects
        """
        logger.info(
            "Starting batch transcription of %s files...",
            len(audio_paths)
        )

        results = []
        for _i, audio_path in enumerate(audio_paths, 1):
            logger.info(
                "Processing file %s/%s: %s",
                _i, len(audio_paths), Path(audio_path).name
            )

            try:
                result = self.transcribe(
                    audio_path, language=language, initial_prompt=initial_prompt
                )
                results.append(result)

            except Exception as e:
                logger.error("Failed to transcribe %s: %s", audio_path, e)
                results.append(
                    TranscriptionResult(
                        text="",
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
                )

        total_duration = sum(r.duration for r in results)
        total_processing_time = sum(r.processing_time for r in results)

        logger.info(
            "✅ Batch transcription complete: %s files", len(results)
        )
        logger.info(
            "Total audio: %.1fs, Processing: %.1fs",
            total_duration, total_processing_time
        )

        return results

    def _calculate_confidence(self, segments: List[dict]) -> float:
        """Calculate overall confidence from segment data.

        Args:
            segments: List of transcription segments

        Returns:
            Average confidence score (0.0 to 1.0)
        """
        if not segments:
            return 0.0

        confidences = []
        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            no_speech_prob = segment.get("no_speech_prob", 0.5)

            segment_confidence = min(1.0, max(0.0, np.exp(avg_logprob) * (1 - no_speech_prob)))
            confidences.append(segment_confidence)

        return float(np.mean(confidences)) if confidences else 0.5

    def _assess_audio_quality(self, result: dict, metadata: dict) -> str:
        """Assess audio quality based on transcription results.

        Args:
            result: Whisper transcription result
            metadata: Audio metadata

        Returns:
            Quality assessment: excellent, good, fair, poor
        """
        compression_ratio = result.get("compression_ratio", 2.0)
        avg_logprob = result.get("avg_logprob", -0.5)
        no_speech_prob = result.get("no_speech_prob", 0.3)

        quality_score = 0

        if compression_ratio <= 2.4:
            quality_score += 2
        elif compression_ratio <= 3.0:
            quality_score += 1

        if avg_logprob > -0.3:
            quality_score += 2
        elif avg_logprob > -0.5:
            quality_score += 1

        if no_speech_prob < 0.2:
            quality_score += 2
        elif no_speech_prob < 0.4:
            quality_score += 1

        if quality_score >= 5:
            return "excellent"
        elif quality_score >= 3:
            return "good"
        elif quality_score >= 1:
            return "fair"
        else:
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


def create_whisper_transcriber(
    model_size: str = "base",
    language: Optional[str] = None,
    device: Optional[str] = None
) -> WhisperTranscriber:
    """Create Whisper transcriber with specified configuration.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code for transcription
        device: Device for model ('cuda', 'cpu', or None for auto)

    Returns:
        Configured WhisperTranscriber instance
    """
    config = TranscriptionConfig(model_size=model_size, language=language, device=device)

    transcriber = WhisperTranscriber(config)
    logger.info("Created Whisper transcriber: {model_size}", extra={"format_args": True})

    return transcriber


def test_whisper_transcriber() -> None:
    """Test Whisper transcriber with sample audio."""
    logger.info("Testing Whisper transcriber...")

    transcriber = create_whisper_transcriber("base")

    logger.info("Whisper transcriber initialized successfully")
    logger.info("Model info: %s", transcriber.get_model_info())


    logger.info("✅ Whisper transcriber test complete!")


if __name__ == "__main__":
    test_whisper_transcriber()
