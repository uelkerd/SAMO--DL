# Configure logging
# G004: Logging f-strings temporarily allowed for development
from pathlib import Path
from pydub import AudioSegment
from typing import Optional, Union
import logging
import tempfile



"""Audio Preprocessing for SAMO Voice Processing.

This module provides audio format handling and preprocessing functionality
for optimal OpenAI Whisper performance.

Key Features:
- Multi-format audio support (MP3, WAV, M4A, OGG, etc.)
- Audio validation and format conversion
- Sample rate normalization to 16kHz
- Mono conversion and noise reduction
- Metadata extraction and quality assessment
"""

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing for optimal Whisper performance."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
    MAX_DURATION = 300  # 5 minutes maximum

    @staticmethod
    def validate_audio_file(audio_path: Union[str, Path]) -> tuple[bool, str]:
        """Validate audio file format and properties.

        Args:
            audio_path: Path to audio file

        Returns:
            tuple of (is_valid, error_message)
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
            return False, f"Error loading audio file: {exc}"

    @staticmethod
    def preprocess_audio(
        audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> tuple[str, dict]:
        """Preprocess audio for optimal Whisper performance.

        Args:
            audio_path: Input audio file path
            output_path: Output path (temporary file if None)

        Returns:
            tuple of (processed_audio_path, metadata)
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


def preprocess_audio(
    audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> tuple[str, dict]:
    """Convenience function for audio preprocessing."""
    return AudioPreprocessor.preprocess_audio(audio_path, output_path)
