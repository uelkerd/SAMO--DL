            # Calculate WER
            # Calculate additional metrics
            # Format results and update metrics
            # Get transcription
            # Perform transcription
            # Process batch through transcriber
            # Return evaluation
            # Return formatted response
            # Update metrics
            # Update processing time
            # Validate audio before transcription
        # Initialize transcriber
        # Track performance metrics
from .audio_preprocessor import AudioPreprocessor
from .whisper_transcriber import create_whisper_transcriber
from pathlib import Path
from typing import Optional, Union, List
import jiwer
import logging
import time
"""Transcription API for SAMO Voice Processing.

This module provides integration between the WhisperTranscriber and the application API
layer, handling transcription requests with proper error handling and performance
monitoring.
"""




logger = logging.getLogger(__name__)


class TranscriptionAPI:
    """API layer for voice transcription services.

    This class provides a simplified interface for the application to interact with the
    Whisper transcription functionality, including error handling, performance
    monitoring, and proper resource management.
    """

    def __init__(
        self, model_size: str = "base", language: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        """Initialize TranscriptionAPI with whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Default language for transcription (None for auto-detect)
            device: Compute device (cuda, cpu, None for auto-detect)
        """
        logger.info(f"Initializing TranscriptionAPI with model_size={model_size}")

        self.total_requests = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self.error_count = 0

        try:
            start_time = time.time()
            self.transcriber = create_whisper_transcriber(
                model_size=model_size, language=language, device=device
            )
            startup_time = time.time() - start_time
            logger.info(f"✅ TranscriptionAPI initialized in {startup_time:.2f}s")
            self.ready = True

        except Exception as exc:
            logger.error(f"❌ Failed to initialize TranscriptionAPI: {exc}")
            self.transcriber = None
            self.ready = False

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> dict:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            initial_prompt: Context prompt for better accuracy

        Returns:
            Dictionary with transcription results and metadata

        Raises:
            ValueError: If audio validation fails
            RuntimeError: If transcription fails
        """
        if not self.ready or self.transcriber is None:
            raise RuntimeError("TranscriptionAPI not initialized properly")

        start_time = time.time()
        self.total_requests += 1

        try:
            is_valid, error_msg = AudioPreprocessor.validate_audio_file(audio_path)
            if not is_valid:
                self.error_count += 1
                raise ValueError(f"Audio validation failed: {error_msg}")

            result = self.transcriber.transcribe(
                audio_path=audio_path, language=language, initial_prompt=initial_prompt
            )

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_audio_duration += result.duration

            return {
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "duration": result.duration,
                "processing_time": processing_time,
                "word_count": result.word_count,
                "speaking_rate": result.speaking_rate,
                "audio_quality": result.audio_quality,
                "metrics": {
                    "processing_time": processing_time,
                    "real_time_factor": processing_time / result.duration
                    if result.duration > 0
                    else 0,
                },
            }

        except Exception as exc:
            self.error_count += 1
            logger.error(f"Transcription failed: {exc}")
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[dict]:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            language: Language code (auto-detect if None)
            initial_prompt: Context prompt for better accuracy

        Returns:
            List of dictionaries with transcription results
        """
        if not self.ready or self.transcriber is None:
            raise RuntimeError("TranscriptionAPI not initialized properly")

        start_time = time.time()
        self.total_requests += len(audio_paths)

        results = []

        try:
            transcription_results = self.transcriber.transcribe_batch(
                audio_paths=audio_paths, language=language, initial_prompt=initial_prompt
            )

            for result in transcription_results:
                self.total_audio_duration += result.duration

                results.append(
                    {
                        "text": result.text,
                        "language": result.language,
                        "confidence": result.confidence,
                        "duration": result.duration,
                        "word_count": result.word_count,
                        "speaking_rate": result.speaking_rate,
                        "audio_quality": result.audio_quality,
                    }
                )

            batch_processing_time = time.time() - start_time
            self.total_processing_time += batch_processing_time

            return results

        except Exception as exc:
            self.error_count += len(audio_paths)
            logger.error(f"Batch transcription failed: {exc}")
            raise RuntimeError(f"Batch transcription failed: {exc}") from exc

    def evaluate_wer(self, audio_path: Union[str, Path], reference_text: str) -> dict:
        """Calculate Word Error Rate for transcription.

        Args:
            audio_path: Path to audio file
            reference_text: Reference transcription text

        Returns:
            Dictionary with WER evaluation metrics
        """
        try:
            result = self.transcribe(audio_path)
            transcription = result["text"]

            wer = jiwer.wer(reference_text, transcription)

            word_accuracy = 1 - wer
            character_error_rate = jiwer.cer(reference_text, transcription)

            return {
                "wer": wer,
                "word_accuracy": word_accuracy,
                "character_error_rate": character_error_rate,
                "transcription": transcription,
                "reference": reference_text,
                "confidence": result["confidence"],
                "duration": result["duration"],
                "audio_quality": result["audio_quality"],
            }

        except Exception as exc:
            logger.error(f"WER evaluation failed: {exc}")
            raise RuntimeError(f"WER evaluation failed: {exc}") from exc

    def get_performance_metrics(self) -> dict:
        """Get transcription performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "total_requests": self.total_requests,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "average_real_time_factor": self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else 0,
            "model_info": self.transcriber.get_model_info() if self.transcriber else {},
        }

        return metrics

    def get_model_info(self) -> dict:
        """Get information about the transcription model.

        Returns:
            Dictionary with model information
        """
        if not self.ready or self.transcriber is None:
            return {"status": "not_initialized"}

        return self.transcriber.get_model_info()


def create_transcription_api(
    model_size: str = "base", language: Optional[str] = None, device: Optional[str] = None
) -> TranscriptionAPI:
    """Create TranscriptionAPI with specified configuration.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Default language (None for auto-detect)
        device: Compute device (cuda, cpu, None for auto-detect)

    Returns:
        Configured TranscriptionAPI instance
    """
    api = TranscriptionAPI(model_size=model_size, language=language, device=device)
    return api
