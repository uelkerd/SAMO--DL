#!/usr/bin/env python3
"""
SAMO-Optimized Whisper Voice Transcription Model (Refactored)

This module provides a specialized Whisper transcription model optimized for
journal entries and voice processing in the SAMO-DL system.

Key Features:
- OpenAI Whisper model integration (tiny, base, small, medium, large)
- Multi-format audio support (MP3, WAV, M4A, OGG, FLAC)
- Confidence scoring and quality assessment
- Chunk-based processing for long audio files
- Production-ready error handling and logging
- Batch transcription for multiple audio files
- Modular architecture for better maintainability
"""

import logging
import os
import time
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch

from .whisper_config import SAMOWhisperConfig
from .whisper_audio_preprocessor import AudioPreprocessor
from .whisper_models import WhisperModelManager
from .whisper_results import TranscriptionResult, ResultProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.model_manager = WhisperModelManager(self.config, self.device)
        self.result_processor = ResultProcessor()
        
        # Load the model
        self.model_manager.load_model()
    
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
        processed_audio_path, audio_metadata = self.preprocessor.preprocess_audio(audio_path)
        
        try:
            # Prepare transcription options
            transcribe_options = self.config.get_transcription_options()
            
            # Override with provided parameters
            if language is not None:
                transcribe_options["language"] = language
            if initial_prompt is not None:
                transcribe_options["initial_prompt"] = initial_prompt
            
            # Transcribe
            model = self.model_manager.get_model()
            result = model.transcribe(processed_audio_path, **transcribe_options)
            
            processing_time = time.time() - start_time
            word_count = len(result['text'].split())
            speaking_rate = self.result_processor.calculate_speaking_rate(
                word_count, audio_metadata['duration']
            )
            
            # Calculate confidence from segments
            confidence = self.result_processor.calculate_confidence(result.get('segments', []))
            
            # Calculate no_speech_probability from segments
            no_speech_probability = self.result_processor.calculate_no_speech_probability(result.get('segments', []))
            
            # Assess audio quality
            audio_quality = self.preprocessor.assess_audio_quality(result, audio_metadata)
            
            transcription_result = TranscriptionResult(
                text=result['text'].strip() if isinstance(result.get('text'), str) else '',
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
                error_msg = f"Failed to transcribe {audio_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Create error result with detailed error information
                error_result = self.result_processor.create_error_result(str(e))
                results.append(error_result)
        
        total_duration = sum(r.duration for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        successful_transcriptions = sum(1 for r in results if not r.text.startswith("[ERROR:"))
        
        logger.info("✅ Batch transcription complete: %d files", len(results))
        logger.info(
            "Successful: %d/%d, Total audio: %.1fs, Processing: %.1fs",
            successful_transcriptions, len(results), total_duration, total_processing_time
        )
        
        if errors:
            logger.warning("Batch transcription had %d errors:", len(errors))
            for error in errors:
                logger.warning("  - %s", error)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_manager.get_model_info()


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
