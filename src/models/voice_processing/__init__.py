"""SAMO Deep Learning - Voice Processing Module.

This module implements OpenAI Whisper-based voice-to-text processing for
SAMO's voice-first journaling experience with high accuracy transcription.

Key Components:
- WhisperTranscriber: Core OpenAI Whisper implementation
- AudioPreprocessor: Audio format handling and preprocessing
- TranscriptionAPI: FastAPI endpoints for Web Dev integration
- VoiceQualityAnalyzer: Audio quality assessment and confidence scoring

Performance Targets:
- Voice Transcription Accuracy: <10% WER for clear speech
- Response Latency: <500ms for P95 requests
- Audio Format Support: MP3, WAV, M4A, OGG
- Real-time Processing: Up to 5-minute audio clips
"""

from .audio_preprocessor import AudioPreprocessor, preprocess_audio
from .transcription_api import TranscriptionAPI
from .whisper_transcriber import WhisperTranscriber, create_whisper_transcriber

__version__ = "0.1.0"
__author__ = "SAMO Deep Learning Team"

__all__ = [
    "AudioPreprocessor",
    "TranscriptionAPI",
    "WhisperTranscriber",
    "create_whisper_transcriber",
    "preprocess_audio",
]
