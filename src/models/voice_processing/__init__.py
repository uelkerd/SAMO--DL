# Only import what's actually needed for the API
try:
    from .whisper_transcriber import create_whisper_transcriber
except ImportError:
    create_whisper_transcriber = None

# Optional imports for training/development
try:
    from .audio_preprocessor import AudioPreprocessor, preprocess_audio
except ImportError:
    AudioPreprocessor = None
    preprocess_audio = None

try:
    from .transcription_api import TranscriptionAPI
except ImportError:
    TranscriptionAPI = None


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

__version__ = "0.1.0"
__author__ = "SAMO Deep Learning Team"

__all__ = [
    "create_whisper_transcriber",
]
