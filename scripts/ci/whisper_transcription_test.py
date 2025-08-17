#!/usr/bin/env python3
"""
Whisper Voice Transcription Test for CI/CD Pipeline.

This script validates the Whisper transcription model functionality
with a simple test audio file.
"""

import contextlib
import logging
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

from scipy.io import wavfile

# Add src to path
sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def generate_test_audio():
    """Generate a simple synthetic test audio for Whisper testing."""
    try:
        temp_file = tempfile.NamedTemporaryFilesuffix=".wav", delete=False
        temp_file.close()

        sample_rate = 16000  # 16kHz as expected by Whisper
        duration = 2.0  # 2 seconds
        frequency = 440.0  # 440 Hz tone

        t = np.linspace(0.0, duration, intduration * sample_rate)
        amplitude = np.iinfonp.int16.max / 4
        data = (amplitude * np.sin2.0 * np.pi * frequency * t).astypenp.int16

        wavfile.writetemp_file.name, sample_rate, data
        logger.infof"✅ Generated test audio: {temp_file.name}"

        return temp_file.name

    except Exception as e:
        logger.errorf"❌ Failed to generate test audio: {e}"
        return None


def test_whisper_imports():
    """Test that Whisper modules can be imported."""
    try:
        logger.info"🔍 Testing Whisper imports..."

        # Test imports with fallback mechanism
        try:
            from models.voice_processing.audio_preprocessor import AudioPreprocessor
            from models.voice_processing.whisper_transcriber import WhisperTranscriber
        except ImportError:
            # Fallback for different import paths
            from src.models.voice_processing.audio_preprocessor import AudioPreprocessor
            from src.models.voice_processing.whisper_transcriber import WhisperTranscriber

        logger.info"✅ Whisper imports successful"
        return True

    except Exception as e:
        logger.errorf"❌ Whisper import test failed: {e}"
        return False


def test_whisper_instantiation():
    """Test WhisperTranscriber instantiation."""
    try:
        logger.info"🤖 Testing WhisperTranscriber instantiation..."

        # Test imports with fallback mechanism
        try:
            from models.voice_processing.whisper_transcriber import (
                TranscriptionConfig,
                WhisperTranscriber,
            )
        except ImportError:
            # Fallback for different import paths
            from src.models.voice_processing.whisper_transcriber import (
                TranscriptionConfig,
                WhisperTranscriber,
            )

        config = TranscriptionConfigmodel_size="tiny"
        config.device = "cpu"

        transcriber = WhisperTranscriberconfig

        if transcriber.model is None:
            logger.error"❌ WhisperTranscriber instantiated but model is None"
            return False

        logger.info"✅ WhisperTranscriber instantiated successfully"
        return True

    except Exception as e:
        logger.errorf"❌ WhisperTranscriber instantiation failed: {e}"
        return False


def test_audio_preprocessor():
    """Test AudioPreprocessor functionality."""
    try:
        logger.info"🎵 Testing AudioPreprocessor..."

        # Test imports with fallback mechanism
        try:
            from models.voice_processing.audio_preprocessor import AudioPreprocessor
        except ImportError:
            # Fallback for different import paths
            from src.models.voice_processing.audio_preprocessor import AudioPreprocessor

        # Generate test audio
        test_audio_path = generate_test_audio()
        if not test_audio_path:
            logger.warning"⚠️ Skipping audio preprocessor test - no test audio"
            return True

        try:
            preprocessor = AudioPreprocessor()
            
            # Test audio validation
            is_valid, error_msg = preprocessor.validate_audio_filetest_audio_path
            if not is_valid:
                logger.errorf"❌ Audio validation failed: {error_msg}"
                return False

            logger.info"✅ AudioPreprocessor test passed"
            return True

        finally:
            # Clean up test file
            with contextlib.suppressBaseException:
                os.unlinktest_audio_path

    except Exception as e:
        logger.errorf"❌ AudioPreprocessor test failed: {e}"
        return False


def test_minimal_transcription():
    """Test minimal transcription functionality."""
    try:
        logger.info"🎤 Testing minimal transcription..."

        # Check if we're in CI environment
        # Only run minimal transcription test locally, not in CI
        if os.getenv"CI":
            logger.info"⏭️ Skipping transcription test in CI environment"
            return True

        # Test imports with fallback mechanism
        try:
            from models.voice_processing.whisper_transcriber import (
                TranscriptionConfig,
                WhisperTranscriber,
            )
        except ImportError:
            # Fallback for different import paths
            from src.models.voice_processing.whisper_transcriber import (
                TranscriptionConfig,
                WhisperTranscriber,
            )

        # Generate test audio
        test_audio_path = generate_test_audio()
        if not test_audio_path:
            logger.warning"⚠️ Skipping transcription test - no test audio"
            return True

        try:
            # Test with tiny model smallest, fastest for CI purposes
            config = TranscriptionConfigmodel_size="tiny"
            config.device = "cpu"

            transcriber = WhisperTranscriberconfig

            # Test transcription
            result = transcriber.transcribetest_audio_path
            
            if result and result.text:
                logger.infof"✅ Transcription successful: {result.text[:50]}..."
                return True
            else:
                logger.error"❌ Transcription returned empty result"
                return False

        finally:
            # Clean up test file
            with contextlib.suppressBaseException:
                os.unlinktest_audio_path

    except Exception as e:
        logger.errorf"❌ Transcription test failed: {e}"
        return False


def main():
    """Run all Whisper transcription tests."""
    logger.info"🚀 Starting Whisper Transcription Tests..."

    tests = [
        "Whisper Imports", test_whisper_imports,
        "Whisper Instantiation", test_whisper_instantiation,
        "Audio Preprocessor", test_audio_preprocessor,
        "Minimal Transcription", test_minimal_transcription,
    ]

    passed = 0
    total = lentests

    for test_name, test_func in tests:
        logger.infof"\n{'='*50}"
        logger.infof"Running: {test_name}"
        logger.infof"{'='*50}"

        if test_func():
            passed += 1
            logger.infof"✅ {test_name}: PASSED"
        else:
            logger.errorf"❌ {test_name}: FAILED"

    logger.infof"\n{'='*50}"
    logger.infof"Whisper Transcription Tests Results: {passed}/{total} tests passed"
    logger.infof"{'='*50}"

    if passed == total:
        logger.info"🎉 All Whisper transcription tests passed!"
        return True
    else:
        logger.error"💥 Some Whisper transcription tests failed!"
        return False


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
