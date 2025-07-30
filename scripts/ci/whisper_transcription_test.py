import numpy as np
import os
import sys

#!/usr/bin/env python3
import logging
import tempfile
from pathlib import Path

import contextlib

# Add src to path
        from scipy.io import wavfile

        # Create a temporary WAV file with a simple tone
        from models.voice_processing.whisper_transcriber import (
        from models.voice_processing.audio_preprocessor import AudioPreprocessor

        # Generate test audio
        from models.voice_processing.whisper_transcriber import (

"""
Whisper Voice Transcription Test for CI/CD Pipeline.

This script validates the Whisper transcription model functionality
with a simple test audio file.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio():
    """Generate a simple synthetic test audio for Whisper testing."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        # Sample parameters
        sample_rate = 16000  # 16kHz as expected by Whisper
        duration = 2.0  # 2 seconds
        frequency = 440.0  # 440 Hz tone

        # Generate test tone
        t = np.linspace(0.0, duration, int(duration * sample_rate))
        amplitude = np.iinfo(np.int16).max / 4
        data = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.int16)

        # Write to file
        wavfile.write(temp_file.name, sample_rate, data)
        logger.info("✅ Generated test audio: {temp_file.name}")

        return temp_file.name

    except Exception as _:
        logger.error("❌ Failed to generate test audio: {e}")
        return None


def test_whisper_imports():
    """Test that Whisper modules can be imported."""
    try:
        logger.info("🔍 Testing Whisper imports...")

        # Import required modules

        logger.info("✅ Whisper imports successful")
        return True

    except Exception as _:
        logger.error("❌ Whisper import test failed: {e}")
        return False


def test_whisper_instantiation():
    """Test WhisperTranscriber instantiation."""
    try:
        logger.info("🤖 Testing WhisperTranscriber instantiation...")

        # Import WhisperTranscriber
            TranscriptionConfig,
            WhisperTranscriber,
        )

        # Test with tiny model (smallest, fastest) for CI purposes
        config = TranscriptionConfig(model_size="tiny")

        # For CI, use CPU even if GPU is available
        config.device = "cpu"

        # Create transcriber
        transcriber = WhisperTranscriber(config)

        # Verify model is loaded
        if transcriber.model is None:
            logger.error("❌ WhisperTranscriber instantiated but model is None")
            return False

        logger.info("✅ WhisperTranscriber instantiated successfully")
        return True

    except Exception as _:
        logger.error("❌ WhisperTranscriber instantiation failed: {e}")
        return False


def test_audio_preprocessor():
    """Test AudioPreprocessor functionality."""
    try:
        logger.info("🎵 Testing AudioPreprocessor...")

        # Import AudioPreprocessor
        audio_path = generate_test_audio()
        if not audio_path:
            logger.error("❌ Cannot test AudioPreprocessor without test audio")
            return False

        # Validate audio file
        is_valid, message = AudioPreprocessor.validate_audio_file(audio_path)
        if not is_valid:
            logger.error("❌ AudioPreprocessor validation failed: {message}")
            return False

        logger.info("✅ AudioPreprocessor validation successful")

        # Test preprocessing
        processed_path, metadata = AudioPreprocessor.preprocess_audio(audio_path)

        # Verify metadata
        expected_keys = ["duration", "sample_rate", "channels", "processed_sample_rate"]
        for key in expected_keys:
            if key not in metadata:
                logger.error("❌ Missing expected metadata key: {key}")
                return False

        logger.info("✅ AudioPreprocessor preprocessing successful")

        # Clean up
        try:
            os.unlink(audio_path)
            os.unlink(processed_path)
        except Exception:
            pass

        return True

    except Exception as _:
        logger.error("❌ AudioPreprocessor test failed: {e}")
        return False


def test_minimal_transcription():
    """Test a minimal transcription to ensure the pipeline works."""
    try:
        logger.info("📝 Testing minimal transcription...")

        # Import necessary components
            TranscriptionConfig,
            WhisperTranscriber,
        )

        # Generate test audio
        audio_path = generate_test_audio()
        if not audio_path:
            logger.error("❌ Cannot test transcription without test audio")
            return False

        # Create transcriber with tiny model
        config = TranscriptionConfig(model_size="tiny", device="cpu")
        transcriber = WhisperTranscriber(config)

        # Attempt transcription
        # Note: This will likely result in empty or nonsensical transcription
        # since our test audio is just a sine wave, but we're testing the pipeline
        result = transcriber.transcribe(audio_path)

        logger.info("Transcription result: {result.text}")
        logger.info("Confidence: {result.confidence:.2f}")
        logger.info("Audio quality: {result.audio_quality}")

        logger.info("✅ Minimal transcription test completed")

        # Clean up
        with contextlib.suppress(Exception):
            os.unlink(audio_path)

        return True

    except Exception as _:
        logger.error("❌ Minimal transcription test failed: {e}")
        return False


def main():
    """Run all Whisper transcription tests."""
    logger.info("🚀 Starting Whisper Transcription Tests...")

    tests = [
        ("Whisper Imports", test_whisper_imports),
        ("Whisper Instantiation", test_whisper_instantiation),
        ("Audio Preprocessor", test_audio_preprocessor),
        # Skip this test if in CI to avoid long model downloads
        # ("Minimal Transcription", test_minimal_transcription),
    ]

    # Check if we're in CI environment
    in_ci = os.environ.get("CI") == "true"

    # Only run minimal transcription test locally, not in CI
    if not in_ci:
        tests.append(("Minimal Transcription", test_minimal_transcription))

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("\n{'='*50}")
        logger.info("Running: {test_name}")
        logger.info("{'='*50}")

        if test_func():
            passed += 1
            logger.info("✅ {test_name}: PASSED")
        else:
            logger.error("❌ {test_name}: FAILED")

    logger.info("\n{'='*50}")
    logger.info("Whisper Transcription Test Results: {passed}/{total} tests passed")
    logger.info("{'='*50}")

    if passed == total:
        logger.info("🎉 All Whisper transcription tests passed!")
        return True
    else:
        logger.error("💥 Some Whisper transcription tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
