#!/usr/bin/env python3
"""
Whisper Voice Transcription Test for CI/CD Pipeline.

This script validates the Whisper transcription model functionality
with a simple test audio file.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio():
    """Generate a simple synthetic test audio for Whisper testing."""
    try:
        import soundfile as sf
        from scipy.io import wavfile

        # Create a temporary WAV file with a simple tone
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        # Sample parameters
        sample_rate = 16000  # 16kHz as expected by Whisper
        duration = 2.0  # 2 seconds
        frequency = 440.0  # 440 Hz tone

        # Generate test tone
        t = np.linspace(0., duration, int(duration * sample_rate))
        amplitude = np.iinfo(np.int16).max / 4
        data = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        
        # Write to file
        wavfile.write(temp_file.name, sample_rate, data)
        logger.info(f"✅ Generated test audio: {temp_file.name}")
        
        return temp_file.name
    
    except Exception as e:
        logger.error(f"❌ Failed to generate test audio: {e}")
        return None


def test_whisper_imports():
    """Test that Whisper modules can be imported."""
    try:
        logger.info("🔍 Testing Whisper imports...")

        # Import required modules
        from models.voice_processing.whisper_transcriber import (
            AudioPreprocessor,
            WhisperTranscriber,
            create_whisper_transcriber,
        )

        logger.info("✅ Whisper imports successful")
        return True

    except Exception as e:
        logger.error(f"❌ Whisper import test failed: {e}")
        return False


def test_whisper_instantiation():
    """Test WhisperTranscriber instantiation."""
    try:
        logger.info("🤖 Testing WhisperTranscriber instantiation...")

        # Import WhisperTranscriber
        from models.voice_processing.whisper_transcriber import (
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

    except Exception as e:
        logger.error(f"❌ WhisperTranscriber instantiation failed: {e}")
        return False


def test_audio_preprocessor():
    """Test AudioPreprocessor functionality."""
    try:
        logger.info("🎵 Testing AudioPreprocessor...")

        # Import AudioPreprocessor
        from models.voice_processing.audio_preprocessor import AudioPreprocessor
        
        # Generate test audio
        audio_path = generate_test_audio()
        if not audio_path:
            logger.error("❌ Cannot test AudioPreprocessor without test audio")
            return False
        
        # Validate audio file
        is_valid, message = AudioPreprocessor.validate_audio_file(audio_path)
        if not is_valid:
            logger.error(f"❌ AudioPreprocessor validation failed: {message}")
            return False
            
        logger.info("✅ AudioPreprocessor validation successful")
        
        # Test preprocessing
        processed_path, metadata = AudioPreprocessor.preprocess_audio(audio_path)
        
        # Verify metadata
        expected_keys = ["duration", "sample_rate", "channels", "processed_sample_rate"]
        for key in expected_keys:
            if key not in metadata:
                logger.error(f"❌ Missing expected metadata key: {key}")
                return False
        
        logger.info("✅ AudioPreprocessor preprocessing successful")
        
        # Clean up
        try:
            os.unlink(audio_path)
            os.unlink(processed_path)
        except Exception:
            pass
            
        return True

    except Exception as e:
        logger.error(f"❌ AudioPreprocessor test failed: {e}")
        return False


def test_minimal_transcription():
    """Test a minimal transcription to ensure the pipeline works."""
    try:
        logger.info("📝 Testing minimal transcription...")

        # Import necessary components
        from models.voice_processing.whisper_transcriber import (
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
        
        logger.info(f"Transcription result: {result.text}")
        logger.info(f"Confidence: {result.confidence:.2f}")
        logger.info(f"Audio quality: {result.audio_quality}")
        
        logger.info("✅ Minimal transcription test completed")
        
        # Clean up
        try:
            os.unlink(audio_path)
        except Exception:
            pass
        
        return True

    except Exception as e:
        logger.error(f"❌ Minimal transcription test failed: {e}")
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
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")

    logger.info(f"\n{'='*50}")
    logger.info(f"Whisper Transcription Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info("🎉 All Whisper transcription tests passed!")
        return True
    else:
        logger.error("💥 Some Whisper transcription tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 