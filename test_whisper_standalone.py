"""
Standalone test script for SAMO Whisper transcriber.

This script tests the Whisper transcription functionality independently
without requiring the full SAMO API infrastructure.
"""

import sys
from pathlib import Path
import textwrap

# Add src to path
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.transcription.whisper_transcriber import create_samo_whisper_transcriber


def test_whisper_transcriber():
    """Test the SAMO Whisper transcriber functionality."""
    print("🎤 Testing SAMO Whisper Transcriber")
    print("=" * 50)

    try:
        # Initialize transcriber
        config_path = REPO_ROOT / "configs" / "whisper_config.yaml"
        transcriber = create_samo_whisper_transcriber(str(config_path))
        print("✅ Whisper transcriber initialized successfully")

        # Test with sample audio files
        sample_files = [
            "american_sample.wav",
            "french_sample.wav",
            "interview_audio.wav"
        ]

        for audio_file in sample_files:
            audio_path = REPO_ROOT / audio_file
            if audio_path.exists():
                print(f"\n🎵 Testing with {audio_file}")
                print("-" * 30)

                try:
                    # Transcribe audio
                    result = transcriber.transcribe_audio(
                        str(audio_path),
                        language="auto",
                        return_timestamps=True
                    )

                    print(f"📝 Transcription: {result['text']}")
                    print(f"🌍 Language: {result['language']}")
                    print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
                    print(f"🎯 Confidence: {result['confidence']:.3f}")
                    print(f"⏰ Audio duration: {result['audio_duration']:.2f}s")

                    if result['timestamps']:
                        print(f"📍 Timestamps: {len(result['timestamps'])} segments")

                except Exception as e:
                    print(f"❌ Error transcribing {audio_file}: {e}")

            else:
                print(f"⚠️  Sample file not found: {audio_file}")

        # Test batch transcription
        print(f"\n🔄 Testing batch transcription")
        print("-" * 30)

        available_files = [
            str(REPO_ROOT / f) for f in sample_files
            if (REPO_ROOT / f).exists()
        ]

        if available_files:
            batch_results = transcriber.transcribe_batch(
                available_files,
                language="auto",
                return_timestamps=True
            )

            print(f"📊 Batch processed {len(batch_results)} files")
            for i, result in enumerate(batch_results):
                if "error" in result:
                    print(f"❌ File {i+1}: {result['error']}")
                else:
                    print(f"✅ File {i+1}: {result['text'][:50]}...")

        print(f"\n🎉 Whisper transcriber test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration")
    print("-" * 30)

    try:
        config_path = REPO_ROOT / "configs" / "whisper_config.yaml"
        transcriber = create_samo_whisper_transcriber(str(config_path))

        print(f"✅ Configuration loaded successfully")
        print(f"📱 Device: {transcriber.device}")
        print(f"🤖 Model: {transcriber.config['model']['name']}")
        print(f"🎵 Sample rate: {transcriber.config['audio']['sample_rate']}")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 Starting SAMO Whisper Transcriber Tests")
    print("=" * 60)

    # Test configuration
    config_success = test_configuration()

    # Test transcriber functionality
    transcriber_success = test_whisper_transcriber()

    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"Transcriber: {'✅ PASS' if transcriber_success else '❌ FAIL'}")

    if config_success and transcriber_success:
        print("\n🎉 All tests passed! Whisper transcriber is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
