#!/usr/bin/env python3
"""
Standalone test for SAMO Whisper Transcription Model

This script tests the Whisper transcription model independently
to ensure it works correctly before integration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.voice_processing.samo_whisper_transcriber import create_samo_whisper_transcriber

def test_audio_files():
    """Test available audio files."""
    test_audio_files = [
        "american_sample.wav",
        "french_sample.wav", 
        "interview_audio.wav",
        "test_audio.wav"
    ]
    
    available_audio = []
    for audio_file in test_audio_files:
        if Path(audio_file).exists():
            available_audio.append(audio_file)
            print(f"   ✅ Found: {audio_file}")
        else:
            print(f"   ⚠️  Not found: {audio_file}")
    
    return available_audio


def test_single_transcription(transcriber, audio_file, file_num, total_files, expected_language=None):
    """Test transcription of a single audio file."""
    print(f"\n   Testing file {file_num}: {audio_file}")
    try:
        result = transcriber.transcribe(audio_file)
        
        print("   ✅ Transcription successful!")
        text_preview = result.text[:100] + ('...' if len(result.text) > 100 else '')
        print(f"   Text: {text_preview}")
        print(f"   Language: {result.language}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Word count: {result.word_count}")
        print(f"   Speaking rate: {result.speaking_rate:.1f} WPM")
        print(f"   Audio quality: {result.audio_quality}")
        print(f"   No speech probability: {result.no_speech_probability:.3f}")

        if expected_language is not None:
            assert result.language == expected_language, (
                f"Detected language '{result.language}' does not match expected '{expected_language}'"
            )
            print(f"   ✅ Language detection correct: {result.language}")
        
    except Exception as e:
        print(f"   ❌ Transcription failed: {e}")


def test_batch_transcription(transcriber, available_audio):
    """Test batch transcription."""
    print(f"\n5. Testing batch transcription with {len(available_audio)} files...")
    try:
        results = transcriber.transcribe_batch(available_audio)
        successful = sum(1 for r in results if r.text.strip())
        print(f"   ✅ Batch transcription complete: {successful}/{len(results)} successful")
        
        total_duration = sum(r.duration for r in results)
        total_processing = sum(r.processing_time for r in results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        print(f"   Total audio: {total_duration:.1f}s")
        print(f"   Total processing: {total_processing:.1f}s")
        print(f"   Average confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        print(f"   ❌ Batch transcription failed: {e}")


def test_silence_detection(transcriber):
    """Test silence detection with silent audio."""
    print("\n6. Testing silence detection...")
    
    
    # Generate 2 seconds of silence at 16kHz
    silent_wav_path = "silent_test.wav"
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    
    try:
        # Create silent audio file
        sf.write(silent_wav_path, silence, sr)
        print(f"   Created silent audio file: {silent_wav_path}")
        
        # Test transcription
        result = transcriber.transcribe(silent_wav_path)
        print(f"   Text: {result.text!r}")
        print(f"   No speech probability: {result.no_speech_probability:.3f}")
        print(f"   Audio quality: {result.audio_quality}")
        
        
        # Assert high no speech probability for silence
        assert result.no_speech_probability > 0.5, f"No speech probability should be high for silence, got {result.no_speech_probability:.3f}"
        print("   ✅ Silence detection test passed")
        
    except Exception as e:
        print(f"   ❌ Silence detection test failed: {e}")
        raise
    finally:
        # Clean up the test file
        if os.path.exists(silent_wav_path):
            os.remove(silent_wav_path)
            print(f"   Cleaned up: {silent_wav_path}")


def test_samo_whisper_transcriber():
    """Test the SAMO Whisper transcriber functionality."""
    print("🎤 Testing SAMO Whisper Transcription Model")
    print("=" * 50)

    try:
        
        # Initialize transcriber
        print("1. Initializing SAMO Whisper Transcriber...")
        transcriber = create_samo_whisper_transcriber("configs/samo_whisper_config.yaml")
        print("✅ Transcriber initialized successfully")

        # Test model info
        print("\n2. Checking model information...")
        model_info = transcriber.get_model_info()
        print(f"   Model: {model_info['model_size']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Language: {model_info['language']}")
        print(f"   Task: {model_info['task']}")
        print(f"   Supported formats: {', '.join(model_info['supported_formats'])}")
        print(f"   Max duration: {model_info['max_duration']}s")
        print(f"   Target sample rate: {model_info['target_sample_rate']}Hz")

        # Test audio preprocessing
        print("\n3. Testing audio preprocessing...")
        available_audio = test_audio_files()
        
        if not available_audio:
            print("   ⚠️  No test audio files found. Creating a simple test...")
            # Test with a simple audio validation
            test_path = "test_audio.wav"
            if Path(test_path).exists():
                is_valid, msg = transcriber.preprocessor.validate_audio_file(test_path)
                print(f"   Audio validation: {msg}")
            else:
                print("   ⚠️  No audio files available for testing")
        else:
            # Test transcription with available audio
            print(f"\n4. Testing transcription with {len(available_audio)} audio file(s)...")
            
            for i, audio_file in enumerate(available_audio[:2], 1):  # Test max 2 files
                test_single_transcription(transcriber, audio_file, i, len(available_audio))

        # Test batch transcription if multiple files
        if len(available_audio) > 1:
            test_batch_transcription(transcriber, available_audio)

        # Test silence detection
        test_silence_detection(transcriber)

        print("\n" + "=" * 50)
        print("🎉 SAMO Whisper Transcriber test completed successfully!")
        print("✅ Model loaded and ready for production use")
        
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_samo_whisper_transcriber()
    sys.exit(0 if success else 1)
