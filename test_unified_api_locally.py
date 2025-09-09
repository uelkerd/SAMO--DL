#!/usr/bin/env python3
"""
Local Test Script for Unified AI API
Tests all three features: emotion detection, voice transcription, text summarization
"""

import requests
import io
import numpy as np
import wave

# Configuration
API_BASE_URL = "http://localhost:8000"

def generate_test_audio(duration=2.0, sample_rate=16000, freq=440):
    """Generate a simple test audio file (tone)"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer

def test_health_check():
    """Test the health endpoint"""
    print("🩺 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Models loaded: emotion={data['models']['emotion_detection']['loaded']}, "
                  f"summarizer={data['models']['text_summarization']['loaded']}, "
                  f"voice={data['models']['voice_processing']['loaded']}")
            return True
        print(f"❌ Health check failed: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection"""
    print("\n😊 Testing emotion detection...")
    test_texts = [
        "I am so happy and excited about this!",
        "I feel frustrated and overwhelmed with all this work",
        "I am feeling calm and content today"
    ]

    for text in test_texts:
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze/journal",
                json={"text": text, "generate_summary": False}
            )

            if response.status_code == 200:
                data = response.json()
                emotion = data['emotion_analysis']['primary_emotion']
                confidence = data['emotion_analysis']['confidence']
                print(f"✅ '{text[:30]}...' → {emotion} ({confidence:.3f})")
            else:
                print(f"❌ Emotion detection failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return False

    return True

def test_text_summarization():
    """Test text summarization"""
    print("\n📝 Testing text summarization...")
    test_text = """
    Today I had an amazing experience at the conference. I learned so much about artificial intelligence
    and machine learning. The speakers were incredibly knowledgeable and the networking opportunities
    were fantastic. I met several people who are working on similar projects to mine. Overall, it was
    a very productive and inspiring day that has motivated me to continue working on my AI research.
    """

    try:
        response = requests.post(
            f"{API_BASE_URL}/summarize/text",
            data={
                "text": test_text,
                "model": "t5-small",
                "max_length": 50,
                "min_length": 10
            }
        )

        if response.status_code == 200:
            data = response.json()
            summary = data['summary']
            print("✅ Text summarization successful!")
            print(f"   Original: {len(test_text)} chars")
            print(f"   Summary: {len(summary)} chars")
            print(f"   Content: {summary}")
            return True
        print(f"❌ Text summarization failed: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ Text summarization error: {e}")
        return False

def test_voice_transcription():
    """Test voice transcription"""
    print("\n🎤 Testing voice transcription...")
    try:
        # Generate test audio
        audio_buffer = generate_test_audio(duration=2.0)

        # Create multipart form data
        files = {
            'audio_file': ('test_audio.wav', audio_buffer, 'audio/wav')
        }

        response = requests.post(
            f"{API_BASE_URL}/transcribe/voice",
            files=files,
            data={'language': 'en'}
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get('text', '')
            confidence = data.get('confidence', 0)
            print("✅ Voice transcription successful!")
            print(f"   Transcribed text: '{text}'")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Language: {data.get('language', 'unknown')}")
            return True
        print(f"❌ Voice transcription failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    except Exception as e:
        print(f"❌ Voice transcription error: {e}")
        return False

def test_complete_pipeline():
    """Test the complete analysis pipeline"""
    print("\n🔄 Testing complete analysis pipeline...")

    # This would require an actual audio file for voice transcription
    # For now, we'll test text-only analysis
    test_text = "Today I received a promotion at work and I'm really excited about it!"

    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/journal",
            json={"text": test_text, "generate_summary": True}
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ Complete pipeline successful!")
            print(f"   📝 Text: {data['emotion_analysis']['text'][:50]}...")
            print(f"   😊 Emotion: {data['emotion_analysis']['primary_emotion']} "
                  f"({data['emotion_analysis']['confidence']:.3f})")
            print(f"   📋 Summary: {data['summary']['summary'][:50]}...")
            print(f"   ⏱️ Processing time: {data['processing_time_ms']:.1f}ms")
            return True
        print(f"❌ Complete pipeline failed: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ Complete pipeline error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING UNIFIED AI API")
    print("=" * 50)

    # Check if API is running
    print("🔍 Checking if API is running...")
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=5)
    except:
        print("❌ API is not running!")
        print("   Please start the API first:")
        print("   cd /workspace && python -m uvicorn src.unified_ai_api:app --host 0.0.0.0 --port 8000")
        return 1

    tests = [
        ("Health Check", test_health_check),
        ("Emotion Detection", test_emotion_detection),
        ("Text Summarization", test_text_summarization),
        ("Voice Transcription", test_voice_transcription),
        ("Complete Pipeline", test_complete_pipeline),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"🎉 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! Unified API is working perfectly!")
        print("\n🎯 All three features are operational:")
        print("   ✅ Emotion Detection")
        print("   ✅ Text Summarization")
        print("   ✅ Voice Transcription")
        return 0
    print(f"❌ {total - passed} tests failed. Check the implementation.")
    return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())