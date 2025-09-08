#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE API TEST SCRIPT
================================
Tests all SAMO API endpoints including:
- Emotion Detection (existing)
- T5 Summarization (new)
- Whisper Transcription (new)
- Complete Analysis Pipeline (new)
"""

import requests
import sys
import time
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://emotion-detection-api-frrnetyhfa-uc.a.run.app")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    print("❌ API_KEY environment variable not set!")
    print("   Please set API_KEY environment variable before running tests")
    sys.exit(1)

def test_endpoint(name, method, url, **kwargs):
    """Test an API endpoint and return results"""
    print(f"\n🧪 Testing {name}...")
    print(f"   URL: {url}")
    print(f"   Method: {method}")

    headers = {"X-API-Key": API_KEY}
    if 'headers' in kwargs:
        headers.update(kwargs['headers'])
        del kwargs['headers']

    start_time = time.time()
    
    # Use method mapping to avoid conditionals
    method_handlers = {
        'GET': requests.get,
        'POST': requests.post
    }
    
    try:
        handler = method_handlers.get(method.upper())
        if not handler:
            print(f"   ❌ Unsupported method: {method}")
            return False, f"Unsupported method: {method}"

        response = handler(url, headers=headers, **kwargs)
        elapsed = time.time() - start_time

        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")

        # Use early return pattern to avoid nested conditionals
        if response.status_code != 200:
            print(f"   ❌ Failed - {name}")
            print(f"   Response: {response.text[:200]}...")
            return False, response.text

        # Success case
        try:
            data = response.json()
            print(f"   ✅ Success - {name}")
            return True, data
        except:
            print(f"   ⚠️  Success but invalid JSON - {name}")
            return True, response.text

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error - {name}: {e}")
        print(f"   Time: {elapsed:.2f}s")
        return False, str(e)

def main():
    """Run comprehensive API tests"""
    print("🚀 SAMO Complete AI API Test Suite")
    print("=" * 50)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {'****' + API_KEY[-4:] if API_KEY else 'NOT SET'}")
    print()

    results = {}

    # Test 1: Health Check
    success, data = test_endpoint(
        "Health Check",
        "GET",
        f"{API_BASE_URL}/health"
    )
    results['health'] = success

    if success and isinstance(data, dict):
        print(f"   Models available: {data.get('models', {})}")

    # Test 2: Emotion Detection (existing functionality)
    test_text = "Today I received a promotion at work and I'm really excited about it. This is such a great achievement!"
    success, data = test_endpoint(
        "Emotion Detection",
        "POST",
        f"{API_BASE_URL}/predict",
        json={"text": test_text, "threshold": 0.1}
    )
    results['emotion'] = success

    if success and isinstance(data, dict):
        primary_emotion = data.get('primary_emotion', 'unknown')
        confidence = data.get('confidence', 0.0)
        print(f"   Primary emotion: {primary_emotion} ({confidence:.2f})")

    # Test 2b: Emotion Detection - Missing Input
    invalid_success, invalid_data = test_endpoint(
        "Emotion Detection (Missing Input)",
        "POST",
        f"{API_BASE_URL}/predict",
        json={}  # Missing 'text' field
    )
    results['emotion_missing_input'] = invalid_success
    print(f"   Emotion Detection (Missing Input): {'PASS' if not invalid_success else 'FAIL'} - Expected error, got: {invalid_data}")

    # Test 2c: Emotion Detection - Invalid Data Type
    invalid_type_success, invalid_type_data = test_endpoint(
        "Emotion Detection (Invalid Data Type)",
        "POST",
        f"{API_BASE_URL}/predict",
        json={"text": 12345}  # 'text' should be a string
    )
    results['emotion_invalid_type'] = invalid_type_success
    print(f"   Emotion Detection (Invalid Data Type): {'PASS' if not invalid_type_success else 'FAIL'} - Expected error, got: {invalid_type_data}")

    # Test 2d: Emotion Detection - Negative Sentiment
    negative_text = "I'm feeling really sad and disappointed about everything that happened today."
    success, data = test_endpoint(
        "Emotion Detection (Negative)",
        "POST",
        f"{API_BASE_URL}/predict",
        json={"text": negative_text, "threshold": 0.1}
    )
    results['emotion_negative'] = success
    if success and isinstance(data, dict):
        primary_emotion = data.get('primary_emotion', 'unknown')
        print(f"   Negative emotion detected: {primary_emotion}")

    # Test 2e: Emotion Detection - Neutral Sentiment
    neutral_text = "The weather is cloudy today and the temperature is moderate."
    success, data = test_endpoint(
        "Emotion Detection (Neutral)",
        "POST",
        f"{API_BASE_URL}/predict",
        json={"text": neutral_text, "threshold": 0.1}
    )
    results['emotion_neutral'] = success
    if success and isinstance(data, dict):
        primary_emotion = data.get('primary_emotion', 'unknown')
        print(f"   Neutral emotion detected: {primary_emotion}")

    # Test 3: T5 Summarization (NEW)
    success, data = test_endpoint(
        "T5 Summarization",
        "POST",
        f"{API_BASE_URL}/summarize",
        json={
            "text": test_text,
            "max_length": 100,
            "min_length": 20
        }
    )
    results['summarization'] = success

    if success and isinstance(data, dict):
        summary = data.get('summary', '')
        compression = data.get('compression_ratio', 0.0)
        print(f"   Summary: {summary[:100]}...")
        print(f"   Compression: {compression:.2f}")

    # Test 3b: T5 Summarization - Missing Input
    invalid_success, invalid_data = test_endpoint(
        "T5 Summarization (Missing Input)",
        "POST",
        f"{API_BASE_URL}/summarize",
        json={}  # Missing 'text' field
    )
    results['summarization_missing_input'] = invalid_success
    print(f"   T5 Summarization (Missing Input): {'PASS' if not invalid_success else 'FAIL'} - Expected error, got: {invalid_data}")

    # Test 3c: T5 Summarization - Text Too Long
    long_text = "This is a very long text. " * 200  # Create text longer than 5000 chars
    invalid_success, invalid_data = test_endpoint(
        "T5 Summarization (Text Too Long)",
        "POST",
        f"{API_BASE_URL}/summarize",
        json={"text": long_text, "max_length": 100, "min_length": 20}
    )
    results['summarization_too_long'] = invalid_success
    print(f"   T5 Summarization (Text Too Long): {'PASS' if not invalid_success else 'FAIL'} - Expected error, got: {invalid_data}")

    # Test 4: Complete Analysis Pipeline (NEW) - Text Input
    success, data = test_endpoint(
        "Complete Analysis (Text)",
        "POST",
        f"{API_BASE_URL}/analyze/complete",
        data={
            "text": test_text,
            "generate_summary": "true",
            "emotion_threshold": "0.1"
        }
    )
    results['complete_analysis_text'] = success

    if success and isinstance(data, dict):
        pipeline_status = data.get('pipeline_status', {})
        print(f"   Pipeline status: {pipeline_status}")

        if data.get('emotion_analysis'):
            emotion = data['emotion_analysis'].get('primary_emotion', 'unknown')
            print(f"   Emotion: {emotion}")

        if data.get('summary'):
            summary = data['summary'].get('summary', '')[:50]
            print(f"   Summary: {summary}...")

    # Test 4b: Complete Analysis Pipeline - Audio Input (if available)
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):
        print("\n🎵 Testing Complete Analysis with Audio Input...")
        print(f"   Audio file found: {test_audio_path}")

        with open(test_audio_path, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            data = {
                'language': 'en',
                'generate_summary': 'true',
                'emotion_threshold': '0.1'
            }

            success, data = test_endpoint(
                "Complete Analysis (Audio)",
                "POST",
                f"{API_BASE_URL}/analyze/complete",
                files=files,
                data=data
            )
            results['complete_analysis_audio'] = success

            if success and isinstance(data, dict):
                pipeline_status = data.get('pipeline_status', {})
                print(f"   Pipeline status: {pipeline_status}")

                if data.get('transcription'):
                    transcription = data['transcription'].get('text', '')[:100]
                    print(f"   Transcription: {transcription}...")

                if data.get('emotion_analysis'):
                    emotion = data['emotion_analysis'].get('primary_emotion', 'unknown')
                    print(f"   Emotion: {emotion}")

                if data.get('summary'):
                    summary = data['summary'].get('summary', '')[:50]
                    print(f"   Summary: {summary}...")
    else:
        print("\n🎵 Complete Analysis with Audio test SKIPPED (no test audio file)")
        print(f"   To test complete analysis with audio, create a {test_audio_path} file")
        results['complete_analysis_audio'] = None

    # Test 5: Voice Transcription (NEW) - requires audio file
    # Skip if no test audio file available
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):
        print("\n🎵 Testing Voice Transcription...")
        print(f"   Audio file found: {test_audio_path}")

        with open(test_audio_path, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            data = {'language': 'en'}

            success, data = test_endpoint(
                "Voice Transcription",
                "POST",
                f"{API_BASE_URL}/transcribe",
                files=files,
                data=data
            )
            results['transcription'] = success

            if success and isinstance(data, dict):
                transcription = data.get('text', '')
                confidence = data.get('confidence', 0.0)
                print(f"   Transcription: {transcription[:100]}...")
                print(f"   Confidence: {confidence:.2f}")
    else:
        print("\n🎵 Voice Transcription test SKIPPED (no test audio file)")
        print(f"   To test transcription, create a {test_audio_path} file")
        results['transcription'] = None

    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)

    total_tests = len([r for r in results.values() if r is not None])
    passed_tests = len([r for r in results.values() if r is True])

    for test_name, result in results.items():
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠️  SKIP")
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n🏆 Overall Score: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("   🎉 All tests passed! Your Complete AI API is working perfectly!")
        return True
    print("   ⚠️  Some tests failed. Check the logs above for details.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
