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
import time
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://emotion-detection-api-frrnetyhfa-uc.a.run.app")
API_KEY = os.getenv("API_KEY", "your-api-key-here")

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
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, **kwargs)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, **kwargs)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False

        elapsed = time.time() - start_time

        print(f"   Status: {response.status_code}")
        print(".2f")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ Success - {name}")
                return True, data
            except:
                print(f"   ⚠️  Success but invalid JSON - {name}")
                return True, response.text
        else:
            print(f"   ❌ Failed - {name}")
            print(f"   Response: {response.text[:200]}...")
            return False, response.text

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error - {name}: {e}")
        print(".2f")
        return False, str(e)

def main():
    """Run comprehensive API tests"""
    print("🚀 SAMO Complete AI API Test Suite")
    print("=" * 50)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {'****' + API_KEY[-4:] if API_KEY != 'your-api-key-here' else 'NOT SET'}")
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
        print(".2f")

    # Test 4: Complete Analysis Pipeline (NEW)
    success, data = test_endpoint(
        "Complete Analysis",
        "POST",
        f"{API_BASE_URL}/analyze/complete",
        data={
            "text": test_text,
            "generate_summary": "true",
            "emotion_threshold": "0.1"
        }
    )
    results['complete_analysis'] = success

    if success and isinstance(data, dict):
        pipeline_status = data.get('pipeline_status', {})
        print(f"   Pipeline status: {pipeline_status}")

        if data.get('emotion_analysis'):
            emotion = data['emotion_analysis'].get('primary_emotion', 'unknown')
            print(f"   Emotion: {emotion}")

        if data.get('summary'):
            summary = data['summary'].get('summary', '')[:50]
            print(f"   Summary: {summary}...")

    # Test 5: Voice Transcription (NEW) - requires audio file
    # Skip if no test audio file available
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):
        print("
🎵 Testing Voice Transcription..."        print(f"   Audio file found: {test_audio_path}")

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
                print(".2f")
    else:
        print("
🎵 Voice Transcription test SKIPPED (no test audio file)"        print(f"   To test transcription, create a {test_audio_path} file")
        results['transcription'] = None

    # Summary
    print("
📊 TEST RESULTS SUMMARY"    print("=" * 30)

    total_tests = len([r for r in results.values() if r is not None])
    passed_tests = len([r for r in results.values() if r is True])

    for test_name, result in results.items():
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠️  SKIP")
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    print("
🏆 Overall Score: {passed_tests}/{total_tests} tests passed"

    if passed_tests == total_tests:
        print("   🎉 All tests passed! Your Complete AI API is working perfectly!")
        return True
    else:
        print("   ⚠️  Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
