"""🧪 COMPREHENSIVE API TEST SCRIPT.
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
    raise ValueError("API_KEY not set")

def test_endpoint(name, method, url, timeout=30, **kwargs):
    """Test an API endpoint and return results."""
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
        # Ensure a default timeout unless caller overrides
        kwargs.setdefault("timeout", timeout)
        handler = method_handlers.get(method.upper())
        if not handler:
            return False, f"Unsupported method: {method}"

        response = handler(url, headers=headers, **kwargs)
        time.time() - start_time


        # Use early return pattern to avoid nested conditionals
        if response.status_code != 200:
            return False, response.text

        # Success case
        try:
            data = response.json()
            return True, data
        except ValueError:
            return True, response.text

    except requests.exceptions.RequestException as e:
        time.time() - start_time
        return False, str(e)

def main() -> bool:
    """Run comprehensive API tests."""
    results = {}

    # Test 1: Health Check
    success, data = test_endpoint(
        "Health Check",
        "GET",
        f"{API_BASE_URL}/api/health"
    )
    results['health'] = success

    # Test 2: Emotion Detection (existing functionality)
    test_text = "Today I received a promotion at work and I'm really excited about it. This is such a great achievement!"
    success, data = test_endpoint(
        "Emotion Detection",
        "POST",
        f"{API_BASE_URL}/api/predict",
        json={"text": test_text}
    )
    results['emotion'] = success

    if success and isinstance(data, dict):
        data.get('primary_emotion', 'unknown')
        data.get('confidence', 0.0)

    # Test 2b: Emotion Detection - Missing Input
    invalid_success, invalid_data = test_endpoint(
        "Emotion Detection (Missing Input)",
        "POST",
        f"{API_BASE_URL}/api/predict",
        json={}  # Missing 'text' field
    )
    results['emotion_missing_input'] = invalid_success

    # Test 2c: Emotion Detection - Invalid Data Type
    invalid_type_success, _invalid_type_data = test_endpoint(
        "Emotion Detection (Invalid Data Type)",
        "POST",
        f"{API_BASE_URL}/api/predict",
        json={"text": 12345}  # 'text' should be a string
    )
    results['emotion_invalid_type'] = invalid_type_success

    # Test 2d: Emotion Detection - Negative Sentiment
    negative_text = "I'm feeling really sad and disappointed about everything that happened today."
    success, data = test_endpoint(
        "Emotion Detection (Negative)",
        "POST",
        f"{API_BASE_URL}/api/predict",
        json={"text": negative_text}
    )
    results['emotion_negative'] = success
    if success and isinstance(data, dict):
        data.get('primary_emotion', 'unknown')

    # Test 2e: Emotion Detection - Neutral Sentiment
    neutral_text = "The weather is cloudy today and the temperature is moderate."
    success, data = test_endpoint(
        "Emotion Detection (Neutral)",
        "POST",
        f"{API_BASE_URL}/api/predict",
        json={"text": neutral_text}
    )
    results['emotion_neutral'] = success
    if success and isinstance(data, dict):
        data.get('primary_emotion', 'unknown')

    # Test 3: T5 Summarization (NEW)
    success, data = test_endpoint(
        "T5 Summarization",
        "POST",
        f"{API_BASE_URL}/api/summarize",
        json={
            "text": test_text,
            "max_length": 100,
            "min_length": 20
        }
    )
    results['summarization'] = success

    if success and isinstance(data, dict):
        data.get('summary', '')
        data.get('compression_ratio', 0.0)

    # Test 3b: T5 Summarization - Missing Input
    invalid_success, _invalid_data = test_endpoint(
        "T5 Summarization (Missing Input)",
        "POST",
        f"{API_BASE_URL}/api/summarize",
        json={}  # Missing 'text' field
    )
    results['summarization_missing_input'] = invalid_success

    # Test 3c: T5 Summarization - Text Too Long
    long_text = "This is a very long text. " * 200  # Create text longer than 5000 chars
    invalid_success, invalid_data = test_endpoint(
        "T5 Summarization (Text Too Long)",
        "POST",
        f"{API_BASE_URL}/api/summarize",
        json={"text": long_text, "max_length": 100, "min_length": 20}
    )
    results['summarization_too_long'] = invalid_success

    # Test 4: Complete Analysis Pipeline (NEW) - Text Input
    success, data = test_endpoint(
        "Complete Analysis (Text)",
        "POST",
        f"{API_BASE_URL}/api/analyze/complete",
        data={
            "text": test_text,
            "generate_summary": "true",
            "emotion_threshold": "0.1"
        }
    )
    results['complete_analysis_text'] = success

    if success and isinstance(data, dict):
        data.get('pipeline_status', {})

        if data.get('emotion_analysis'):
            data['emotion_analysis'].get('primary_emotion', 'unknown')

        if data.get('summary'):
            data['summary'].get('summary', '')[:50] if data.get('summary') else ''

    # Test 4b: Complete Analysis Pipeline - Audio Input (if available)
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):

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
                f"{API_BASE_URL}/api/analyze/complete",
                files=files,
                data=data
            )
            results['complete_analysis_audio'] = success

            if success and isinstance(data, dict):
                data.get('pipeline_status', {})

                if data.get('transcription'):
                    data['transcription'].get('text', '')[:100] if data.get('transcription') else ''

                if data.get('emotion_analysis'):
                    data['emotion_analysis'].get('primary_emotion', 'unknown')

                if data.get('summary'):
                    data['summary'].get('summary', '')[:50] if data.get('summary') else ''
    else:
        results['complete_analysis_audio'] = None

    # Test 5: Voice Transcription (NEW) - requires audio file
    # Skip if no test audio file available
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):

        with open(test_audio_path, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            data = {'language': 'en'}

            success, data = test_endpoint(
                "Voice Transcription",
                "POST",
                f"{API_BASE_URL}/api/transcribe",
                files=files,
                data=data
            )
            results['transcription'] = success

            if success and isinstance(data, dict):
                data.get('text', '')
                data.get('confidence', 0.0)
    else:
        results['transcription'] = None

    # Summary

    total_tests = len([r for r in results.values() if r is not None])
    passed_tests = len([r for r in results.values() if r is True])


    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    if not success:
        raise ValueError("Test failed")
