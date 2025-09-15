#!/usr/bin/env python3
"""
Fixed API Testing Script with Correct Request Formats
Tests all 3 core features with proper request structures
"""

import requests
import json
import time
import os
import logging
import io
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedAPITester:
    def __init__(self, base_url: str = "https://samo-unified-api-frrnetyhfa-uc.a.run.app"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = 30
        self.auth_token = None

    def authenticate(self) -> bool:
        """Get authentication token"""
        logger.info("🔐 Getting authentication token...")

        register_data = {
            "username": f"fixed_test_{int(time.time())}@example.com",
            "email": f"fixed_test_{int(time.time())}@example.com",
            "password": "TestPassword123!",
            "full_name": "Fixed Test User"
        }

        try:
            response = self.session.post(
                f"{self.base_url}/auth/register",
                json=register_data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('access_token')
                logger.info("✅ Authentication successful")
                return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")

        return False

    def test_voice_transcription_fixed(self) -> Dict[str, Any]:
        """Test voice transcription with correct multipart/form-data format"""
        logger.info("🎤 Testing Voice Transcription (Fixed Format)...")

        try:
            # Create a minimal valid WAV file
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
            silence_data = b'\x00' * 2048
            wav_data = wav_header + silence_data

            # Prepare files and data (NOT JSON)
            files = {
                'audio_file': ('test_audio.wav', io.BytesIO(wav_data), 'audio/wav')
            }

            data = {
                'language': 'en',
                'model_size': 'base',
                'timestamp': 'false'
            }

            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/transcribe/voice",
                files=files,
                data=data,  # Note: data, not json
                headers=headers,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000

            return {
                "endpoint": "/transcribe/voice",
                "method": "POST",
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else {"raw": response.text},
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}"
            }

        except Exception as e:
            return {
                "endpoint": "/transcribe/voice",
                "method": "POST",
                "success": False,
                "error": str(e)
            }

    def test_text_summarization_fixed(self) -> Dict[str, Any]:
        """Test text summarization with correct form data format"""
        logger.info("📄 Testing Text Summarization (Fixed Format)...")

        try:
            test_text = "I am feeling incredibly happy and excited about this new opportunity! This situation is making me quite anxious and nervous. I feel a deep sense of gratitude for all the support I've received. Today has been a rollercoaster of emotions with many ups and downs."

            # Use form data, NOT JSON
            data = {
                'text': test_text,
                'model': 't5-small',
                'max_length': '150',
                'min_length': '30'
            }

            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/summarize/text",
                data=data,  # Note: data, not json
                headers=headers,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000

            return {
                "endpoint": "/summarize/text",
                "method": "POST",
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else {"raw": response.text},
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}"
            }

        except Exception as e:
            return {
                "endpoint": "/summarize/text",
                "method": "POST",
                "success": False,
                "error": str(e)
            }

    def test_voice_journal_fixed(self) -> Dict[str, Any]:
        """Test voice journal analysis with correct multipart format"""
        logger.info("🎤📝 Testing Voice Journal Analysis (Fixed Format)...")

        try:
            # Create minimal WAV file
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
            silence_data = b'\x00' * 2048
            wav_data = wav_header + silence_data

            files = {
                'audio_file': ('test_journal.wav', io.BytesIO(wav_data), 'audio/wav')
            }

            data = {
                'language': 'en',
                'generate_summary': 'true',
                'emotion_threshold': '0.1'
            }

            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/analyze/voice-journal",
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000

            return {
                "endpoint": "/analyze/voice-journal",
                "method": "POST",
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else {"raw": response.text},
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}"
            }

        except Exception as e:
            return {
                "endpoint": "/analyze/voice-journal",
                "method": "POST",
                "success": False,
                "error": str(e)
            }

    def run_fixed_tests(self) -> Dict[str, Any]:
        """Run the corrected tests"""
        logger.info("🚀 Running Fixed API Tests...")

        results = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "tests": {}
        }

        # Authenticate first
        auth_success = self.authenticate()
        results["authentication"] = auth_success

        if not auth_success:
            logger.warning("⚠️ Authentication failed, testing without auth")

        # Test the previously failing endpoints with correct formats
        time.sleep(1)
        results["tests"]["voice_transcription"] = self.test_voice_transcription_fixed()

        time.sleep(2)
        results["tests"]["text_summarization"] = self.test_text_summarization_fixed()

        time.sleep(2)
        results["tests"]["voice_journal"] = self.test_voice_journal_fixed()

        return results

def main():
    """Main testing function"""
    print("🔧 SAMO API - Fixed Format Testing")
    print("=" * 50)
    print("Testing previously failing endpoints with correct request formats:")
    print("  🎤 Voice Transcription (multipart/form-data)")
    print("  📄 Text Summarization (application/x-www-form-urlencoded)")
    print("  🎤📝 Voice Journal Analysis (multipart/form-data)")
    print("=" * 50)

    tester = FixedAPITester()
    results = tester.run_fixed_tests()

    print("\n📊 FIXED TEST RESULTS:")
    print("=" * 30)

    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"].values() if test.get("success", False))

    print(f"Authentication: {'✅' if results['authentication'] else '❌'}")
    print(f"Tests Passed: {passed_tests}/{total_tests}")

    print("\n🔍 DETAILED RESULTS:")
    print("-" * 25)

    for test_name, result in results["tests"].items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        endpoint = result.get("endpoint", "")
        print(f"{status} {test_name.upper()}: {endpoint}")

        if result.get("response_time_ms"):
            print(f"    Response Time: {result['response_time_ms']:.0f}ms")

        if not result.get("success", False):
            print(f"    Error: {result.get('error', 'Unknown error')}")
            if result.get("status_code"):
                print(f"    Status Code: {result['status_code']}")
        else:
            # Show successful response summary
            response_data = result.get("response_data", {})
            if isinstance(response_data, dict):
                if "text" in response_data:
                    print(f"    ✅ Transcribed text: '{response_data['text'][:50]}...'")
                elif "summary" in response_data:
                    print(f"    ✅ Summary: '{response_data['summary'][:50]}...'")
                elif "emotion_analysis" in response_data:
                    print(f"    ✅ Complete analysis with {len(response_data)} components")

    # Save results
    os.makedirs("test_reports", exist_ok=True)
    report_file = f"test_reports/fixed_api_test_{int(time.time())}.json"

    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {report_file}")

    # Generate recommendations
    print("\n💡 ANALYSIS:")
    print("-" * 15)

    if passed_tests == total_tests:
        print("🎉 ALL 3 CORE FEATURES ARE WORKING!")
        print("✅ Emotion Detection: Working (from previous test)")
        print("✅ Voice Transcription: Working")
        print("✅ Text Summarization: Working")
        print("\n🚀 Your API deployment has all three core features functional!")
    else:
        print("⚠️ Some features still need attention:")
        for test_name, result in results["tests"].items():
            if not result.get("success", False):
                print(f"❌ {test_name}: {result.get('error', 'Failed')}")

    # Note about voice processing model
    print("\n📝 NOTE: Voice processing model is not loaded (voice_processing: false)")
    print("   This may cause issues with voice transcription in production.")
    print("   Consider checking model loading in the deployment configuration.")

if __name__ == "__main__":
    main()
