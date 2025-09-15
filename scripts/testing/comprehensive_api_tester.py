#!/usr/bin/env python3
"""
Comprehensive API Testing Script for SAMO Cloud Run Deployment
Tests all 3 core features: Emotion Detection, Voice Transcription, Text Summarization
"""

import requests
import json
import time
import sys
import os
import logging
import tempfile
import io
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_api_test.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Structure for test results"""
    endpoint: str
    method: str
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    response_data: Optional[Dict] = None
    error_message: Optional[str] = None
    feature: Optional[str] = None

@dataclass
class APITestReport:
    """Comprehensive test report"""
    timestamp: str
    base_url: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    features_tested: List[str]
    rate_limited: bool
    authentication_working: bool
    overall_status: str
    test_results: List[TestResult]
    recommendations: List[str]

class ComprehensiveAPITester:
    def __init__(self, base_url: str = "https://samo-unified-api-frrnetyhfa-uc.a.run.app"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SAMO-Comprehensive-Tester/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        # Test configuration
        self.timeout = 30
        self.retry_delay = 2
        self.max_retries = 3
        self.rate_limit_delay = 10

        # Authentication token (will be obtained via login)
        self.auth_token = None

        # Test results storage
        self.test_results = []

        # Test data
        self.test_texts = [
            "I am feeling incredibly happy and excited about this new opportunity!",
            "This situation is making me quite anxious and nervous.",
            "I feel a deep sense of gratitude for all the support I've received.",
            "I'm neutral about this entire situation.",
            "This is absolutely frustrating and disappointing."
        ]

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                    files: Optional[Dict] = None, auth_required: bool = False) -> TestResult:
        """Make HTTP request with proper error handling and rate limiting"""

        url = f"{self.base_url}{endpoint}"
        headers = self.session.headers.copy()

        # Add authentication if required and available
        if auth_required and self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'

        start_time = time.time()

        try:
            # Make request based on method
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, timeout=self.timeout)
            elif method.upper() == 'POST':
                if files:
                    # For file uploads, don't set Content-Type (let requests handle it)
                    headers.pop('Content-Type', None)
                    response = self.session.post(url, headers=headers, files=files,
                                               data=data, timeout=self.timeout)
                else:
                    response = self.session.post(url, headers=headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response_time = (time.time() - start_time) * 1000

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', self.rate_limit_delay))
                logger.warning(f"Rate limited on {endpoint}, waiting {retry_after}s")
                time.sleep(retry_after)
                return TestResult(
                    endpoint=endpoint,
                    method=method,
                    success=False,
                    status_code=429,
                    response_time_ms=response_time,
                    error_message="Rate limited"
                )

            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}

            success = response.status_code < 400

            return TestResult(
                endpoint=endpoint,
                method=method,
                success=success,
                status_code=response.status_code,
                response_time_ms=response_time,
                response_data=response_data,
                error_message=None if success else response_data.get('error', f'HTTP {response.status_code}')
            )

        except requests.exceptions.Timeout:
            return TestResult(
                endpoint=endpoint,
                method=method,
                success=False,
                error_message=f"Request timeout after {self.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            return TestResult(
                endpoint=endpoint,
                method=method,
                success=False,
                error_message=f"Request failed: {str(e)}"
            )

    def test_authentication(self) -> Tuple[bool, str]:
        """Test authentication endpoints"""
        logger.info("🔐 Testing Authentication...")

        # Test user registration
        register_data = {
            "username": f"test_user_{int(time.time())}@example.com",
            "email": f"test_user_{int(time.time())}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User"
        }

        register_result = self.make_request('POST', '/auth/register', register_data)
        self.test_results.append(register_result)

        if register_result.success and register_result.response_data:
            # Extract token from registration response
            access_token = register_result.response_data.get('access_token')
            if access_token:
                self.auth_token = access_token
                logger.info("✅ Authentication successful - token obtained")
                return True, "Authentication working"

        # Try login as fallback
        login_data = {
            "username": "test@example.com",
            "password": "password123"
        }

        login_result = self.make_request('POST', '/auth/login', login_data)
        self.test_results.append(login_result)

        if login_result.success and login_result.response_data:
            access_token = login_result.response_data.get('access_token')
            if access_token:
                self.auth_token = access_token
                logger.info("✅ Login successful - token obtained")
                return True, "Authentication working"

        logger.warning("⚠️ Authentication failed or not required")
        return False, "Authentication failed or not available"

    def test_health_endpoints(self) -> List[TestResult]:
        """Test health and status endpoints"""
        logger.info("🏥 Testing Health & Status Endpoints...")

        results = []

        # Test basic health endpoint
        health_result = self.make_request('GET', '/health')
        health_result.feature = "System Health"
        results.append(health_result)

        # Test root endpoint
        root_result = self.make_request('GET', '/')
        root_result.feature = "System Info"
        results.append(root_result)

        # Test models status
        models_result = self.make_request('GET', '/models/status')
        models_result.feature = "Models Status"
        results.append(models_result)

        self.test_results.extend(results)
        return results

    def test_emotion_detection(self) -> List[TestResult]:
        """Test emotion detection endpoints"""
        logger.info("😊 Testing Emotion Detection...")

        results = []

        for i, text in enumerate(self.test_texts[:3]):  # Test first 3 texts
            # Test unified journal analysis endpoint
            journal_data = {
                "text": text,
                "generate_summary": True,
                "emotion_threshold": 0.1
            }

            journal_result = self.make_request('POST', '/analyze/journal', journal_data, auth_required=True)
            journal_result.feature = "Emotion Detection"
            results.append(journal_result)

            # Add small delay to avoid rate limiting
            time.sleep(1)

        self.test_results.extend(results)
        return results

    def test_voice_transcription(self) -> List[TestResult]:
        """Test voice transcription endpoints"""
        logger.info("🎤 Testing Voice Transcription...")

        results = []

        # Create a minimal WAV file for testing (silence)
        def create_minimal_wav() -> bytes:
            """Create a minimal valid WAV file with silence"""
            # WAV header for 1 second of silence, mono, 16-bit, 8kHz
            header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
            # 1 second of silence at 8kHz, 16-bit = 16000 bytes of zeros
            silence = b'\x00' * 2048  # Shorter for testing
            return header + silence

        # Test voice transcription endpoint
        try:
            wav_data = create_minimal_wav()
            files = {
                'audio_file': ('test.wav', io.BytesIO(wav_data), 'audio/wav')
            }
            data = {
                'language': 'en',
                'model_size': 'base'
            }

            transcribe_result = self.make_request('POST', '/transcribe/voice', data=data,
                                                files=files, auth_required=True)
            transcribe_result.feature = "Voice Transcription"
            results.append(transcribe_result)

        except Exception as e:
            error_result = TestResult(
                endpoint='/transcribe/voice',
                method='POST',
                success=False,
                feature="Voice Transcription",
                error_message=f"Failed to create test audio: {str(e)}"
            )
            results.append(error_result)

        self.test_results.extend(results)
        return results

    def test_text_summarization(self) -> List[TestResult]:
        """Test text summarization endpoints"""
        logger.info("📄 Testing Text Summarization...")

        results = []

        # Test text summarization endpoint
        long_text = " ".join(self.test_texts)  # Combine texts for summarization

        summarize_data = {
            'text': long_text,
            'model': 't5-small',
            'max_length': 150,
            'min_length': 30
        }

        # Use form data for summarization endpoint
        summarize_result = self.make_request('POST', '/summarize/text', summarize_data, auth_required=True)
        summarize_result.feature = "Text Summarization"
        results.append(summarize_result)

        self.test_results.extend(results)
        return results

    def test_advanced_features(self) -> List[TestResult]:
        """Test advanced features like WebSocket, batch processing"""
        logger.info("🚀 Testing Advanced Features...")

        results = []

        # Note: WebSocket testing would require different approach
        # For now, we'll focus on HTTP endpoints

        # Test voice journal analysis (end-to-end)
        try:
            wav_data = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00' + b'\x00' * 2048

            files = {
                'audio_file': ('test.wav', io.BytesIO(wav_data), 'audio/wav')
            }
            data = {
                'language': 'en',
                'generate_summary': 'true',
                'emotion_threshold': '0.1'
            }

            voice_journal_result = self.make_request('POST', '/analyze/voice-journal',
                                                   data=data, files=files, auth_required=True)
            voice_journal_result.feature = "Voice Journal Analysis"
            results.append(voice_journal_result)

        except Exception as e:
            error_result = TestResult(
                endpoint='/analyze/voice-journal',
                method='POST',
                success=False,
                feature="Voice Journal Analysis",
                error_message=f"Failed to test voice journal: {str(e)}"
            )
            results.append(error_result)

        self.test_results.extend(results)
        return results

    def generate_report(self) -> APITestReport:
        """Generate comprehensive test report"""
        logger.info("📊 Generating Test Report...")

        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - passed_tests

        # Determine features tested
        features_tested = list(set([r.feature for r in self.test_results if r.feature]))

        # Check for rate limiting
        rate_limited = any(r.status_code == 429 for r in self.test_results)

        # Check authentication
        auth_working = self.auth_token is not None

        # Overall status
        if passed_tests == total_tests:
            overall_status = "EXCELLENT"
        elif passed_tests >= total_tests * 0.8:
            overall_status = "GOOD"
        elif passed_tests >= total_tests * 0.5:
            overall_status = "FAIR"
        else:
            overall_status = "POOR"

        # Generate recommendations
        recommendations = []

        if rate_limited:
            recommendations.append("API has aggressive rate limiting - consider implementing authentication for testing")

        if not auth_working:
            recommendations.append("Authentication should be implemented for secure testing")

        if failed_tests > 0:
            failed_features = set([r.feature for r in self.test_results if not r.success and r.feature])
            if failed_features:
                recommendations.append(f"The following features need attention: {', '.join(failed_features)}")

        # Check which core features are working
        core_features = ["Emotion Detection", "Voice Transcription", "Text Summarization"]
        working_features = []
        failing_features = []

        for feature in core_features:
            feature_tests = [r for r in self.test_results if r.feature == feature]
            if feature_tests:
                if any(r.success for r in feature_tests):
                    working_features.append(feature)
                else:
                    failing_features.append(feature)
            else:
                failing_features.append(f"{feature} (not tested)")

        if len(working_features) == 3:
            recommendations.append("✅ All 3 core features are working!")
        else:
            recommendations.append(f"❌ Missing or failing features: {', '.join(failing_features)}")
            recommendations.append(f"✅ Working features: {', '.join(working_features)}")

        return APITestReport(
            timestamp=datetime.now().isoformat(),
            base_url=self.base_url,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            features_tested=features_tested,
            rate_limited=rate_limited,
            authentication_working=auth_working,
            overall_status=overall_status,
            test_results=self.test_results,
            recommendations=recommendations
        )

    def run_comprehensive_test(self) -> APITestReport:
        """Run all tests and generate comprehensive report"""
        logger.info("🚀 Starting Comprehensive API Testing...")
        logger.info(f"Testing API at: {self.base_url}")

        try:
            # Phase 1: Authentication
            self.test_authentication()
            time.sleep(2)  # Rate limiting delay

            # Phase 2: Health & Status
            self.test_health_endpoints()
            time.sleep(2)

            # Phase 3: Core Features
            self.test_emotion_detection()
            time.sleep(3)

            self.test_voice_transcription()
            time.sleep(3)

            self.test_text_summarization()
            time.sleep(3)

            # Phase 4: Advanced Features
            self.test_advanced_features()

        except Exception as e:
            logger.error(f"Critical error during testing: {str(e)}")

        # Generate final report
        return self.generate_report()

def main():
    """Main function"""
    print("🧪 SAMO Cloud Run API - Comprehensive Testing")
    print("=" * 60)
    print("Testing ALL 3 Core Features:")
    print("  1. 😊 Emotion Detection")
    print("  2. 🎤 Voice Transcription")
    print("  3. 📄 Text Summarization")
    print("=" * 60)

    # Create tester
    tester = ComprehensiveAPITester()

    # Run comprehensive test
    report = tester.run_comprehensive_test()

    # Print summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"Overall Status: {report.overall_status}")
    print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
    print(f"Features Tested: {', '.join(report.features_tested)}")
    print(f"Authentication: {'✅ Working' if report.authentication_working else '❌ Failed'}")
    print(f"Rate Limited: {'⚠️ Yes' if report.rate_limited else '✅ No'}")

    print("\n🔍 DETAILED RESULTS:")
    print("-" * 30)
    for result in report.test_results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        feature = f"[{result.feature}]" if result.feature else ""
        print(f"{status} {result.method} {result.endpoint} {feature}")
        if not result.success and result.error_message:
            print(f"    Error: {result.error_message}")
        if result.response_time_ms:
            print(f"    Response Time: {result.response_time_ms:.0f}ms")

    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")

    # Save detailed report
    os.makedirs("test_reports", exist_ok=True)
    report_file = f"test_reports/comprehensive_api_test_{int(time.time())}.json"

    with open(report_file, 'w') as f:
        # Convert dataclasses to dict for JSON serialization
        report_dict = asdict(report)
        json.dump(report_dict, f, indent=2, default=str)

    print(f"\n💾 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    exit_code = 0 if report.overall_status in ["EXCELLENT", "GOOD"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()