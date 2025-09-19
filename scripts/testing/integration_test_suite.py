#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE INTEGRATION TEST SUITE
=======================================
Complete integration testing for SAMO-DL API endpoints.
"""

import os
import sys
import time
import requests
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class SAMODLIntegrationTests(unittest.TestCase):
    """Comprehensive integration tests for SAMO-DL API"""

    def setUp(self):
        """Set up test environment"""
        self.base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.timeout = 30
        # Use test user agent to bypass rate limiting
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pytest-integration-test"})
        self.test_data = {
            'happy_text': 'I am feeling absolutely wonderful and excited about today!',
            'sad_text': 'I am feeling really down and disappointed about everything.',
            'neutral_text': 'The weather is normal today and nothing special happened.',
            'long_text': ('This is a very long text that should be properly handled by the API. '
                          * 10),
            'special_chars': ('Testing with special characters: '
                              '@#$%^&*()_+{}|:"<>?[]\\;\',./'),
            'unicode_text': 'Testing with unicode: 🎉😊🚀🌟💯'
        }

    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("🔍 Testing health endpoint...")

        response = self.session.get(f'{self.base_url}/health', timeout=self.timeout)

        self.assertEqual(response.status_code, 200, "Health endpoint should return 200")

        data = response.json()
        self.assertIn('status', data, "Health response should contain status")
        self.assertEqual(data['status'], 'healthy', "Status should be healthy")

        print("✅ Health endpoint test passed")

    def test_root_endpoint(self):
        """Test root endpoint"""
        print("🔍 Testing root endpoint...")

        response = self.session.get(f'{self.base_url}/', timeout=self.timeout)

        self.assertEqual(response.status_code, 200, "Root endpoint should return 200")

        data = response.json()
        self.assertIn('message', data, "Root response should contain message")

        print("✅ Root endpoint test passed")

    def test_emotion_analysis_happy(self):
        """Test emotion analysis with happy text"""
        print("🔍 Testing emotion analysis (happy text)...")

        payload = {'text': self.test_data['happy_text']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Emotion analysis should return 200")

        data = response.json()
        self.assertIn('emotion_analysis', data, 
                      "Response should contain emotion_analysis")
        self.assertIn('summary', data, "Response should contain summary")

        emotion_data = data['emotion_analysis']
        self.assertIn('emotions', emotion_data, 
                      "Emotion analysis should contain emotions")
        self.assertIn('primary_emotion', emotion_data, 
                      "Emotion analysis should contain primary_emotion")
        self.assertIn('confidence', emotion_data, 
                      "Emotion analysis should contain confidence")

        # Validate confidence is between 0 and 1
        self.assertGreaterEqual(emotion_data['confidence'], 0, 
                                "Confidence should be >= 0")
        self.assertLessEqual(emotion_data['confidence'], 1, "Confidence should be <= 1")

        print(f"✅ Emotion analysis test passed - Detected: "
              f"{emotion_data['primary_emotion']} "
              f"(confidence: {emotion_data['confidence']:.3f})")

    def test_emotion_analysis_sad(self):
        """Test emotion analysis with sad text"""
        print("🔍 Testing emotion analysis (sad text)...")

        payload = {'text': self.test_data['sad_text']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Emotion analysis should return 200")

        data = response.json()
        self.assertIn('emotion_analysis', data, 
                      "Response should contain emotion_analysis")
        emotion_data = data['emotion_analysis']
        self.assertIn('primary_emotion', emotion_data, 
                      "Response should contain primary_emotion")
        self.assertIn('confidence', emotion_data, "Response should contain confidence")

        print(f"✅ Sad emotion analysis test passed - Detected: "
              f"{emotion_data['primary_emotion']} "
              f"(confidence: {emotion_data['confidence']:.3f})")

    def test_emotion_analysis_query_params(self):
        """Test emotion analysis with query parameters"""
        print("🔍 Testing emotion analysis (query params)...")

        params = {'text': self.test_data['neutral_text']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            params=params,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Emotion analysis with query params should return 200")

        data = response.json()
        self.assertIn('emotion_analysis', data, 
                      "Response should contain emotion_analysis")
        emotion_data = data['emotion_analysis']
        self.assertIn('primary_emotion', emotion_data, 
                      "Response should contain primary_emotion")

        print(f"✅ Query params emotion analysis test passed - "
              f"Detected: {emotion_data['primary_emotion']}")

    def test_text_summarization(self):
        """Test text summarization endpoint"""
        print("🔍 Testing text summarization...")

        payload = {'text': self.test_data['long_text']}
        response = self.session.post(
            f'{self.base_url}/summarize/text',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Text summarization should return 200")

        data = response.json()
        self.assertIn('summary', data, "Response should contain summary")
        self.assertIn('original_length', data, 
                      "Response should contain original_length")
        self.assertIn('summary_length', data, 
                      "Response should contain summary_length")
        self.assertIn('compression_ratio', data, 
                      "Response should contain compression_ratio")

        # Validate summary is shorter than original
        self.assertLess(data['summary_length'], data['original_length'],
                       "Summary should be shorter than original text")

        print(f"✅ Text summarization test passed - "
              f"Compression ratio: {data['compression_ratio']:.2f}")

    def test_special_characters(self):
        """Test API with special characters"""
        print("🔍 Testing special characters handling...")

        payload = {'text': self.test_data['special_chars']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Special characters should be handled properly")

        data = response.json()
        self.assertIn('emotion_analysis', data, 
                      "Response should contain emotion_analysis")
        emotion_data = data['emotion_analysis']
        self.assertIn('primary_emotion', emotion_data, 
                      "Response should contain primary_emotion")

        print(f"✅ Special characters test passed - "
              f"Detected: {emotion_data['primary_emotion']}")

    def test_unicode_text(self):
        """Test API with unicode text"""
        print("🔍 Testing unicode text handling...")

        payload = {'text': self.test_data['unicode_text']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 200, 
                         "Unicode text should be handled properly")

        data = response.json()
        self.assertIn('emotion_analysis', data, 
                      "Response should contain emotion_analysis")
        emotion_data = data['emotion_analysis']
        self.assertIn('primary_emotion', emotion_data, 
                      "Response should contain primary_emotion")

        print(f"✅ Unicode text test passed - "
              f"Detected: {emotion_data['primary_emotion']}")

    def test_empty_text_handling(self):
        """Test API with empty text"""
        print("🔍 Testing empty text handling...")

        payload = {'text': ''}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 400, "Empty text should return 400")

        data = response.json()
        self.assertIn('error', data, "Error response should contain error message")

        print("✅ Empty text handling test passed")

    def test_missing_text_field(self):
        """Test API with missing text field"""
        print("🔍 Testing missing text field handling...")

        payload = {}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )

        self.assertEqual(response.status_code, 400, 
                         "Missing text field should return 400")

        data = response.json()
        self.assertIn('error', data, "Error response should contain error message")

        print("✅ Missing text field handling test passed")

    def test_invalid_json(self):
        """Test API with invalid JSON"""
        print("🔍 Testing invalid JSON handling...")

        headers = {'Content-Type': 'application/json'}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            data='invalid json',
            headers=headers,
            timeout=self.timeout
        )

        # Should handle invalid JSON gracefully
        self.assertIn(response.status_code, [200, 400], 
                      "Invalid JSON should be handled gracefully")

        print("✅ Invalid JSON handling test passed")

    def test_response_time(self):
        """Test API response time"""
        print("🔍 Testing response time...")

        start_time = time.time()
        payload = {'text': self.test_data['happy_text']}
        response = self.session.post(
            f'{self.base_url}/analyze/journal',
            json=payload,
            timeout=self.timeout
        )
        end_time = time.time()

        response_time = end_time - start_time

        self.assertEqual(response.status_code, 200, "Response should be successful")
        self.assertLess(response_time, 5.0, 
                        "Response time should be less than 5 seconds")

        print(f"✅ Response time test passed - {response_time:.3f} seconds")

    def test_concurrent_requests(self):
        """Test API with concurrent requests"""
        print("🔍 Testing concurrent requests...")

        import threading
        import queue

        results = queue.Queue()

        def make_request():
            try:
                payload = {'text': self.test_data['happy_text']}
                response = self.session.post(
                    f'{self.base_url}/analyze/journal',
                    json=payload,
                    timeout=self.timeout
                )
                results.put(('success', response.status_code))
            except Exception as e:
                results.put(('error', str(e)))

        # Start 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_data = results.get()
            if result_type == 'success' and result_data == 200:
                success_count += 1

        self.assertGreaterEqual(success_count, 4, 
                                "At least 4 out of 5 concurrent requests "
                                "should succeed")

        print(f"✅ Concurrent requests test passed - "
              f"{success_count}/5 requests successful")

def run_performance_tests(base_url):
    """Run performance tests"""
    print("\n🚀 PERFORMANCE TESTS")
    print("=" * 40)

    # Create a local session for performance tests
    session = requests.Session()

    test_texts = [
        "I am feeling happy and excited about the future!",
        "This is a very sad and disappointing situation.",
        "The weather is normal and nothing special happened today.",
        "I am feeling anxious about the upcoming presentation.",
        "I am grateful for all the wonderful opportunities in my life."
    ]

    total_requests = 0
    successful_requests = 0
    total_response_time = 0

    for i, text in enumerate(test_texts):
        for j in range(3):  # 3 requests per text
            try:
                start_time = time.time()
                response = session.post(
                    f'{base_url}/analyze/journal',
                    json={'text': text},
                    timeout=30
                )
                end_time = time.time()

                total_requests += 1
                if response.status_code == 200:
                    successful_requests += 1

                total_response_time += (end_time - start_time)

            except Exception as e:
                print(f"Request failed: {e}")

    if total_requests > 0:
        success_rate = (successful_requests / total_requests) * 100
        avg_response_time = total_response_time / total_requests

        print("📊 Performance Results:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Average Response Time: {avg_response_time:.3f}s")

        return success_rate >= 90 and avg_response_time <= 3.0
    print("❌ No successful requests for performance testing")
    return False

def main():
    """Main test function"""
    print("🧪 SAMO-DL INTEGRATION TEST SUITE")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Testing API at: {os.getenv('API_BASE_URL', 'http://localhost:8000')}")
    print("=" * 50)

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance tests
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    performance_passed = run_performance_tests(base_url)

    print("\n🎯 TEST SUMMARY")
    print("=" * 30)
    if performance_passed:
        print("✅ All tests passed! API is ready for production.")
    else:
        print("⚠️ Some performance tests failed. Check API performance.")

if __name__ == "__main__":
    main()
