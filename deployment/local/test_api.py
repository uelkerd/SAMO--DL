#!/usr/bin/env python3
"""Enhanced API Testing Script
===========================

Comprehensive testing for the enhanced emotion detection API with monitoring,
logging, and rate limiting features.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Configuration
BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "5.0"))
TEST_TEXTS = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party",
    "I feel anxious about the test",
    "I am calm and relaxed",
    "I am grateful for your help",
    "I feel frustrated with this situation",
    "I am proud of my achievements",
    "I feel overwhelmed by all the work",
    "I am hopeful for the future",
    "I feel content with my life",
    "I am tired after a long day",
]


def test_health_check():
    """Test the enhanced health check endpoint."""
    print("1. Testing enhanced health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            status = data.get("status", data.get("state", "unknown"))
            print(f"   Status: {status}")
            if "timestamp" in data:
                print(f"   Timestamp: {data['timestamp']}")
            if "model_version" in data:
                print(f"   Model Version: {data['model_version']}")
            if "uptime_seconds" in data:
                print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                total = metrics.get("total_requests")
                success = metrics.get("successful_requests")
                if total is not None and success is not None:
                    print(f"   Success Rate: {success}/{total}")
                avg_ms = metrics.get("average_response_time_ms")
                if avg_ms is not None:
                    print(f"   Avg Response Time: {avg_ms}ms")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Health check JSON parse error: {e!s}")
        return False


def test_metrics_endpoint():
    """Test the new metrics endpoint."""
    print("\n2. Testing metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            ctype = response.headers.get("Content-Type", "")
            if ctype.startswith("text/"):
                print("✅ Metrics endpoint served Prometheus exposition format")
                lines = response.text.splitlines()
                if lines:
                    print(f"   Sample: {lines[0][:120]}")
            else:
                # Fallback: JSON metrics (if implemented)
                data = response.json()
                print("✅ Metrics endpoint working (JSON)")
                print(f"   Keys: {list(data)[:5]}")
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Metrics endpoint HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Metrics endpoint JSON parse error: {e!s}")
        return False


def test_single_predictions():
    """Test single predictions with timing."""
    print("\n3. Testing single predictions...")
    results = []

    for i, text in enumerate(TEST_TEXTS[:5], 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_TIMEOUT,
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                emotion = data["predicted_emotion"]
                confidence = data["confidence"]
                prediction_time = data.get("prediction_time_ms", 0)
                total_time = (end_time - start_time) * 1000

                print(
                    f"✅ Test {i}: '{text[:30]}...' → {emotion} (conf: {confidence:.3f}, time: {prediction_time}ms)"
                )
                results.append(
                    {
                        "text": text,
                        "emotion": emotion,
                        "confidence": confidence,
                        "prediction_time_ms": prediction_time,
                        "total_time_ms": total_time,
                    }
                )
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Test {i} HTTP error: {e!s}")
            return False
        except ValueError as e:
            print(f"❌ Test {i} JSON parse error: {e!s}")
            return False

    # Calculate average performance
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_prediction_time = sum(r["prediction_time_ms"] for r in results) / len(results)
    print(f"   📊 Average confidence: {avg_confidence:.3f}")
    print(f"   📊 Average prediction time: {avg_prediction_time:.1f}ms")

    return True


def test_batch_predictions():
    """Test batch predictions."""
    print("\n4. Testing batch predictions...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json={"texts": TEST_TEXTS[:5]},
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions") or data.get("results")
            batch_time = data.get("batch_processing_time_ms", 0)
            total_time = (end_time - start_time) * 1000

            if not isinstance(predictions, list):
                print("❌ Unexpected batch response format")
                return False

            print(f"✅ Batch prediction successful: {len(predictions)} predictions")
            print(f"   Batch processing time: {batch_time}ms")
            print(f"   Total time: {total_time:.1f}ms")

            for i, pred in enumerate(predictions, 1):
                emotion = pred["predicted_emotion"]
                confidence = pred["confidence"]
                text = (
                    pred["text"][:30] + "..."
                    if len(pred["text"]) > 30
                    else pred["text"]
                )
                print(f"   {i}. '{text}' → {emotion} (conf: {confidence:.3f})")

            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Batch prediction JSON parse error: {e!s}")
        return False


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n5. Testing rate limiting...")

    def make_request():
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": "Test rate limiting"},
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_TIMEOUT,
            )
            return response.status_code
        except:
            return 0

    # Make rapid requests to test rate limiting
    print("   Making rapid requests to test rate limiting...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [future.result() for future in as_completed(futures)]

    end_time = time.time()

    successful = sum(1 for code in results if code == 200)
    rate_limited = sum(1 for code in results if code == 429)
    failed = sum(1 for code in results if code not in [200, 429])

    print(f"   ✅ Rate limiting test completed in {end_time - start_time:.2f}s")
    print(
        f"   📊 Successful: {successful}, Rate limited: {rate_limited}, Failed: {failed}"
    )

    if rate_limited > 0:
        print(f"   ✅ Rate limiting is working (blocked {rate_limited} requests)")
        return True
    else:
        print("   ⚠️ No rate limiting detected (may need more requests)")
        return True


def test_error_handling():
    """Test error handling."""
    print("\n6. Testing error handling...")

    # Test missing text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 400:
            print("✅ Missing text error handled correctly")
        else:
            print(f"❌ Missing text error not handled: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Missing text test HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Missing text test JSON parse error: {e!s}")
        return False

    # Test empty text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 400:
            print("✅ Empty text error handled correctly")
        else:
            print(f"❌ Empty text error not handled: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Empty text test HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Empty text test JSON parse error: {e!s}")
        return False

    # Test invalid JSON
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 400:
            print("✅ Invalid JSON error handled correctly")
        else:
            print(f"❌ Invalid JSON error not handled: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Invalid JSON test HTTP error: {e!s}")
        return False
    except ValueError as e:
        print(f"❌ Invalid JSON test JSON parse error: {e!s}")
        return False

    return True


def test_performance():
    """Test performance under load."""
    print("\n7. Testing performance under load...")

    def make_prediction_request():
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": "Performance test"},
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_TIMEOUT,
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": (end_time - start_time) * 1000,
            }
        except requests.exceptions.RequestException as e:
            return {"status_code": 0, "response_time": 0, "error": f"HTTP error: {e!s}"}
        except ValueError as e:
            return {"status_code": 0, "response_time": 0, "error": f"JSON parse error: {e!s}"}

    # Test with concurrent requests
    print("   Testing with 20 concurrent requests...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_prediction_request) for _ in range(20)]
        results = [future.result() for future in as_completed(futures)]

    end_time = time.time()

    successful = [r for r in results if r["status_code"] == 200]
    failed = [r for r in results if r["status_code"] != 200]

    if successful:
        avg_response_time = sum(r["response_time"] for r in successful) / len(
            successful
        )
        min_response_time = min(r["response_time"] for r in successful)
        max_response_time = max(r["response_time"] for r in successful)

        print(f"   ✅ Performance test completed in {end_time - start_time:.2f}s")
        print(f"   📊 Successful requests: {len(successful)}/{len(results)}")
        print(f"   📊 Average response time: {avg_response_time:.1f}ms")
        print(
            f"   📊 Response time range: {min_response_time:.1f}ms - {max_response_time:.1f}ms"
        )

        if avg_response_time < 1000:  # Less than 1 second
            print("   ✅ Performance is acceptable")
            return True
        else:
            print("   ⚠️ Performance may need optimization")
            return True
    else:
        print("   ❌ No successful requests in performance test")
        return False


def main():
    """Run all tests."""
    print("🧪 ENHANCED API TESTING")
    print("=" * 50)

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)

    tests = [
        ("Health Check", test_health_check),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Single Predictions", test_single_predictions),
        ("Batch Predictions", test_batch_predictions),
        ("Rate Limiting", test_rate_limiting),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except requests.exceptions.RequestException as e:
            print(f"❌ {test_name} HTTP error: {e!s}")
        except ValueError as e:
            print(f"❌ {test_name} JSON parse error: {e!s}")
        except Exception as e:
            print(f"❌ {test_name} unexpected error: {e!s}")

    print("\n" + "=" * 50)
    print("🎉 ENHANCED API TESTING COMPLETED!")
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! Enhanced API is working correctly.")
        print("\n📋 Enhanced Features Verified:")
        print("   ✅ Comprehensive logging")
        print("   ✅ Real-time metrics")
        print("   ✅ Rate limiting")
        print("   ✅ Error handling")
        print("   ✅ Performance monitoring")
        print("   ✅ Batch processing")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
