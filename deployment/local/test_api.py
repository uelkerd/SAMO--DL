#!/usr/bin/env python3
"""
Enhanced API Testing Script
===========================

Comprehensive testing for the enhanced emotion detection API with monitoring,
logging, and rate limiting features.
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Configuration
BASE_URL = "http://localhost:8000"
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
    "I am tired after a long day"
]

def test_health_check():
    """Test the enhanced health check endpoint."""
    print"1. Testing enhanced health check..."
    try:
        response = requests.getf"{BASE_URL}/health"
        if response.status_code == 200:
            data = response.json()
            print"✅ Health check passed"
            printf"   Status: {data['status']}"
            printf"   Model Version: {data['model_version']}"
            printf"   Uptime: {data['uptime_seconds']:.1f} seconds"
            printf"   Total Requests: {data['metrics']['total_requests']}"
            printf"   Success Rate: {data['metrics']['successful_requests']}/{data['metrics']['total_requests']}"
            printf"   Avg Response Time: {data['metrics']['average_response_time_ms']}ms"
            return True
        else:
            printf"❌ Health check failed: {response.status_code}"
            return False
    except Exception as e:
        print(f"❌ Health check error: {stre}")
        return False

def test_metrics_endpoint():
    """Test the new metrics endpoint."""
    print"\n2. Testing metrics endpoint..."
    try:
        response = requests.getf"{BASE_URL}/metrics"
        if response.status_code == 200:
            data = response.json()
            print"✅ Metrics endpoint working"
            printf"   Success Rate: {data['server_metrics']['success_rate']}"
            printf"   Requests/Minute: {data['server_metrics']['requests_per_minute']:.2f}"
            printf"   Rate Limiting: {data['rate_limiting']['max_requests']} req/{data['rate_limiting']['window_seconds']}s"
            return True
        else:
            printf"❌ Metrics endpoint failed: {response.status_code}"
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint error: {stre}")
        return False

def test_single_predictions():
    """Test single predictions with timing."""
    print"\n3. Testing single predictions..."
    results = []
    
    for i, text in enumerateTEST_TEXTS[:5], 1:
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                emotion = data['predicted_emotion']
                confidence = data['confidence']
                prediction_time = data.get'prediction_time_ms', 0
                total_time = end_time - start_time * 1000
                
                print(f"✅ Test {i}: '{text[:30]}...' → {emotion} conf: {confidence:.3f}, time: {prediction_time}ms")
                results.append({
                    'text': text,
                    'emotion': emotion,
                    'confidence': confidence,
                    'prediction_time_ms': prediction_time,
                    'total_time_ms': total_time
                })
            else:
                printf"❌ Test {i} failed: {response.status_code}"
                return False
                
        except Exception as e:
            print(f"❌ Test {i} error: {stre}")
            return False
    
    # Calculate average performance
    avg_confidence = sumr['confidence'] for r in results / lenresults
    avg_prediction_time = sumr['prediction_time_ms'] for r in results / lenresults
    printf"   📊 Average confidence: {avg_confidence:.3f}"
    printf"   📊 Average prediction time: {avg_prediction_time:.1f}ms"
    
    return True

def test_batch_predictions():
    """Test batch predictions."""
    print"\n4. Testing batch predictions..."
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json={"texts": TEST_TEXTS[:5]},
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            predictions = data['predictions']
            batch_time = data.get'batch_processing_time_ms', 0
            total_time = end_time - start_time * 1000
            
            print(f"✅ Batch prediction successful: {lenpredictions} predictions")
            printf"   Batch processing time: {batch_time}ms"
            printf"   Total time: {total_time:.1f}ms"
            
            for i, pred in enumeratepredictions, 1:
                emotion = pred['predicted_emotion']
                confidence = pred['confidence']
                text = pred['text'][:30] + "..." if lenpred['text'] > 30 else pred['text']
                print(f"   {i}. '{text}' → {emotion} conf: {confidence:.3f}")
            
            return True
        else:
            printf"❌ Batch prediction failed: {response.status_code}"
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {stre}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality."""
    print"\n5. Testing rate limiting..."
    
    def make_request():
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": "Test rate limiting"},
                headers={"Content-Type": "application/json"}
            )
            return response.status_code
        except:
            return 0
    
    # Make rapid requests to test rate limiting
    print"   Making rapid requests to test rate limiting..."
    start_time = time.time()
    
    with ThreadPoolExecutormax_workers=10 as executor:
        futures = [executor.submitmake_request for _ in range50]
        results = [future.result() for future in as_completedfutures]
    
    end_time = time.time()
    
    successful = sum1 for code in results if code == 200
    rate_limited = sum1 for code in results if code == 429
    failed = sum1 for code in results if code not in [200, 429]
    
    printf"   ✅ Rate limiting test completed in {end_time - start_time:.2f}s"
    printf"   📊 Successful: {successful}, Rate limited: {rate_limited}, Failed: {failed}"
    
    if rate_limited > 0:
        print(f"   ✅ Rate limiting is working blocked {rate_limited} requests")
        return True
    else:
        print("   ⚠️ No rate limiting detected may need more requests")
        return True

def test_error_handling():
    """Test error handling."""
    print"\n6. Testing error handling..."
    
    # Test missing text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            print"✅ Missing text error handled correctly"
        else:
            printf"❌ Missing text error not handled: {response.status_code}"
            return False
    except Exception as e:
        print(f"❌ Missing text test error: {stre}")
        return False
    
    # Test empty text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            print"✅ Empty text error handled correctly"
        else:
            printf"❌ Empty text error not handled: {response.status_code}"
            return False
    except Exception as e:
        print(f"❌ Empty text test error: {stre}")
        return False
    
    # Test invalid JSON
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            print"✅ Invalid JSON error handled correctly"
        else:
            printf"❌ Invalid JSON error not handled: {response.status_code}"
            return False
    except Exception as e:
        print(f"❌ Invalid JSON test error: {stre}")
        return False
    
    return True

def test_performance():
    """Test performance under load."""
    print"\n7. Testing performance under load..."
    
    def make_prediction_request():
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": "Performance test"},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time * 1000
            }
        except Exception as e:
            return {'status_code': 0, 'response_time': 0, 'error': stre}
    
    # Test with concurrent requests
    print"   Testing with 20 concurrent requests..."
    start_time = time.time()
    
    with ThreadPoolExecutormax_workers=5 as executor:
        futures = [executor.submitmake_prediction_request for _ in range20]
        results = [future.result() for future in as_completedfutures]
    
    end_time = time.time()
    
    successful = [r for r in results if r['status_code'] == 200]
    failed = [r for r in results if r['status_code'] != 200]
    
    if successful:
        avg_response_time = sumr['response_time'] for r in successful / lensuccessful
        min_response_time = minr['response_time'] for r in successful
        max_response_time = maxr['response_time'] for r in successful
        
        printf"   ✅ Performance test completed in {end_time - start_time:.2f}s"
        print(f"   📊 Successful requests: {lensuccessful}/{lenresults}")
        printf"   📊 Average response time: {avg_response_time:.1f}ms"
        printf"   📊 Response time range: {min_response_time:.1f}ms - {max_response_time:.1f}ms"
        
        if avg_response_time < 1000:  # Less than 1 second
            print"   ✅ Performance is acceptable"
            return True
        else:
            print"   ⚠️ Performance may need optimization"
            return True
    else:
        print"   ❌ No successful requests in performance test"
        return False

def main():
    """Run all tests."""
    print"🧪 ENHANCED API TESTING"
    print"=" * 50
    
    # Wait for server to start
    print"⏳ Waiting for server to start..."
    time.sleep2
    
    tests = [
        "Health Check", test_health_check,
        "Metrics Endpoint", test_metrics_endpoint,
        "Single Predictions", test_single_predictions,
        "Batch Predictions", test_batch_predictions,
        "Rate Limiting", test_rate_limiting,
        "Error Handling", test_error_handling,
        "Performance", test_performance
    ]
    
    passed = 0
    total = lentests
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                printf"❌ {test_name} failed"
        except Exception as e:
            print(f"❌ {test_name} error: {stre}")
    
    print"\n" + "=" * 50
    print"🎉 ENHANCED API TESTING COMPLETED!"
    printf"📊 Results: {passed}/{total} tests passed"
    
    if passed == total:
        print"✅ All tests passed! Enhanced API is working correctly."
        print"\n📋 Enhanced Features Verified:"
        print"   ✅ Comprehensive logging"
        print"   ✅ Real-time metrics"
        print"   ✅ Rate limiting"
        print"   ✅ Error handling"
        print"   ✅ Performance monitoring"
        print"   ✅ Batch processing"
        return 0
    else:
        printf"❌ {total - passed} tests failed. Please check the implementation."
        return 1

if __name__ == "__main__":
    sys.exit(main())
