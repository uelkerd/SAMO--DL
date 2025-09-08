#!/usr/bin/env python3
"""Enhanced API Testing Script.
===========================

Comprehensive testing for the enhanced emotion detection API with monitoring,
logging, and rate limiting features.
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Optional

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

def test_health_check() -> Optional[bool]:
    """Test the enhanced health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            response.json()
            return True
        else:
            return False
    except Exception:
        return False

def test_metrics_endpoint() -> Optional[bool]:
    """Test the new metrics endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            response.json()
            return True
        else:
            return False
    except Exception:
        return False

def test_single_predictions() -> bool:
    """Test single predictions with timing."""
    results = []
    
    for _i, text in enumerate(TEST_TEXTS[:5], 1):
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
                prediction_time = data.get('prediction_time_ms', 0)
                total_time = (end_time - start_time) * 1000
                
                results.append({
                    'text': text,
                    'emotion': emotion,
                    'confidence': confidence,
                    'prediction_time_ms': prediction_time,
                    'total_time_ms': total_time
                })
            else:
                return False
                
        except Exception:
            return False
    
    # Calculate average performance
    sum(r['confidence'] for r in results) / len(results)
    sum(r['prediction_time_ms'] for r in results) / len(results)
    
    return True

def test_batch_predictions() -> Optional[bool]:
    """Test batch predictions."""
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
            data.get('batch_processing_time_ms', 0)
            
            
            for _i, pred in enumerate(predictions, 1):
                pred['text'][:30] + "..." if len(pred['text']) > 30 else pred['text']
            
            return True
        else:
            return False
            
    except Exception:
        return False

def test_rate_limiting() -> bool:
    """Test rate limiting functionality."""
    
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
    time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [future.result() for future in as_completed(futures)]
    
    time.time()
    
    sum(1 for code in results if code == 200)
    rate_limited = sum(1 for code in results if code == 429)
    sum(1 for code in results if code not in [200, 429])
    
    
    return rate_limited > 0

def test_error_handling() -> bool:
    """Test error handling."""
    # Test missing text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            pass
        else:
            return False
    except Exception:
        return False
    
    # Test empty text
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            pass
        else:
            return False
    except Exception:
        return False
    
    # Test invalid JSON
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            pass
        else:
            return False
    except Exception:
        return False
    
    return True

def test_performance() -> bool:
    """Test performance under load."""
    
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
                'response_time': (end_time - start_time) * 1000
            }
        except Exception as e:
            return {'status_code': 0, 'response_time': 0, 'error': str(e)}
    
    # Test with concurrent requests
    time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_prediction_request) for _ in range(20)]
        results = [future.result() for future in as_completed(futures)]
    
    time.time()
    
    successful = [r for r in results if r['status_code'] == 200]
    
    if successful:
        avg_response_time = sum(r['response_time'] for r in successful) / len(successful)
        min(r['response_time'] for r in successful)
        max(r['response_time'] for r in successful)
        
        
        return avg_response_time < 1000
    else:
        return False

def main() -> int:
    """Run all tests."""
    # Wait for server to start
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Single Predictions", test_single_predictions),
        ("Batch Predictions", test_batch_predictions),
        ("Rate Limiting", test_rate_limiting),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for _test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                pass
        except Exception:
            pass
    
    
    if passed == total:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
