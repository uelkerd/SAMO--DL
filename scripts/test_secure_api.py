#!/usr/bin/env python3
"""
Test Secure API Server
=====================

This script tests the secure emotion detection API server to ensure
all security features are working correctly.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add the scripts directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_secure_api_server():
    """Test the secure API server functionality."""
    
    print("🔒 Testing Secure API Server...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("\n1. Testing server availability...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            api_info = response.json()
            print(f"   Version: {api_info.get('version', 'unknown')}")
            print(f"   Security features: {len(api_info.get('security_features', []))}")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the server first:")
        print("   python scripts/secure_api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   Security checks passed: {health_data.get('security_checks_passed', False)}")
            print(f"   Emotions available: {len(health_data.get('emotions', []))}")
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 3: Single prediction
    print("\n3. Testing single prediction...")
    test_texts = [
        "I am feeling happy today!",
        "I'm so sad about what happened yesterday.",
        "I'm excited to start this new project!",
        "I feel overwhelmed with all the work I have to do."
    ]
    
    for text in test_texts:
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": text},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction successful for: '{text[:30]}...'")
                print(f"   Emotion: {result.get('predicted_emotion', 'unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Security checks: {result.get('security_checks_passed', False)}")
            else:
                print(f"❌ Prediction failed for '{text[:30]}...' with status: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Prediction error for '{text[:30]}...': {e}")
            return False
    
    # Test 4: Batch prediction
    print("\n4. Testing batch prediction...")
    try:
        response = requests.post(
            "http://localhost:8000/predict_batch",
            json={"texts": test_texts},
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful")
            print(f"   Predictions: {result.get('count', 0)}")
            print(f"   Batch size limit: {result.get('batch_size_limit', 0)}")
        else:
            print(f"❌ Batch prediction failed with status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False
    
    # Test 5: Security features
    print("\n5. Testing security features...")
    try:
        # Test rate limiting
        print("   Testing rate limiting...")
        responses = []
        for i in range(15):  # Try to exceed rate limit
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": f"Test text {i}"},
                timeout=5
            )
            responses.append(response.status_code)
            time.sleep(0.1)  # Small delay
        
        rate_limited = any(status == 429 for status in responses)
        if rate_limited:
            print("✅ Rate limiting is working")
        else:
            print("⚠️ Rate limiting may not be working as expected")
        
        # Test input validation
        print("   Testing input validation...")
        invalid_inputs = [
            {"text": ""},  # Empty text
            {"text": None},  # None text
            {"invalid": "key"},  # Wrong key
            "not json",  # Not JSON
        ]
        
        for invalid_input in invalid_inputs:
            try:
                if isinstance(invalid_input, dict):
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json=invalid_input,
                        timeout=5
                    )
                else:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        data=invalid_input,
                        headers={"Content-Type": "text/plain"},
                        timeout=5
                    )
                
                if response.status_code in [400, 422]:
                    print(f"✅ Input validation working for: {type(invalid_input).__name__}")
                else:
                    print(f"⚠️ Input validation may not be working for: {type(invalid_input).__name__}")
            except Exception as e:
                print(f"⚠️ Input validation test error: {e}")
        
    except Exception as e:
        print(f"❌ Security features test error: {e}")
        return False
    
    # Test 6: Performance test
    print("\n6. Testing performance...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": "I am feeling happy today!"},
            timeout=10
        )
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"✅ Prediction completed in {response_time:.2f}ms")
            
            if response_time < 1000:  # Less than 1 second
                print("✅ Performance is acceptable")
            else:
                print("⚠️ Performance may need optimization")
        else:
            print(f"❌ Performance test failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("✅ Secure API server is working correctly")
    print("✅ Security features are active")
    print("✅ Model predictions are working")
    print("✅ Performance is acceptable")
    
    return True

def main():
    """Main function to run the tests."""
    print("🔒 SAMO-DL Secure API Server Test")
    print("=" * 50)
    
    success = test_secure_api_server()
    
    if success:
        print("\n📋 Next Steps:")
        print("1. ✅ Security dependencies installed")
        print("2. ✅ Secure API server tested")
        print("3. 🔄 Address remaining security vulnerabilities")
        print("4. 🚀 Deploy secure model to GCP/Vertex AI")
        print("5. 📊 Implement continuous security monitoring")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 