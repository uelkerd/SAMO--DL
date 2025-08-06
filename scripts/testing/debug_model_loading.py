#!/usr/bin/env python3
"""
Debug Model Loading Issues
Comprehensive debugging script for Cloud Run deployment model loading problems.
"""

import os
import sys
import time
import requests
import json
import argparse
from config import TestConfig, APIClient

def debug_model_loading(base_url=None, include_auth=True):
    """Debug the model loading issues"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")
    
    print(f"🔍 Debugging model loading at: {base_url}")
    print("=" * 50)
    
    # Use centralized API client
    client = APIClient(base_url, include_auth)
    
    # Test model status with API key
    print("\n1. Testing model status with API key...")
    try:
        headers = {"X-API-Key": client.headers["X-API-Key"]}
        response = requests.get(f"{base_url}/model_status", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model Status: {json.dumps(data, indent=2)}")
        elif response.status_code == 401:
            print("   🔐 Unauthorized - API key mismatch")
            print(f"   Response: {response.text}")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Model status error: {e}")
    
    # Test security status
    print("\n2. Testing security status...")
    try:
        headers = {"X-API-Key": client.headers["X-API-Key"]}
        response = requests.get(f"{base_url}/security_status", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Security Status: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Security status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Security status error: {e}")
    
    # Test prediction with detailed error analysis - FIXED: Added API key
    print("\n3. Testing prediction with error analysis...")
    try:
        payload = {"text": "I am happy"}
        response = client.post("/predict", payload)
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction successful: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Prediction failed")
            print(f"   Response Text: {response.text}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"   Error Data: {json.dumps(error_data, indent=2)}")
            except:
                print(f"   Raw Response: {response.text}")
                
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Test batch prediction - FIXED: Added API key
    print("\n4. Testing batch prediction...")
    try:
        payload = {"texts": ["I am happy", "I am sad", "I am excited"]}
        response = client.post("/predict_batch", payload)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Batch prediction successful: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Batch prediction failed")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")
    
    # Test with different input formats - FIXED: Added API key
    print("\n5. Testing different input formats...")
    test_cases = [
        {"text": "I am happy"},
        {"text": "This is a test"},
        {"text": "I feel great"},
        {"text": ""},  # Empty text
        {"invalid": "field"},  # Invalid payload
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case}")
        try:
            response = client.post("/predict", test_case)
            print(f"     Status: {response.status_code}")
            if response.status_code != 200:
                print(f"     Error: {response.text}")
        except Exception as e:
            print(f"     Exception: {e}")

def main():
    """Main function with argument parsing."""
    config = TestConfig()
    parser = config.get_parser("Debug model loading issues")
    args = parser.parse_args()
    
    debug_model_loading(args.base_url, not args.no_auth)

if __name__ == "__main__":
    main() 