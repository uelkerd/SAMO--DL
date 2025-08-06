#!/usr/bin/env python3
"""
Debug Model Loading Issues
Get detailed information about why the model is not loading properly.
"""

import requests
import json
import os
import secrets
import argparse

def generate_api_key():
    """Generate an API key securely or use environment variable"""
    api_key = os.environ.get("API_KEY")
    if api_key:
        return api_key
    # Fallback: generate a secure random key
    return "samo-admin-key-" + secrets.token_urlsafe(32)

def debug_model_loading(base_url=None):
    """Debug the model loading issues"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")

    print("🔍 Debugging Model Loading Issues")
    print("=" * 50)

    # Generate API key
    api_key = generate_api_key()
    print(f"🔑 Generated API Key: {api_key}")

    # Set up headers with API key
    headers = {"X-API-Key": api_key}

    # Test model status with API key
    print("\n1. Testing model status with API key...")
    try:
        response = requests.get(f"{base_url}/model_status", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model Status: {json.dumps(data, indent=2)}")
        elif response.status_code == 401:
            print("   🔐 Unauthorized - API key mismatch")
            print(f"   Response: {response.text}")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Model status error: {e}")

    # Test security status
    print("\n2. Testing security status...")
    try:
        response = requests.get(f"{base_url}/security_status", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Security Status: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Security status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Security status error: {e}")

    # Test prediction with detailed error analysis
    print("\n3. Testing prediction with error analysis...")
    try:
        payload = {"text": "I am happy"}
        response = requests.post(f"{base_url}/predict", json=payload, headers=headers, timeout=30)
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction successful: {json.dumps(data, indent=2)}")
        else:
            print("   ❌ Prediction failed")
            print(f"   Response Text: {response.text}")

            # Try to parse error response
            try:
                error_data = response.json()
                print(f"   Error Data: {json.dumps(error_data, indent=2)}")
            except json.JSONDecodeError:
                print(f"   Raw Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Prediction error: {e}")

    # Test batch prediction
    print("\n4. Testing batch prediction...")
    try:
        payload = {"texts": ["I am happy", "I am sad", "I am excited"]}
        response = requests.post(f"{base_url}/predict_batch", json=payload, headers=headers, timeout=30)
        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Batch prediction successful: {json.dumps(data, indent=2)}")
        else:
            print("   ❌ Batch prediction failed")
            print(f"   Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Batch prediction error: {e}")

    # Test with different input formats
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
            response = requests.post(f"{base_url}/predict", json=test_case, headers=headers, timeout=30)
            print(f"     Status: {response.status_code}")
            if response.status_code != 200:
                print(f"     Error: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"     Exception: {e}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Debug model loading issues")
    parser.add_argument("--base-url", help="Base URL for the API")
    args = parser.parse_args()

    debug_model_loading(args.base_url)

if __name__ == "__main__":
    main() 
