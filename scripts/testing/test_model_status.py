#!/usr/bin/env python3
"""
Test Model Status Endpoint
Get detailed information about model loading status and any errors.
"""

import requests
import os
import argparse

def test_health_endpoint(base_url):
    """Test health endpoint"""
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Health: {data.get('status')}")
        print(f"   📊 Version: {data.get('version')}")
        print(f"   🔒 Security: {data.get('security')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Health error: {e}")
        return False

def test_emotions_endpoint(base_url):
    """Test emotions endpoint"""
    print("\n2. Testing emotions endpoint...")
    try:
        response = requests.get(f"{base_url}/emotions", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Emotions: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Emotions error: {e}")
        return False

def test_model_status_endpoint(base_url):
    """Test model status endpoint"""
    print("\n3. Testing model status endpoint...")
    print("   ⚠️  This requires an API key - will likely fail")
    try:
        response = requests.get(f"{base_url}/model_status", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Model Status: {data}")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("   🔐 Unauthorized - API key required")
        else:
            print(f"   ❌ Model status failed: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Model status error: {e}")
        return False

def test_prediction_endpoint(base_url):
    """Test prediction endpoint"""
    print("\n4. Testing prediction endpoint...")
    try:
        payload = {"text": "I am happy"}
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Prediction successful: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Prediction error: {e}")
        return False

def test_model_status(base_url=None):
    """Test the model status endpoint"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")
    
    print("🔍 Testing Model Status")
    print("=" * 40)
    
    # Run all tests
    health_success = test_health_endpoint(base_url)
    emotions_success = test_emotions_endpoint(base_url)
    model_status_success = test_model_status_endpoint(base_url)
    prediction_success = test_prediction_endpoint(base_url)
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Health: {'✅' if health_success else '❌'}")
    print(f"   Emotions: {'✅' if emotions_success else '❌'}")
    print(f"   Model Status: {'✅' if model_status_success else '❌'}")
    print(f"   Prediction: {'✅' if prediction_success else '❌'}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Test model status endpoint")
    parser.add_argument("--base-url", help="Base URL for the API")
    args = parser.parse_args()
    
    test_model_status(args.base_url)

if __name__ == "__main__":
    main() 