#!/usr/bin/env python3
"""
Test Model Status Endpoint
Get detailed information about model loading status and any errors.
"""

import requests
import json

def test_model_status():
    """Test the model status endpoint"""
    base_url = "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app"
    
    print("🔍 Testing Model Status")
    print("=" * 40)
    
    # Test health endpoint first
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data.get('status')}")
            print(f"   📊 Version: {data.get('version')}")
            print(f"   🔒 Security: {data.get('security')}")
        else:
            print(f"   ❌ Health failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health error: {e}")
    
    # Test emotions endpoint
    print("\n2. Testing emotions endpoint...")
    try:
        response = requests.get(f"{base_url}/emotions")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Emotions: {data}")
        else:
            print(f"   ❌ Emotions failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Emotions error: {e}")
    
    # Test model status endpoint (requires API key)
    print("\n3. Testing model status endpoint...")
    print("   ⚠️  This requires an API key - will likely fail")
    try:
        response = requests.get(f"{base_url}/model_status")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model Status: {data}")
        elif response.status_code == 401:
            print("   🔐 Unauthorized - API key required")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Model status error: {e}")
    
    # Test a simple prediction to see the actual error
    print("\n4. Testing prediction endpoint...")
    try:
        payload = {"text": "I am happy"}
        response = requests.post(f"{base_url}/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction successful: {data}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")

if __name__ == "__main__":
    test_model_status() 