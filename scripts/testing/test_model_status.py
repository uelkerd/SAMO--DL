#!/usr/bin/env python3
"""
Test Model Status Endpoint
Get detailed information about model loading status and any errors.
"""

import requests
import json
import argparse
from config import TestConfig, APIClient
import os

def test_model_status(base_url=None, include_auth=True):
    """Test the model status endpoint"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")
    
    print(f"🔍 Testing Model Status at {base_url}")
    print("=" * 40)
    
    # Use centralized API client
    client = APIClient(base_url, include_auth)
    
    # Test health endpoint first
    print("1. Testing health endpoint...")
    try:
        response = client.get("/")
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
        response = client.get("/emotions")
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
        response = client.get("/model_status")
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
        response = client.post("/predict", payload)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction successful: {data}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")

def main():
    """Main function with argument parsing."""
    config = TestConfig()
    parser = config.get_parser("Test model status endpoint")
    args = parser.parse_args()
    
    test_model_status(args.base_url, not args.no_auth)

if __name__ == "__main__":
    main() 