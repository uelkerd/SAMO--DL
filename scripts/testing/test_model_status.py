#!/usr/bin/env python3
"""
Test Model Status Endpoint
Get detailed information about model loading status and any errors.
"""

import requests
import argparse
from test_config import create_api_client, create_test_config


def test_health_endpoint(client):
    """Test the health endpoint"""
    print("1. Testing health endpoint...")
    try:
        data = client.get("/")
        print(f"   ✅ Health: {data.get('status')}")
        print(f"   📊 Version: {data.get('version')}")
        print(f"   🔒 Security: {data.get('security')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Health failed: {e}")
        return False


def test_emotions_endpoint(client):
    """Test the emotions endpoint"""
    print("\n2. Testing emotions endpoint...")
    try:
        data = client.get("/emotions")
        print(f"   ✅ Emotions: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Emotions failed: {e}")
        return False


def test_model_status_endpoint(client):
    """Test the model status endpoint"""
    print("\n3. Testing model status endpoint...")
    print("   ⚠️  This requires an API key - will likely fail")
    try:
        data = client.get("/model_status")
        print(f"   ✅ Model Status: {data}")
        return True
    except requests.exceptions.RequestException as e:
        if "401" in str(e):
            print("   🔐 Unauthorized - API key required")
        else:
            print(f"   ❌ Model status failed: {e}")
        return False


def test_prediction_endpoint(client):
    """Test the prediction endpoint"""
    print("\n4. Testing prediction endpoint...")
    try:
        payload = {"text": "I am happy"}
        data = client.post("/predict", payload)
        print(f"   ✅ Prediction successful: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Prediction failed: {e}")
        return False


def test_model_status(base_url=None):
    """Test the model status endpoint"""
    config = create_test_config()
    client = create_api_client()
    
    print("🔍 Testing Model Status")
    print("=" * 40)
    print(f"Testing URL: {base_url or config.base_url}")
    
    # Run all tests
    health_success = test_health_endpoint(client)
    emotions_success = test_emotions_endpoint(client)
    model_status_success = test_model_status_endpoint(client)
    prediction_success = test_prediction_endpoint(client)
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Health: {'✅' if health_success else '❌'}")
    print(f"   Emotions: {'✅' if emotions_success else '❌'}")
    print(f"   Model Status: {'✅' if model_status_success else '❌'}")
    print(f"   Prediction: {'✅' if prediction_success else '❌'}")
    
    return health_success and emotions_success and prediction_success


def main():
    """Main function with CLI argument support"""
    parser = argparse.ArgumentParser(description="Test Model Status Endpoint")
    parser.add_argument("--base-url", help="API base URL")
    args = parser.parse_args()
    
    success = test_model_status(args.base_url)
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 