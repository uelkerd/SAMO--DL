#!/usr/bin/env python3
""""
Debug Model Loading Issues
Get detailed information about why the model is not loading properly.
""""

import argparse
import json
import requests
import time
from test_config import create_api_client, create_test_config


def debug_model_loading():
    """Debug the model loading issues"""
    config = create_test_config()
    client = create_api_client()

    print(" Debugging Model Loading Issues")
    print("=" * 50)
    print(f"Testing URL: {config.base_url}")
    print(f"API Key: {config.api_key[:20]}...")

    # Test model status with API key
    print("\n1. Testing model status with API key...")
    try:
        data = client.get("/model_status")
        print(f"    Model Status: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        if "401" in str(e):
            print("   🔐 Unauthorized - API key mismatch")
        else:
            print(f"   ❌ Model status error: {e}")

    # Test security status
    print("\n2. Testing security status...")
    try:
        data = client.get("/security_status")
        print(f"    Security Status: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Security status error: {e}")

    # Test prediction with detailed error analysis
    print("\n3. Testing prediction with error analysis...")
    try:
        payload = {"text": "I am happy"}
        data = client.post("/predict", payload)
        print(f"    Prediction successful: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Prediction error: {e}")
    except ValueError as e:
        print(f"   ❌ Invalid response format: {e}")

    # Test batch prediction
    print("\n4. Testing batch prediction...")
    try:
        payload = {"texts": ["I am happy", "I am sad", "I am excited"]}
        data = client.post("/predict_batch", payload)
        print(f"    Batch prediction successful: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Batch prediction error: {e}")
    except ValueError as e:
        print(f"   ❌ Invalid response format: {e}")

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
            data = client.post("/predict", test_case)
            print("      Success: {data.get("emotion', 'Unknown')}")"
        except requests.exceptions.RequestException as e:
            print(f"     ❌ Request failed: {e}")
        except ValueError as e:
            print(f"     ❌ Invalid response: {e}")


    if __name__ == "__main__":
    debug_model_loading()
