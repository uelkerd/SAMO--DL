#!/usr/bin/env python3
"""
Model Loading Health Check
Check if the model is loading properly in the container.
"""

import requests
import json
from test_config import create_api_client, create_test_config


def check_model_health():
    """Check model health status"""
    config = create_test_config()
    client = create_api_client()
    
    print("🔍 Model Health Check")
    print("=" * 30)
    print(f"Testing URL: {config.base_url}")
    
    # Test health endpoint
    try:
        data = client.get("/")
        print(f"✅ Health: {data.get('status')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test emotions endpoint
    try:
        data = client.get("/emotions")
        print(f"✅ Emotions: {data.get('count')} emotions available")
    except requests.exceptions.RequestException as e:
        print(f"❌ Emotions check error: {e}")
        return False

    # Test prediction endpoint
    try:
        payload = {"text": "I am happy"}
        data = client.post("/predict", payload)
        
        # Handle confidence formatting with null checks
        confidence = data.get('confidence')
        if confidence is not None:
            confidence_str = f"{confidence:.3f}"
        else:
            confidence_str = "N/A"
        
        print(f"✅ Prediction: {data.get('emotion')} (confidence: {confidence_str})")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction check error: {e}")
        return False
    except ValueError as e:
        print(f"❌ Invalid response format: {e}")
        return False


if __name__ == "__main__":
    success = check_model_health()
    exit(0 if success else 1)
