#!/usr/bin/env python3
"""
Model Loading Health Check
Check if the model is loading properly in the container.
"""

import os
import sys
import time
import requests
import json

def check_model_health():
    """Check model health status"""
    base_url = "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app"
    
    print("🔍 Model Health Check")
    print("=" * 30)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status')}")
        else:
            print(f"❌ Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test emotions endpoint
    try:
        response = requests.get(f"{base_url}/emotions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Emotions: {data.get('count')} emotions available")
        else:
            print(f"❌ Emotions failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Emotions check error: {e}")
        return False
    
    # Test prediction endpoint
    try:
        payload = {"text": "I am happy"}
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction: {data.get('emotion')} (confidence: {data.get('confidence'):.3f})")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

if __name__ == "__main__":
    success = check_model_health()
    sys.exit(0 if success else 1)
