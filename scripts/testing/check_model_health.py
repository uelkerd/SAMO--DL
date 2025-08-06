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
import argparse
from config import TestConfig, APIClient

def check_model_health(base_url=None, include_auth=True):
    """Check model health status"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")
    
    print(f"🔍 Model Health Check for {base_url}")
    print("=" * 30)
    
    # Use centralized API client
    client = APIClient(base_url, include_auth)
    
    # Test health endpoint
    try:
        response = client.get("/")
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
        response = client.get("/emotions")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Emotions: {data.get('count')} emotions available")
        else:
            print(f"❌ Emotions failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Emotions check error: {e}")
        return False
    
    # Test prediction endpoint - FIXED: Handle None confidence values
    try:
        payload = {"text": "I am happy"}
        response = client.post("/predict", payload)
        if response.status_code == 200:
            data = response.json()
            emotion = data.get('emotion', 'unknown')
            confidence = data.get('confidence')
            
            # FIXED: Handle None confidence values to prevent TypeError
            if confidence is not None:
                confidence_str = f"{confidence:.3f}"
            else:
                confidence_str = "N/A"
            
            print(f"✅ Prediction: {emotion} (confidence: {confidence_str})")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Main function with argument parsing."""
    config = TestConfig()
    parser = config.get_parser("Check model health status")
    args = parser.parse_args()
    
    success = check_model_health(args.base_url, not args.no_auth)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
