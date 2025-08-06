#!/usr/bin/env python3
"""
Model Loading Health Check
Check if the model is loading properly in the container.
"""

import requests
import os
import argparse
import sys

def check_model_health(base_url=None):
    """Check model health status"""
    if base_url is None:
        base_url = os.environ.get("API_BASE_URL", "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app")

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
    except requests.exceptions.RequestException as e:
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
    except requests.exceptions.RequestException as e:
        print(f"❌ Emotions check error: {e}")
        return False

    # Test prediction endpoint
    try:
        payload = {"text": "I am happy"}
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            confidence = data.get('confidence')
            # Handle None confidence values
            if confidence is not None:
                confidence_str = f"{confidence:.3f}"
            else:
                confidence_str = "N/A"
            print(f"✅ Prediction: {data.get('emotion')} (confidence: {confidence_str})")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction check error: {e}")
        return False

    print("\n✅ All health checks passed!")
    return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Check model health status")
    parser.add_argument("--base-url", help="Base URL for the API")
    args = parser.parse_args()

    success = check_model_health(args.base_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
