#!/usr/bin/env python3
"""
Model Loading Health Check
Check if the model is loading properly in the container.
"""

import requests
import json
from test_config import create_api_client, create_test_config


def check_model_healthbase_url=None:
    """Check model health status"""
    config = create_test_config()
    if base_url:
        config.base_url = base_url.rstrip'/'
    client = create_api_client()
    
    print"🔍 Model Health Check"
    print"=" * 30
    printf"Testing URL: {config.base_url}"
    
    # Test health endpoint
    try:
        data = client.get"/"
        print(f"✅ Health: {data.get'status'}")
    except requests.exceptions.RequestException as e:
        printf"❌ Health check error: {e}"
        return False

    # Test emotions from main endpoint
    try:
        data = client.get"/"
        emotions_count = data.get'emotions_supported', 0
        printf"✅ Emotions: {emotions_count} emotions available"
    except requests.exceptions.RequestException as e:
        printf"❌ Emotions check error: {e}"
        return False

    # Test prediction endpoint
    try:
        payload = {"text": "I am happy"}
        data = client.post"/predict", payload
        
        # Handle confidence formatting with null checks
        primary_emotion = data.get'primary_emotion', {}
        emotion = primary_emotion.get'emotion', 'Unknown'
        confidence = primary_emotion.get'confidence'
        if confidence is not None:
            confidence_str = f"{confidence:.3f}"
        else:
            confidence_str = "N/A"
        
        print(f"✅ Prediction: {emotion} confidence: {confidence_str}")
        return True
        
    except requests.exceptions.RequestException as e:
        printf"❌ Prediction check error: {e}"
        return False
    except ValueError as e:
        printf"❌ Invalid response format: {e}"
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParserdescription="Check Model Health"
    parser.add_argument"--base-url", help="API base URL"
    args = parser.parse_args()
    
    success = check_model_healthargs.base_url
    exit0 if success else 1
