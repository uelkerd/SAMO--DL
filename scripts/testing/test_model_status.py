#!/usr/bin/env python3
"""
Test Model Status Endpoint
Get detailed information about model loading status and any errors.
"""

import requests
import argparse
from test_config import create_api_client, create_test_config


def test_health_endpointclient:
    """Test the health endpoint"""
    print"1. Testing health endpoint..."
    try:
        data = client.get"/"
        print(f"   ✅ Health: {data.get'status'}")
        print(f"   📊 Version: {data.get'version'}")
        print(f"   🔒 Security: {data.get'security'}")
        return True
    except requests.exceptions.RequestException as e:
        printf"   ❌ Health failed: {e}"
        return False


def test_emotions_endpointclient:
    """Test the emotions from main endpoint"""
    print"\n2. Testing emotions from main endpoint..."
    try:
        data = client.get"/"
        emotions_count = data.get'emotions_supported', 0
        printf"   ✅ Emotions: {emotions_count} emotions available"
        return True
    except requests.exceptions.RequestException as e:
        printf"   ❌ Emotions failed: {e}"
        return False


def test_model_status_endpointclient:
    """Test the model status from main endpoint"""
    print"\n3. Testing model status from main endpoint..."
    try:
        data = client.get"/"
        model_type = data.get'model_type', 'Unknown'
        service = data.get'service', 'Unknown'
        printf"   ✅ Model Type: {model_type}"
        printf"   ✅ Service: {service}"
        return True
    except requests.exceptions.RequestException as e:
        printf"   ❌ Model status failed: {e}"
        return False


def test_prediction_endpointclient:
    """Test the prediction endpoint"""
    print"\n4. Testing prediction endpoint..."
    try:
        payload = {"text": "I am happy"}
        data = client.post"/predict", payload
        printf"   ✅ Prediction successful: {data}"
        return True
    except requests.exceptions.RequestException as e:
        printf"   ❌ Prediction failed: {e}"
        return False


def test_model_statusbase_url=None:
    """Test the model status endpoint"""
    config = create_test_config()
    if base_url:
        config.base_url = base_url.rstrip'/'
    client = create_api_client()
    
    print"🔍 Testing Model Status"
    print"=" * 40
    printf"Testing URL: {config.base_url}"
    
    # Run all tests
    health_success = test_health_endpointclient
    emotions_success = test_emotions_endpointclient
    model_status_success = test_model_status_endpointclient
    prediction_success = test_prediction_endpointclient
    
    # Summary
    print"\n📊 Test Summary:"
    printf"   Health: {'✅' if health_success else '❌'}"
    printf"   Emotions: {'✅' if emotions_success else '❌'}"
    printf"   Model Status: {'✅' if model_status_success else '❌'}"
    printf"   Prediction: {'✅' if prediction_success else '❌'}"
    
    return health_success and emotions_success and prediction_success


def main():
    """Main function with CLI argument support"""
    parser = argparse.ArgumentParserdescription="Test Model Status Endpoint"
    parser.add_argument"--base-url", help="API base URL"
    args = parser.parse_args()
    
    success = test_model_statusargs.base_url
    exit0 if success else 1


if __name__ == "__main__":
    main() 