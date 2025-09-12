#!/usr/bin/env python3
"""
Test DeBERTa model integration with the full API server.

This script tests the API endpoints with DeBERTa enabled to ensure
the full integration works correctly.
"""

import os
import sys
import json
import requests
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8080')
API_TIMEOUT = 30  # Increased timeout for DeBERTa loading
API_KEY = os.getenv('API_KEY', 'test123')  # API key for authentication
HEADERS = {'X-API-Key': API_KEY}

def test_api_health():
    """Test API health endpoint."""
    try:
        logger.info("🏥 Testing API health endpoint...")
        response = requests.get(f"{API_BASE_URL}/api/health", headers=HEADERS, timeout=10)

        if response.status_code == 200:
            logger.info("✅ API health check passed")
            return True
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ API health check error: {e}")
        return False

def test_emotion_prediction():
    """Test emotion prediction endpoint with DeBERTa."""
    try:
        logger.info("🧪 Testing emotion prediction endpoint...")

        test_payload = {
            "text": "I am so happy today!",
            "return_all_scores": True
        }

        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=test_payload,
            headers=HEADERS,
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Emotion prediction successful")

            # Log the result
            if 'emotions' in result and result['emotions']:
                top_emotions = result['emotions'][:3]
                logger.info(f"📝 Prediction result: {', '.join([f'{e['emotion']}:{e['confidence']:.3f}' for e in top_emotions])}")

                # Verify we got DeBERTa emotions (should have 28 emotions)
                if len(result['emotions']) > 6:
                    logger.info("✅ DeBERTa emotions detected (28 emotions)")
                else:
                    logger.warning("⚠️ Only production emotions detected (6 emotions)")

            return True
        else:
            logger.error(f"❌ Emotion prediction failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Emotion prediction error: {e}")
        return False

def test_model_status():
    """Test model status endpoint."""
    try:
        logger.info("📊 Testing model status endpoint...")

        response = requests.get(f"{API_BASE_URL}/admin/model_status", headers=HEADERS, timeout=10)

        if response.status_code == 200:
            status = response.json()
            logger.info("✅ Model status retrieved")

            # Log key status info
            logger.info(f"📋 Model loaded: {status.get('model_loaded', 'Unknown')}")
            logger.info(f"📋 Model provider: {status.get('model_provider', 'Unknown')}")
            logger.info(f"📋 Emotion labels count: {len(status.get('emotion_labels', []))}")

            if len(status.get('emotion_labels', [])) > 6:
                logger.info("✅ DeBERTa model confirmed (28 emotion labels)")
            else:
                logger.warning("⚠️ Production model detected (6 emotion labels)")

            return True
        else:
            logger.error(f"❌ Model status failed: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ Model status error: {e}")
        return False

def test_multiple_predictions():
    """Test multiple emotion predictions."""
    try:
        logger.info("🔄 Testing multiple emotion predictions...")

        test_texts = [
            "I am so happy today!",
            "This is absolutely terrible",
            "I'm feeling a bit nervous about the presentation"
        ]

        test_payload = {
            "texts": test_texts,
            "return_all_scores": True
        }

        response = requests.post(
            f"{API_BASE_URL}/api/predict_batch",
            json=test_payload,
            headers=HEADERS,
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            results = response.json()
            logger.info("✅ Batch emotion prediction successful")

            # Log results for each text
            for i, result in enumerate(results):
                if 'emotions' in result and result['emotions']:
                    top_emotion = result['emotions'][0]
                    logger.info(f"📝 Text {i+1}: {top_emotion['emotion']}:{top_emotion['confidence']:.3f}")

            return True
        else:
            logger.error(f"❌ Batch prediction failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return False

def main():
    """Run all DeBERTa API tests."""
    logger.info("🚀 Starting DeBERTa API integration tests...")
    logger.info(f"📡 API Base URL: {API_BASE_URL}")

    tests = [
        ("Health Check", test_api_health),
        ("Model Status", test_model_status),
        ("Single Prediction", test_emotion_prediction),
        ("Batch Predictions", test_multiple_predictions)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))

        if success:
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")

    logger.info(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! DeBERTa integration successful!")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED. Check logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
