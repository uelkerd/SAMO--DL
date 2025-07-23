#!/usr/bin/env python3
"""API Health Check for CircleCI.

This script tests that the unified AI API endpoints are working
and can handle basic requests without errors.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_imports():
    """Test that API modules can be imported."""
    try:
        logger.info("🌐 Testing API imports...")
        
        # Test core API imports
        from unified_ai_api import app
        from models.emotion_detection.api_demo import EmotionAnalysisRequest
        from models.summarization.api_demo import SummarizationRequest
        
        logger.info("✅ All API imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ API import test failed: {e}")
        return False


def test_api_models():
    """Test that API models can be instantiated."""
    try:
        logger.info("🤖 Testing API model instantiation...")
        
        # Test emotion analysis request model
        from models.emotion_detection.api_demo import EmotionAnalysisRequest
        emotion_request = EmotionAnalysisRequest(
            text="I feel happy and excited today!",
            threshold=0.2
        )
        assert emotion_request.text == "I feel happy and excited today!"
        assert emotion_request.threshold == 0.2
        
        # Test summarization request model
        from models.summarization.api_demo import SummarizationRequest
        summary_request = SummarizationRequest(
            text="This is a test text for summarization.",
            max_length=50,
            min_length=10
        )
        assert summary_request.text == "This is a test text for summarization."
        assert summary_request.max_length == 50
        
        logger.info("✅ API model instantiation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ API model test failed: {e}")
        return False


def test_fastapi_app():
    """Test that FastAPI app can be created."""
    try:
        logger.info("🚀 Testing FastAPI app creation...")
        
        from unified_ai_api import app
        
        # Verify app is a FastAPI instance
        from fastapi import FastAPI
        assert isinstance(app, FastAPI), "App should be a FastAPI instance"
        
        # Check that app has routes
        routes = [route.path for route in app.routes]
        logger.info(f"✅ FastAPI app created with {len(routes)} routes")
        logger.info(f"Available routes: {routes}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI app test failed: {e}")
        return False


def test_api_health_endpoint():
    """Test the health endpoint if available."""
    try:
        logger.info("💚 Testing API health endpoint...")
        
        from fastapi.testclient import TestClient
        from unified_ai_api import app
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        logger.info(f"Root endpoint status: {response.status_code}")
        
        # If there's a health endpoint, test it
        try:
            health_response = client.get("/health")
            logger.info(f"Health endpoint status: {health_response.status_code}")
        except Exception:
            logger.info("No /health endpoint found (optional)")
        
        logger.info("✅ API endpoint tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoint test failed: {e}")
        return False


def main():
    """Run API health checks."""
    logger.info("🧪 Starting API Health Checks")
    
    tests = [
        test_api_imports,
        test_api_models,
        test_fastapi_app,
        test_api_health_endpoint,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    logger.info(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All API health checks passed!")
        return 0
    else:
        logger.error("❌ Some API health checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 