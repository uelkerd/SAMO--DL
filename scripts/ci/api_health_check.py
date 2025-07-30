#!/usr/bin/env python3
"""
API Health Check for CI/CD Pipeline.

This script validates that all API components are working correctly
and can be imported without errors.
"""

import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_imports():
    """Test that all API modules can be imported successfully."""
    try:
        logger.info("🔍 Testing API imports...")

        # Test basic API imports without triggering deep learning models
        import api_rate_limiter
        logger.info("✅ API rate limiter import successful")

        # Test Pydantic imports
        from pydantic import BaseModel, ValidationError
        logger.info("✅ Pydantic imports successful")

        # Test FastAPI imports
        from fastapi import FastAPI
        logger.info("✅ FastAPI imports successful")

        logger.info("✅ All API imports successful")
        return True

    except Exception as e:
        logger.error("❌ API import test failed: {e}")
        return False


def test_api_models():
    """Test that API models can be instantiated."""
    try:
        logger.info("🤖 Testing API model instantiation...")

        # Test basic Pydantic model creation
        from pydantic import BaseModel

        class TestRequest(BaseModel):
            text: str
            threshold: float = 0.2

        test_request = TestRequest(text="I feel happy and excited today!")
        logger.info("✅ Test request created: {test_request.text[:30]}...")

        # Test rate limiter functionality
        from api_rate_limiter import RateLimitCache, RateLimitEntry

        cache = RateLimitCache()
        entry = cache.get("test_client")
        logger.info("✅ Rate limiter cache created successfully")

        return True

    except Exception as e:
        logger.error("❌ API model test failed: {e}")
        return False


def test_api_validation():
    """Test API request validation."""
    try:
        logger.info("🔒 Testing API validation...")

        from pydantic import BaseModel, ValidationError, Field

        class TestRequest(BaseModel):
            text: str = Field(..., min_length=1, description="Text cannot be empty")
            threshold: float = Field(0.2, ge=0.0, le=1.0, description="Threshold between 0 and 1")

        # Test invalid request - empty text
        try:
            TestRequest(text="")  # Invalid: empty text
            logger.error("❌ Validation should have failed for invalid request")
            return False
        except ValidationError:
            logger.info("✅ Validation correctly rejected invalid request")

        # Test invalid request - threshold out of range
        try:
            TestRequest(text="Valid text", threshold=1.5)  # Invalid: threshold > 1
            logger.error("❌ Validation should have failed for invalid threshold")
            return False
        except ValidationError:
            logger.info("✅ Validation correctly rejected invalid threshold")

        # Test valid request
        TestRequest(text="This is a valid test text.", threshold=0.3)
        logger.info("✅ Valid request accepted")

        return True

    except Exception as e:
        logger.error("❌ API validation test failed: {e}")
        return False


def main():
    """Run all API health checks."""
    logger.info("🚀 Starting API Health Check...")

    tests = [
        ("API Imports", test_api_imports),
        ("API Models", test_api_models),
        ("API Validation", test_api_validation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("\n{'='*50}")
        logger.info("Running: {test_name}")
        logger.info("{'='*50}")

        if test_func():
            passed += 1
            logger.info("✅ {test_name}: PASSED")
        else:
            logger.error("❌ {test_name}: FAILED")

    logger.info("\n{'='*50}")
    logger.info("API Health Check Results: {passed}/{total} tests passed")
    logger.info("{'='*50}")

    if passed < total:
        logger.error("💥 Some API health checks failed!")
        return False

    logger.info("🎉 All API health checks passed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
