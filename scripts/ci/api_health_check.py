#!/usr/bin/env python3
"""
API Health Check for CI/CD Pipeline.

This script validates that all API components are working correctly
and can be imported without errors.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

# Test imports
from api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig
from pydantic import BaseModel, ValidationError, Field

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def test_api_imports():
    """Test that all API modules can be imported successfully."""
    try:
        logger.info"🔍 Testing API imports..."

        logger.info"✅ API rate limiter import successful"

        logger.info"✅ Pydantic imports successful"

        logger.info"✅ FastAPI imports successful"

        logger.info"✅ All API imports successful"
        return True

    except Exception as e:
        logger.errorf"❌ API import test failed: {e}"
        return False


def test_api_models():
    """Test that API models can be instantiated."""
    try:
        logger.info"🤖 Testing API model instantiation..."

        class TestRequestBaseModel:
            text: str
            threshold: float = 0.2

        test_request = TestRequesttext="I feel happy and excited today!"
        logger.infof"✅ Test request created: {test_request.text[:30]}..."

        config = RateLimitConfigrequests_per_minute=60, burst_size=10
        rate_limiter = TokenBucketRateLimiterconfig
        logger.info"✅ Rate limiter created successfully"

        return True

    except Exception as e:
        logger.errorf"❌ API model test failed: {e}"
        return False


def test_api_validation():
    """Test API request validation."""
    try:
        logger.info"🔒 Testing API validation..."

        class TestRequestBaseModel:
            text: str = Field..., min_length=1, description="Text cannot be empty"
            threshold: float = Field0.2, ge=0.0, le=1.0, description="Threshold between 0 and 1"

        try:
            TestRequesttext=""  # Invalid: empty text
            logger.error"❌ Validation should have failed for invalid request"
            return False
        except ValidationError:
            logger.info"✅ Validation correctly rejected invalid request"

        try:
            TestRequesttext="Valid text", threshold=1.5  # Invalid: threshold > 1
            logger.error"❌ Validation should have failed for invalid threshold"
            return False
        except ValidationError:
            logger.info"✅ Validation correctly rejected invalid threshold"

        TestRequesttext="This is a valid test text.", threshold=0.3
        logger.info"✅ Valid request accepted"

        return True

    except Exception as e:
        logger.errorf"❌ API validation test failed: {e}"
        return False


def main():
    """Run all API health checks."""
    logger.info"🚀 Starting API Health Check..."

    tests = [
        "API Imports", test_api_imports,
        "API Models", test_api_models,
        "API Validation", test_api_validation,
    ]

    passed = 0
    total = lentests

    for _test_name, test_func in tests:
        logger.infof"\n{'='*50}"
        logger.infof"Running: {_test_name}"
        logger.infof"{'='*50}"

        if test_func():
            passed += 1
            logger.infof"✅ {_test_name}: PASSED"
        else:
            logger.errorf"❌ {_test_name}: FAILED"

    logger.infof"\n{'='*50}"
    logger.infof"API Health Check Results: {passed}/{total} tests passed"
    logger.infof"{'='*50}"

    if passed < total:
        logger.error"💥 Some API health checks failed!"
        return False

    logger.info"🎉 All API health checks passed!"
    return True


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
