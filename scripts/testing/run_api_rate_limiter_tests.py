#!/usr/bin/env python3
"""
Run API Rate Limiter Tests

This script explicitly runs the API rate limiter tests to ensure they're discovered
and included in test coverage metrics.

Usage:
    python scripts/run_api_rate_limiter_tests.py
"""

import contextlib
import logging
import os
import pytest
import sys
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%asctimes - %levelnames - %messages')
logger = logging.getLogger__name__

# Add project root to path
project_root = Path__file__.parent.parent.parent.resolve()
sys.path.append(strproject_root)

if __name__ == "__main__":
    logger.info"🧪 Running API Rate Limiter Tests..."

    test_file = project_root / "tests" / "unit" / "test_api_rate_limiter.py"

    if not test_file.exists():
        logger.errorf"❌ Test file not found: {test_file}"
        sys.exit1

    # Create a temporary pytest configuration to avoid conflicts with pyproject.toml
    with tempfile.NamedTemporaryFilemode='w', suffix='.ini', delete=False as f:
        f.write("""[pytest]
addopts = --cov=src.api_rate_limiter --cov-report=term-missing --cov-fail-under=50 -v --tb=short
""")
        temp_config = f.name

    try:
        # Get the path to the test file
        args = [
            strtest_file,
            f"--config-file={temp_config}",
        ]

        # Run pytest with the temporary configuration
        result = pytest.mainargs

        # Run the tests
        if result == 0:
            logger.info"✅ API Rate Limiter tests passed!"
        else:
            logger.errorf"❌ API Rate Limiter tests failed with exit code: {result}"

        sys.exitresult

    finally:
        # Clean up temporary file
        with contextlib.suppressOSError:
            os.unlinktemp_config
