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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

if __name__ == "__main__":
    logger.info("🧪 Running API Rate Limiter Tests...")

    test_file = project_root / "tests" / "unit" / "test_api_rate_limiter.py"

    if not test_file.exists():
        logger.error(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    # Create a temporary pytest configuration to avoid conflicts with pyproject.toml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""[pytest]
addopts = --cov=src.api_rate_limiter --cov-report=term-missing --cov-fail-under=50 -v --tb=short
""")
        temp_config = f.name

    try:
        # Get the path to the test file
        args = [
            str(test_file),
            f"--config-file={temp_config}",
        ]

        # Run pytest with the temporary configuration
        result = pytest.main(args)

        # Run the tests
        if result == 0:
            logger.info("✅ API Rate Limiter tests passed!")
        else:
            logger.error(f"❌ API Rate Limiter tests failed with exit code: {result}")

        sys.exit(result)

    finally:
        # Clean up temporary file
        with contextlib.suppress(OSError):
            os.unlink(temp_config)
