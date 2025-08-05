#!/usr/bin/env python3
"""
Run API Rate Limiter Tests

This script explicitly runs the API rate limiter tests to ensure they're discovered
and included in test coverage metrics.

Usage:
    python scripts/run_api_rate_limiter_tests.py
"""

from pathlib import Path
import contextlib
import logging
import os
import pytest
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))




"""
Run API Rate Limiter Tests

This script explicitly runs the API rate limiter tests to ensure they're discovered
and included in test coverage metrics.

Usage:
    python scripts/run_api_rate_limiter_tests.py
"""

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

if __name__ == "__main__":
    logging.info("🧪 Running API Rate Limiter Tests...")

    test_file = project_root / "tests" / "unit" / "test_api_rate_limiter.py"

    if not test_file.exists():
        logging.info(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""[pytest]
addopts = --cov=src.api_rate_limiter --cov-report=term-missing --cov-fail-under=5 -v --tb=short
""")
        temp_config = f.name

    try:
        args = [
            str(test_file),
            f"--config-file={temp_config}",
        ]

        result = pytest.main(args)

        if result == 0:
            logging.info("✅ API Rate Limiter tests passed!")
        else:
            logging.info(f"❌ API Rate Limiter tests failed with exit code: {result}")

        sys.exit(result)

    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_config)
