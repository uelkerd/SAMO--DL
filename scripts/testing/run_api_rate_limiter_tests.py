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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

if __name__ == "__main__":
    logging.info("🧪 Running API Rate Limiter Tests...")
    
    # Set PYTHONPATH to include src directory
    os.environ['PYTHONPATH'] = str(project_root / "src") + ":" + os.environ.get('PYTHONPATH', '')
    
    test_file = project_root / "tests" / "unit" / "test_api_rate_limiter.py"
    
    logging.info(f"🔍 Looking for test file: {test_file}")
    logging.info(f"🔍 Test file exists: {test_file.exists()}")
    logging.info(f"🔍 Current working directory: {os.getcwd()}")
    logging.info(f"🔍 PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

    if not test_file.exists():
        logging.error(f"❌ Test file not found: {test_file}")
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
        
        logging.info(f"🔍 Running pytest with args: {args}")
        result = pytest.main(args)

        if result == 0:
            logging.info("✅ API Rate Limiter tests passed!")
        else:
            logging.error(f"❌ API Rate Limiter tests failed with exit code: {result}")

        sys.exit(result)

    except Exception as e:
        logging.error(f"❌ Exception during test execution: {e}")
        sys.exit(1)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_config)
