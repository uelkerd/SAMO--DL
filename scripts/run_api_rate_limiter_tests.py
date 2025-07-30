import sys

#!/usr/bin/env python3
"""
Run API Rate Limiter Tests

This script explicitly runs the API rate limiter tests to ensure they're discovered
and included in test coverage metrics.

Usage:
    python scripts/run_api_rate_limiter_tests.py
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

if __name__ == "__main__":
    print("🧪 Running API Rate Limiter Tests...")

    # Get the path to the test file
    test_file = project_root / "tests" / "unit" / "test_api_rate_limiter.py"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    # Create a temporary pytest configuration to avoid conflicts with pyproject.toml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""[pytest]
addopts = --cov=src.api_rate_limiter --cov-report=term-missing --cov-fail-under=5 -v --tb=short
""")
        temp_config = f.name

    try:
        # Run pytest with the temporary configuration
        args = [
            str(test_file),
            f"--config-file={temp_config}",
        ]

        # Run the tests
        result = pytest.main(args)

        # Print results
        if result == 0:
            print("✅ API Rate Limiter tests passed!")
        else:
            print(f"❌ API Rate Limiter tests failed with exit code: {result}")

        sys.exit(result)
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_config)
        except OSError:
            pass
