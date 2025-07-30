#!/usr/bin/env python3
"""
Run API Rate Limiter Tests

This script explicitly runs the API rate limiter tests to ensure they're discovered
and included in test coverage metrics.

Usage:
    python scripts/run_api_rate_limiter_tests.py
"""

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
        print("❌ Test file not found: {test_file}")
        sys.exit(1)

    # Run pytest on the specific test file with coverage and threshold
    args = [
        str(test_file),
        "--cov=src.api_rate_limiter",
        "--cov-report=term-missing",
        "--cov-fail-under=5",
        "-v",
    ]

    # Run the tests
    result = pytest.main(args)

    # Print results
    if result == 0:
        print("✅ API Rate Limiter tests passed!")
    else:
        print("❌ API Rate Limiter tests failed with exit code: {result}")

    sys.exit(result)
