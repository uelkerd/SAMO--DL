#!/usr/bin/env python3
"""
Simple Test Health Check Script

This script provides basic test health information without complex analysis.
Keeps scope small and focused on essential metrics.
"""

import os
import subprocess
import sys
from pathlib import Path


def count_test_files():
    """Count total test files in the project."""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return 0
    
    test_files = list(tests_dir.rglob("test_*.py"))
    return len(test_files)


def count_test_functions():
    """Count total test functions in the project."""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return 0
    
    count = 0
    for test_file in tests_dir.rglob("test_*.py"):
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                # Simple count of test_ functions
                count += content.count("def test_")
        except Exception:
            continue
    
    return count


def check_pytest_available():
    """Check if pytest is available."""
    try:
        import pytest
        return True
    except ImportError:
        return False


def run_basic_test_discovery():
    """Run basic pytest discovery to check test health."""
    if not check_pytest_available():
        print("❌ pytest not available")
        return False
    
    try:
        # Use static command list to prevent command injection
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Test discovery timed out")
        return False
    except Exception as e:
        print(f"❌ Test discovery failed: {e}")
        return False


def main():
    """Main function to run test health check."""
    print("🧪 SAMO-DL Test Health Check")
    print("=" * 40)
    
    # Basic counts
    test_files = count_test_files()
    test_functions = count_test_functions()
    
    print(f"📁 Test files: {test_files}")
    print(f"🔧 Test functions: {test_functions}")
    
    # Check pytest availability
    if check_pytest_available():
        print("✅ pytest available")
    else:
        print("❌ pytest not available")
    
    # Run basic discovery
    print("\n🔍 Running test discovery...")
    if run_basic_test_discovery():
        print("✅ Test discovery successful")
    else:
        print("❌ Test discovery failed")
    
    print("\n📊 Summary:")
    print(f"- Total test files: {test_files}")
    print(f"- Total test functions: {test_functions}")
    print(f"- pytest available: {'Yes' if check_pytest_available() else 'No'}")
    
    if test_files > 0 and test_functions > 0:
        print("🎯 Test suite appears healthy!")
    else:
        print("⚠️  Test suite may need attention")


if __name__ == "__main__":
    main()
