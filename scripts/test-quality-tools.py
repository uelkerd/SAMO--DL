#!/usr/bin/env python3
"""Integration test script for code quality tools.

This script validates that all quality tools in the pre-commit configuration
work together properly and catch common code quality issues.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


def create_test_files() -> Dict[str, str]:
    """Create test Python files with various quality issues."""
    return {
        "good_code.py": '''"""A well-formatted Python module."""

import os
from typing import List


def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers.

    Args:
        numbers: List of integers to sum

    Returns:
        The sum of all numbers
    """
    return sum(numbers)


if __name__ == "__main__":
    result = calculate_sum([1, 2, 3, 4, 5])
    print(f"Sum: {result}")
''',

        "bad_code.py": """# Bad code with various issues
import sys
import unused_module   # Unused import (should be caught by Ruff)

def badfunction(x,y):   # Bad formatting, no type hints
    if x==y:     # Bad spacing
        return "equal"   # Inconsistent quotes
    else:
        return "not equal"

# Trailing whitespace on next line
x = 1

# Missing final newline""",

        "security_issues.py": """# Code with security issues
import os
import subprocess

# Hardcoded password (should be caught by Bandit)
PASSWORD = "hardcoded123"

def run_command(user_input):
    # Command injection vulnerability (should be caught by Bandit)
    subprocess.call(f"echo {user_input}", shell=True)

def use_temp_file():
    # Insecure temp file (should be caught by Bandit)
    return "/tmp/temp_file.txt"
""",

        "type_issues.py": """# Code with type issues (should be caught by MyPy)
def process_data(data):
    return data.upper() + " processed"

def main():
    # This will cause a type error
    result = process_data(123)
    print(result)
""",
    }


def run_tool(command: List[str], description: str) -> bool:
    """Run a tool and return True if it passes (exit code 0)."""
    print(f"\n🔍 Running {description}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        print(f"❌ {description} failed with exit code {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
        return False


def test_individual_tools() -> Dict[str, bool]:
    """Test individual tools on their own."""
    results = {}

    # Test Ruff
    results["ruff"] = run_tool(["ruff", "--version"], "Ruff version check")

    # Test MyPy
    results["mypy"] = run_tool(["mypy", "--version"], "MyPy version check")

    # Test Pylint
    results["pylint"] = run_tool(["pylint", "--version"], "Pylint version check")

    # Test Bandit
    results["bandit"] = run_tool(["bandit", "--version"], "Bandit version check")

    # Test Safety
    results["safety"] = run_tool(["safety", "--version"], "Safety version check")

    # Test Pre-commit
    results["pre-commit"] = run_tool(["pre-commit", "--version"], "Pre-commit version check")

    return results


def test_quality_detection(temp_dir: Path) -> Dict[str, bool]:
    """Test that quality tools detect issues in bad code."""
    results = {}

    # Create test files
    test_files = create_test_files()
    for filename, content in test_files.items():
        (temp_dir / filename).write_text(content)

    # Test Ruff on bad code (should find issues)
    bad_file = temp_dir / "bad_code.py"
    ruff_result = subprocess.run(["ruff", "check", str(bad_file)],
                                capture_output=True, text=True, check=False)
    results["ruff_detects_issues"] = ruff_result.returncode != 0

    # Test Bandit on security issues (should find issues)
    security_file = temp_dir / "security_issues.py"
    bandit_result = subprocess.run(["bandit", str(security_file)],
                                  capture_output=True, text=True, check=False)
    results["bandit_detects_issues"] = bandit_result.returncode != 0

    # Test Ruff on good code (should pass)
    good_file = temp_dir / "good_code.py"
    ruff_good_result = subprocess.run(["ruff", "check", str(good_file)],
                                     capture_output=True, text=True, check=False)
    results["ruff_passes_good_code"] = ruff_good_result.returncode == 0

    return results


def test_pre_commit_integration(temp_dir: Path) -> bool:
    """Test pre-commit integration in the temporary directory."""
    # Initialize git repo in temp dir
    subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, check=False)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                  cwd=temp_dir, capture_output=True, check=False)
    subprocess.run(["git", "config", "user.name", "Test User"],
                  cwd=temp_dir, capture_output=True, check=False)

    # Copy pre-commit config
    project_root = Path(__file__).parent.parent
    precommit_config = project_root / ".pre-commit-config.yaml"
    if precommit_config.exists():
        import shutil
        shutil.copy(precommit_config, temp_dir / ".pre-commit-config.yaml")

        # Install pre-commit hooks
        install_result = subprocess.run(["pre-commit", "install"],
                                      cwd=temp_dir, capture_output=True, check=False)

        return install_result.returncode == 0
    print("❌ Pre-commit config not found")
    return False


def main():
    """Run all quality tool tests."""
    print("🧪 SAMO-DL Code Quality Tools Integration Test")
    print("=" * 50)

    # Test 1: Individual tool availability
    print("\n📋 Phase 1: Testing individual tool availability")
    tool_results = test_individual_tools()

    # Test 2: Quality detection in temporary directory
    print("\n📋 Phase 2: Testing quality issue detection")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        detection_results = test_quality_detection(temp_path)

        # Test 3: Pre-commit integration
        print("\n📋 Phase 3: Testing pre-commit integration")
        precommit_result = test_pre_commit_integration(temp_path)

    # Report results
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)

    print("\n🔧 Tool Availability:")
    for tool, passed in tool_results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {tool}")

    print("\n🔍 Quality Detection:")
    for test, passed in detection_results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test.replace('_', ' ').title()}")

    print(f"\n🔗 Pre-commit Integration: {'✅' if precommit_result else '❌'}")

    # Overall success
    all_tools_available = all(tool_results.values())
    quality_detection_works = all(detection_results.values())
    overall_success = all_tools_available and quality_detection_works and precommit_result

    print(f"\n🎯 Overall Status: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")

    if not overall_success:
        print("\n⚠️  Issues detected. Please check the failing components above.")
        sys.exit(1)
    else:
        print("\n🎉 All quality tools are working correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
