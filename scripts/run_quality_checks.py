#!/usr/bin/env python3
"""
Quality assurance script for SAMO-DL project.
Runs all code quality checks and generates reports.
"""

import subprocess
import sys
import shlex
from pathlib import Path
from typing import List, Tuple

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"Running {description}...")
    try:
        # Validate that all command arguments are strings and not empty
        if not all(isinstance(arg, str) and arg.strip() for arg in cmd):
            return False, "Invalid command arguments: all arguments must be non-empty strings"
        
        # Log the command being executed for security auditing
        escaped_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
        print(f"Executing: {escaped_cmd}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent.parent, 
            check=True)
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr
    except Exception as e:
        return False, str(e)

def main():
    """Run all quality checks."""
    print("🔍 Running SAMO-DL Quality Checks")
    print("=" * 50)

    checks = [
        (["black", "--check", "src/", "tests/"], "Black formatting check"),
        (["isort", "--check-only", "src/", "tests/"], "Import sorting check"),
        (["flake8", "src/", "tests/"], "Flake8 linting"),
        (["pylint", "src/", "tests/"], "Pylint analysis"),
        (["mypy", "src/"], "Type checking"),
        (["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"], "Security analysis"),
        (["safety", "check", "--json", "--output", "safety-report.json"], "Dependency security"),
        (["pytest", "tests/", "--cov=src", "--cov-report=term-missing"], "Unit tests with coverage"),
    ]

    results = []
    for cmd, description in checks:
        success, output = run_command(cmd, description)
        results.append((description, success, output))

        if success:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            print(f"   Error: {output[:200]}...")

    print("\n" + "=" * 50)
    print("📊 Quality Check Summary")
    print("=" * 50)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for description, success, _ in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All quality checks passed!")
        return 0
    print("⚠️  Some quality checks failed. Please review the output above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
