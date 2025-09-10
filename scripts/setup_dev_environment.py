#!/usr/bin/env python3
"""
Development environment setup script for SAMO-DL project.
Installs dependencies and sets up pre-commit hooks.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Setting up {description}...")
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Set up development environment."""
    print("🚀 Setting up SAMO-DL Development Environment")
    print("=" * 50)
    
    # Install dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], "project dependencies"):
        print("❌ Failed to install project dependencies")
        return 1
    
    if not run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], "development dependencies"):
        print("❌ Failed to install development dependencies")
        return 1
    
    # Install pre-commit hooks
    if not run_command(["pre-commit", "install"], "pre-commit hooks"):
        print("❌ Failed to install pre-commit hooks")
        return 1
    
    # Run initial quality checks
    if not run_command([sys.executable, "scripts/run_quality_checks.py"], "initial quality checks"):
        print("⚠️  Some quality checks failed, but environment is set up")
    
    print("\n✅ Development environment setup complete!")
    print("📝 Next steps:")
    print("   1. Run 'python scripts/run_quality_checks.py' to check code quality")
    print("   2. Run 'pytest' to run tests")
    print("   3. Run 'pre-commit run --all-files' to check all files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
