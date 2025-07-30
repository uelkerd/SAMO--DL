#!/usr/bin/env python3
"""
Script to fix CI issues identified in the SAMO Deep Learning project.
"""

import subprocess
from pathlib import Path


def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print("🔄 {description}...")
    try:
        # Split command for security (avoid shell=True)
        cmd_list = cmd.split()
        result = subprocess.run(cmd_list, check=False, capture_output=True, text=True)
        output = result.stdout.strip()
        if result.returncode == 0:
            print("✅ {description} - SUCCESS")
            return True, output
        else:
            print("❌ {description} - FAILED")
            print("Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print("❌ {description} - EXCEPTION: {e}")
        return False, str(e)


def main():
    """Main function to fix CI issues."""
    print("🔧 Fixing CI Issues for SAMO Deep Learning")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Fix 1: Format code with ruff
    success1, _ = run_command("ruff format src/ tests/ scripts/", "Formatting code with ruf")

    # Fix 2: Check for any remaining formatting issues
    success2, _ = run_command("ruff check src/ tests/ scripts/ --fix", "Fixing linting issues")

    # Fix 3: Run specific failing tests to verify fixes
    success3, _ = run_command(
        "python -m pytest tests/unit/test_emotion_detection.py::TestBertEmotionClassifier::test_forward_pass -v",
        "Testing forward pass fix",
    )

    success4, _ = run_command(
        "python -m pytest tests/unit/test_emotion_detection.py::TestBertEmotionClassifier::test_predict_emotions -v",
        "Testing predict emotions fix",
    )

    # Summary
    print("\n" + "=" * 50)
    print("📊 CI Fix Summary:")
    print("Code Formatting: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print("Linting Fixes: {'✅ PASSED' if success2 else '❌ FAILED'}")
    print("Forward Pass Test: {'✅ PASSED' if success3 else '❌ FAILED'}")
    print("Predict Emotions Test: {'✅ PASSED' if success4 else '❌ FAILED'}")

    if all([success1, success2, success3, success4]):
        print("\n🎉 All CI issues fixed successfully!")
        return 0
    else:
        print("\n⚠️ Some issues remain. Please check the output above.")
        return 1


if __name__ == "__main__":

    sys.exit(main())
