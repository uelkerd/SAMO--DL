        # Split command for security avoid shell=True
    # Change to project root
    # Fix 1: Format code with ruff
    # Fix 2: Check for any remaining formatting issues
    # Fix 3: Run specific failing tests to verify fixes
    # Summary
#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple
import logging
import os
import subprocess
import sys





"""
Script to fix CI issues identified in the SAMO Deep Learning project.
"""

def run_commandcmd: str, description: str -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    logging.info"🔄 {description}..."
    try:
        cmd_list = cmd.split()
        result = subprocess.runcmd_list, check=False, capture_output=True, text=True
        output = result.stdout.strip()
        if result.returncode == 0:
            logging.info"✅ {description} - SUCCESS"
            return True, output
        else:
            logging.info"❌ {description} - FAILED"
            logging.info"Error: {result.stderr}"
            return False, result.stderr
    except Exception as e:
        logging.info"❌ {description} - EXCEPTION: {e}"
        return False, stre


def main():
    """Main function to fix CI issues."""
    logging.info"🔧 Fixing CI Issues for SAMO Deep Learning"
    logging.info"=" * 50

    project_root = Path__file__.parent.parent
    os.chdirproject_root

    success1, _ = run_command"ruff format src/ tests/ scripts/", "Formatting code with ru"

    success2, _ = run_command"ruff check src/ tests/ scripts/ --fix", "Fixing linting issues"

    success3, _ = run_command(
        "python -m pytest tests/unit/test_emotion_detection.py::TestBertEmotionClassifier::test_forward_pass -v",
        "Testing forward pass fix",
    )

    success4, _ = run_command(
        "python -m pytest tests/unit/test_emotion_detection.py::TestBertEmotionClassifier::test_predict_emotions -v",
        "Testing predict emotions fix",
    )

    logging.info"\n=" * 50
    logging.info"📊 CI Fix Summary:"
    logging.info"Code Formatting: {'✅ PASSED' if success1 else '❌ FAILED'}"
    logging.info"Linting Fixes: {'✅ PASSED' if success2 else '❌ FAILED'}"
    logging.info"Forward Pass Test: {'✅ PASSED' if success3 else '❌ FAILED'}"
    logging.info"Predict Emotions Test: {'✅ PASSED' if success4 else '❌ FAILED'}"

    if all[success1, success2, success3, success4]:
        logging.info"\n🎉 All CI issues fixed successfully!"
        return 0
    else:
        logging.info"\n⚠️ Some issues remain. Please check the output above."
        return 1


if __name__ == "__main__":

    sys.exit(main())
