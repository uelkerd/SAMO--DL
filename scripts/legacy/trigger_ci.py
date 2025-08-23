#!/usr/bin/env python3
"""Script to check git status and trigger CI pipeline."""

import logging
import subprocess
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_command(cmd: str, description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    logging.info(f"🔄 {description}...")
    try:
        cmd_list = cmd.split()
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
        output = result.stdout.strip()
        if result.returncode == 0:
            logging.info(f"✅ {description} - SUCCESS")
            return True, output
        else:
            logging.info(f"❌ {description} - FAILED")
            logging.info(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logging.info(f"❌ {description} - EXCEPTION: {e}")
        return False, str(e)


def main():
    """Main function to trigger CI."""
    logging.info("🚀 Triggering CI Pipeline for SAMO Deep Learning")
    logging.info("=" * 50)

    # Check current git status
    success, status_output = run_command("git status", "Checking git status")
    if not success:
        logging.info("❌ Failed to check git status")
        return

    logging.info(f"Git Status:\n{status_output}")

    # Check if we have uncommitted changes
    if (
        "Changes not staged for commit" in status_output
        or "Changes to be committed" in status_output
    ):
        logging.info("📝 Found uncommitted changes, committing them...")

        # Add all changes
        success, _ = run_command("git add .", "Adding all changes")
        if not success:
            logging.info("❌ Failed to add changes")
            return

        # Commit changes
        success, _ = run_command(
            'git commit -m "Fix CI test failures: BERT mocking and predict_emotions bug"',
            "Committing changes",
        )
        if not success:
            logging.info("❌ Failed to commit changes")
            return

    success, log_output = run_command("git log --oneline -3", "Checking recent commits")
    if not success:
        logging.info("❌ Failed to check git log")
        return

    logging.info(f"Recent commits:\n{log_output}")

    logging.info("🚀 Force pushing to trigger CI pipeline...")
    # Force push to trigger CI
    success, push_output = run_command(
        "git push --force-with-lease", "Force pushing to remote"
    )

    if success:
        logging.info("✅ Successfully pushed changes!")
        logging.info("🔄 CI pipeline should be triggered now.")
        logging.info("📊 Check CircleCI dashboard for the new pipeline run.")
    else:
        logging.info("❌ Failed to push changes")
        logging.info(f"Push output: {push_output}")

        logging.info("🔄 Trying regular push as fallback...")
        # Try regular push as fallback
        success, push_output = run_command("git push", "Regular push")
        if success:
            logging.info("✅ Successfully pushed changes with regular push!")
        else:
            logging.info("❌ Both force push and regular push failed")


if __name__ == "__main__":
    main()
