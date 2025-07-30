#!/usr/bin/env python3
"""
Script to check git status and trigger CI pipeline.
"""

import subprocess


def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print("🔄 {description}...")
    try:
        # Split command for security (avoid shell=True)
        cmd_list = cmd.split()
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
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
    """Main function to trigger CI."""
    print("🚀 Triggering CI Pipeline for SAMO Deep Learning")
    print("=" * 50)

    # Check current git status
    success, status_output = run_command("git status", "Checking git status")
    if not success:
        print("❌ Failed to check git status")
        return

    print("Git Status:\n{status_output}")

    # Check if we have uncommitted changes
    if (
        "Changes not staged for commit" in status_output
        or "Changes to be committed" in status_output
    ):
        print("📝 Found uncommitted changes, committing them...")

        # Add all changes
        success, _ = run_command("git add .", "Adding all changes")
        if not success:
            print("❌ Failed to add changes")
            return

        # Commit changes
        success, _ = run_command(
            'git commit -m "Fix CI test failures: BERT mocking and predict_emotions bug"',
            "Committing changes",
        )
        if not success:
            print("❌ Failed to commit changes")
            return

    # Check if we need to push
    success, log_output = run_command("git log --oneline -3", "Checking recent commits")
    if not success:
        print("❌ Failed to check git log")
        return

    print("Recent commits:\n{log_output}")

    # Force push to trigger CI
    print("🚀 Force pushing to trigger CI pipeline...")
    success, push_output = run_command("git push --force-with-lease", "Force pushing to remote")

    if success:
        print("✅ Successfully pushed changes!")
        print("🔄 CI pipeline should be triggered now.")
        print("📊 Check CircleCI dashboard for the new pipeline run.")
    else:
        print("❌ Failed to push changes")
        print("Push output: {push_output}")

        # Try regular push as fallback
        print("🔄 Trying regular push as fallback...")
        success, push_output = run_command("git push", "Regular push")
        if success:
            print("✅ Successfully pushed changes with regular push!")
        else:
            print("❌ Both force push and regular push failed")


if __name__ == "__main__":
    main()
