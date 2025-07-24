#!/usr/bin/env python3
"""Verify git status and force push changes."""

import subprocess


def run_git_command(cmd_parts, description):
    """Run a git command safely."""
    try:
        result = subprocess.run(["git", *cmd_parts], capture_output=True, text=True, check=False)
        if result.stderr.strip():
            pass
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def main() -> None:
    # Check status
    success, output = run_git_command(["status", "--porcelain"], "Checking git status")
    if output:
        # Add all changes
        run_git_command(["add", "."], "Adding all changes")

        # Commit with a new message
        run_git_command(
            ["commit", "-m", "Fix CI: device attribute, test mocking, and security issues"],
            "Committing changes",
        )

        # Push with force
        run_git_command(["push", "--force-with-lease"], "Force pushing changes")
    else:
        # Try to push anyway
        run_git_command(["push"], "Attempting push")

    # Show recent commits
    run_git_command(["log", "--oneline", "-3"], "Showing recent commits")


if __name__ == "__main__":
    main()
