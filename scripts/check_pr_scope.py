#!/usr/bin/env python3
"""
PR Scope Checker - Prevents Monster PRs

This script validates that pull requests stay within scope limits:
- Max 25 files changed
- Max 500 lines changed
- Single purpose (one concern per PR)
- No mixing of concerns (API + tests + docs, etc.)

Usage:
    python scripts/check_pr_scope.py [--branch <branch>] [--strict]
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str]) -> Tuple[str, str, int]:
    """Run command and return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_git_stats(branch: str = "HEAD") -> Tuple[int, int, List[str]]:
    """Get git statistics for the current branch."""
    # Get number of files changed
    stdout, stderr, code = run_command(["git", "diff", "--name-only", f"{branch}~1"])
    if code != 0:
        print(f"❌ Git error: {stderr}")
        return 0, 0, []

    files = stdout.split('\n') if stdout else []
    num_files = len([f for f in files if f.strip()])

    # Get lines changed
    stdout, stderr, code = run_command(["git", "diff", "--stat", f"{branch}~1"])
    lines_changed = 0
    if code == 0 and stdout:
        # Parse last line of git diff --stat
        lines = stdout.strip().split('\n')
        if lines:
            last_line = lines[-1]
            # Extract number from format like "3 files changed, 150 insertions(+), 50 deletions(-)"
            parts = last_line.split(',')
            for part in parts:
                if 'insertions' in part or 'deletions' in part:
                    nums = ''.join(c for c in part if c.isdigit())
                    if nums:
                        lines_changed += int(nums)

    return num_files, lines_changed, files


def check_commit_message_quality() -> bool:
    """Check if commit messages follow single-purpose rules."""
    stdout, stderr, code = run_command(["git", "log", "--oneline", "-1"])
    if code != 0:
        print(f"❌ Git log error: {stderr}")
        return False

    commit_msg = stdout.strip()
    if not commit_msg:
        print("❌ No commit message found")
        return False

    # Check for single-purpose keywords
    single_purpose_keywords = ['feat:', 'fix:', 'chore:', 'refactor:', 'docs:', 'test:']
    has_single_purpose = any(commit_msg.startswith(keyword) for keyword in single_purpose_keywords)

    if not has_single_purpose:
        print("❌ Commit message must start with feat:, fix:, chore:, refactor:, docs:, or test:")
        print(f"   Current: {commit_msg}")
        return False

    # Check for mixing concerns (contains 'and', 'also', 'plus')
    mixing_indicators = [' and ', ' also ', ' plus ', ' & ', ' in addition ']
    if any(indicator in commit_msg.lower() for indicator in mixing_indicators):
        print("❌ Commit message indicates multiple concerns (contains 'and', 'also', etc.)")
        print(f"   Current: {commit_msg}")
        return False

    return True


def check_branch_name_quality() -> bool:
    """Check if branch name follows naming conventions."""
    stdout, stderr, code = run_command(["git", "branch", "--show-current"])
    if code != 0:
        print(f"❌ Git branch error: {stderr}")
        return False

    branch_name = stdout.strip()
    if not branch_name:
        print("❌ Could not determine current branch")
        return False

    # Check naming pattern: type/short-description
    import re
    pattern = r'^(feat|fix|chore|refactor|docs|test)/[a-z-]+(-[a-z-]+)*$'
    if not re.match(pattern, branch_name):
        print("❌ Branch name must follow pattern: type/short-description")
        print(f"   Current: {branch_name}")
        print("   Examples: feat/add-user-auth, fix/validate-input, chore/update-deps")
        return False

    return True


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Check PR scope compliance")
    parser.add_argument("--branch", default="HEAD", help="Branch to check (default: HEAD)")
    parser.add_argument("--strict", action="store_true", help="Strict mode - fail on any warning")
    args = parser.parse_args()

    print("🔍 Checking PR Scope Compliance")
    print("=" * 50)

    all_passed = True

    # Check branch name
    print("📋 Checking branch name...")
    if not check_branch_name_quality():
        all_passed = False

    # Check commit message
    print("\n📝 Checking commit message...")
    if not check_commit_message_quality():
        all_passed = False

    # Check file and line limits
    print("\n📊 Checking size limits...")
    num_files, lines_changed, files = get_git_stats(args.branch)

    print(f"   Files changed: {num_files}")
    print(f"   Lines changed: {lines_changed}")

    if num_files > 25:
        print(f"❌ Too many files changed! Max 25 allowed, got {num_files}")
        if args.strict:
            all_passed = False

    if lines_changed > 500:
        print(f"❌ Too many lines changed! Max 500 allowed, got {lines_changed}")
        if args.strict:
            all_passed = False

    # List changed files
    if files:
        print("\n📁 Files changed:")
        for file in files[:10]:  # Show first 10
            if file.strip():
                print(f"   - {file}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")

    # Check for mixed concerns
    print("\n🎯 Checking for mixed concerns...")
    if files:
        file_types = set()
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.h')):
                file_types.add('code')
            elif file.endswith(('.md', '.rst', '.txt', '.adoc')):
                file_types.add('docs')
            elif 'test' in file.lower() or file.startswith('tests/'):
                file_types.add('tests')
            elif 'config' in file.lower() or file.endswith(('.yml', '.yaml', '.json', '.toml', '.cfg')):
                file_types.add('config')
            elif 'docker' in file.lower() or 'Dockerfile' in file:
                file_types.add('docker')
            elif 'api' in file.lower() or 'endpoint' in file.lower():
                file_types.add('api')
            elif 'model' in file.lower() or 'ml' in file.lower() or 'ai' in file.lower():
                file_types.add('ml')

        if len(file_types) > 2:
            print(f"⚠️  Mixed concerns detected: {', '.join(file_types)}")
            print("   Consider splitting into separate PRs")
            if args.strict:
                all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ PR Scope Check PASSED")
        return 0
    else:
        print("❌ PR Scope Check FAILED")
        print("\n💡 Remember the rules:")
        print("   • Max 25 files changed")
        print("   • Max 500 lines changed")
        print("   • ONE purpose per PR")
        print("   • Single concern (no mixing API + tests + docs)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
