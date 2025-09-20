#!/usr/bin/env python3
"""
PR Scope Checker - Prevents Monster PRs

This script validates that pull requests stay within scope limits:
- Max 50 files changed
- Max 1500 lines changed
- Single purpose (one concern per PR)
- No mixing of concerns (API + tests + docs, etc.)

Usage:
    python scripts/check_pr_scope.py [--branch <branch>] [--strict]
"""

import os
import subprocess
import sys
from typing import List, Tuple


def run_command(cmd: List[str]) -> Tuple[str, str, int]:
    """Run command and return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_git_stats(base: str = "HEAD~1", head: str = "HEAD") -> Tuple[int, int, List[str]]:
    """Get git statistics for a commit range."""
    # Use git range operator base...head for both name-only and stats
    range_spec = f"{base}...{head}"

    # Get number of files changed
    stdout, stderr, code = run_command(["git", "diff", "--name-only", range_spec])
    if code != 0:
        print(f"❌ Git diff error: {stderr}")
        return 0, 0, []

    files = stdout.split('\n') if stdout else []
    num_files = len([f for f in files if f.strip()])

    # Get lines changed using shortstat
    stdout, stderr, code = run_command(["git", "diff", "--shortstat", range_spec])
    lines_changed = 0
    if code == 0 and stdout:
        # Parse shortstat format like "1 file changed, 10 insertions(+), 2 deletions(-)"
        shortstat = stdout.strip()
        if shortstat:
            parts = shortstat.split(',')
            for part in parts:
                part = part.strip()
                if 'insertions' in part or 'deletions' in part:
                    nums = ''.join(c for c in part if c.isdigit())
                    if nums:
                        lines_changed += int(nums)

    return num_files, lines_changed, files


def check_commit_message_quality(base: str = None, head: str = None) -> bool:
    """Check if commit messages follow single-purpose rules for all commits in range."""
    # Determine commit range
    if base and head:
        commit_range = f"{base}..{head}"
    else:
        # Try to determine range from environment or git context
        # Check for common CI environment variables
        if os.environ.get('GITHUB_BASE_REF') and os.environ.get('GITHUB_HEAD_REF'):
            base_ref = os.environ['GITHUB_BASE_REF']
            head_ref = os.environ['GITHUB_HEAD_REF']
            commit_range = f"origin/{base_ref}..origin/{head_ref}"
        else:
            # Fallback to HEAD^..HEAD for single commit
            commit_range = "HEAD^..HEAD"

    print(f"🔍 Checking commit messages in range: {commit_range}")

    # Get all non-merge commit SHAs in the range
    stdout, stderr, code = run_command(["git", "rev-list", "--no-merges", commit_range])
    if code != 0:
        print(f"❌ Git rev-list error: {stderr}")
        return False

    commit_shas = [sha.strip() for sha in stdout.split('\n') if sha.strip()]

    if not commit_shas:
        print("ℹ️  No non-merge commits found in range")
        return True

    print(f"📝 Found {len(commit_shas)} commit(s) to check")

    all_passed = True

    for sha in commit_shas:
        # Get the oneline commit message
        stdout, stderr, code = run_command(["git", "log", "-1", "--format=%s", sha])
        if code != 0:
            print(f"❌ Failed to get commit message for {sha}: {stderr}")
            all_passed = False
            continue

        commit_msg = stdout.strip()
        if not commit_msg:
            print(f"❌ Empty commit message for {sha}")
            all_passed = False
            continue

        print(f"🔎 Checking commit {sha[:8]}: {commit_msg}")

        # Check for single-purpose keywords
        single_purpose_keywords = ['feat:', 'fix:', 'chore:', 'refactor:', 'docs:', 'test:']
        has_single_purpose = any(commit_msg.startswith(keyword) for keyword in single_purpose_keywords)

        if not has_single_purpose:
            print(f"❌ Commit {sha[:8]} message must start with feat:, fix:, chore:, refactor:, docs:, or test:")
            print(f"   Message: {commit_msg}")
            all_passed = False
            continue

        # Check for mixing concerns (contains 'and', 'also', 'plus')
        mixing_indicators = [' and ', ' also ', ' plus ', ' & ', ' in addition ']
        if any(indicator in commit_msg.lower() for indicator in mixing_indicators):
            print(f"❌ Commit {sha[:8]} message indicates multiple concerns (contains 'and', 'also', etc.)")
            print(f"   Message: {commit_msg}")
            all_passed = False
            continue

        print(f"✅ Commit {sha[:8]} passed quality checks")

    return all_passed


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
    pattern = r'^(feat|fix|chore|refactor|docs|test)/[a-z]+(?:-[a-z]+)*$'
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
    parser.add_argument("--base", help="Base branch/commit for range comparison (e.g., main, origin/main)")
    parser.add_argument("--head", help="Head branch/commit for range comparison (default: current branch)")
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
    if not check_commit_message_quality(args.base, args.head):
        all_passed = False

    # Check file and line limits
    print("\n📊 Checking size limits...")
    # Determine base for git stats
    base_for_stats = args.base if args.base else f"{args.branch}~1"
    head_for_stats = args.head if args.head else args.branch
    num_files, lines_changed, files = get_git_stats(base_for_stats, head_for_stats)

    print(f"   Files changed: {num_files}")
    print(f"   Lines changed: {lines_changed}")

    if num_files > 50:
        print(f"❌ Too many files changed! Max 50 allowed, got {num_files}")
        if args.strict:
            all_passed = False

    if lines_changed > 1500:
        print(f"❌ Too many lines changed! Max 1500 allowed, got {lines_changed}")
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
    print("❌ PR Scope Check FAILED")
    print("\n💡 Remember the rules:")
    print("   • Max 50 files changed")
    print("   • Max 1500 lines changed")
    print("   • ONE purpose per PR")
    print("   • Single concern (no mixing API + tests + docs)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
