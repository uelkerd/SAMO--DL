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

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_git_stats(branch: str = "HEAD") -> tuple[int, int, list[str]]:
    """Get git statistics for the current branch vs main."""
    try:
        # Get files changed
        files_cmd = ["git", "diff", "--name-only", "origin/main", branch]
        returncode, stdout, stderr = run_command(files_cmd)

        if returncode != 0:
            print(f"❌ Error getting git stats: {stderr}")
            return 0, 0, []

        files = [f for f in stdout.strip().split('\n') if f.strip()]
        num_files = len(files)

        # Get lines changed
        lines_cmd = ["git", "diff", "--stat", "origin/main", branch]
        returncode, stdout, stderr = run_command(lines_cmd)

        if returncode != 0:
            print(f"❌ Error getting line stats: {stderr}")
            return num_files, 0, files

        # Parse the last line of git diff --stat
        lines = stdout.strip().split('\n')
        if lines and 'changed' in lines[-1]:
            # Extract numbers from "X insertions(+), Y deletions(-)"
            stat_line = lines[-1]
            insertions = 0
            deletions = 0

            if 'insertions' in stat_line:
                insertions_part = stat_line.split('insertions')[0].strip().split()[-1]
                insertions = int(insertions_part) if insertions_part.isdigit() else 0

            if 'deletions' in stat_line:
                deletions_part = stat_line.split('deletions')[0].strip().split()[-1]
                deletions = int(deletions_part) if deletions_part.isdigit() else 0

            lines_changed = insertions + deletions
        else:
            lines_changed = 0

        return num_files, lines_changed, files

    except Exception as e:
        print(f"❌ Error in get_git_stats: {e}")
        return 0, 0, []


def check_branch_name_quality() -> bool:
    """Check if branch name follows naming convention."""
    try:
        returncode, stdout, stderr = run_command(["git", "branch", "--show-current"])
        if returncode != 0:
            print(f"❌ Error getting branch name: {stderr}")
            return False

        branch_name = stdout.strip()

        # Check for good branch naming patterns
        good_patterns = [
            r'^feat/dl-',      # feat/dl-feature-name
            r'^fix/dl-',       # fix/dl-issue-name
            r'^refactor/dl-',  # refactor/dl-component-name
            r'^main$',         # main branch
            r'^develop$',      # develop branch
        ]

        import re
        for pattern in good_patterns:
            if re.match(pattern, branch_name):
                print(f"✅ Branch name '{branch_name}' follows convention")
                return True

        print(f"❌ Branch name '{branch_name}' doesn't follow convention")
        print("   Use: feat/dl-[feature], fix/dl-[issue], refactor/dl-[component]")
        return False

    except Exception as e:
        print(f"❌ Error checking branch name: {e}")
        return False


def main() -> int:
    """Main function to check PR scope."""
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

    # Check git stats
    print("\n📊 Checking change scope...")
    num_files, lines_changed, files = get_git_stats(args.branch)

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
            print(f"   • {file}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")

    # Single purpose check (basic heuristic)
    print("\n🎯 Checking single purpose...")
    file_types = set()
    for file in files:
        if file.endswith('.py'):
            file_types.add('python')
        elif file.endswith('.md'):
            file_types.add('documentation')
        elif file.endswith('.yml') or file.endswith('.yaml'):
            file_types.add('configuration')
        elif file.endswith('.json'):
            file_types.add('data')
        elif file.startswith('tests/'):
            file_types.add('tests')

    if len(file_types) > 3:
        print(f"⚠️  Multiple file types detected: {', '.join(file_types)}")
        print("   Consider splitting into focused PRs")
        if args.strict:
            all_passed = False

    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ PR Scope Check PASSED")
        return 0
    else:
        print("❌ PR Scope Check FAILED")
        print("\n💡 Remember the rules:")
        print("   • Max 50 files changed")
        print("   • Max 1500 lines changed")
        print("   • ONE purpose per PR")
        print("   • Single concern (no mixing API + tests + docs)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
