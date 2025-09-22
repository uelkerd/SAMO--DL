#!/usr/bin/env python3
"""PR Scope Checker - Prevents Monster PRs.

This script validates that pull requests stay within scope limits:
- Max 50 files changed
- Max 1500 lines changed
- Single purpose (one concern per PR)
- No mixing of concerns (API + tests + docs, etc.)

Usage:
    python scripts/check_pr_scope.py [--branch <branch>] [--strict]
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

try:
    from pr_scope_config import (
        ACCEPTABLE_COMBINATIONS,
        BRANCH_NAME_PATTERN,
        FILE_TYPE_PATTERNS,
        MAX_FILES_CHANGED,
        MAX_FILES_TO_DISPLAY,
        MAX_FILE_TYPES_FOR_MIXED_CONCERNS,
        MAX_FILE_TYPES_FOR_WARNING,
        MAX_LINES_CHANGED,
        MIXING_INDICATORS,
        SINGLE_PURPOSE_KEYWORDS,
    )
except ImportError:
    # Fallback for when running from different directory
    import os
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from pr_scope_config import (
        ACCEPTABLE_COMBINATIONS,
        BRANCH_NAME_PATTERN,
        FILE_TYPE_PATTERNS,
        MAX_FILES_CHANGED,
        MAX_FILES_TO_DISPLAY,
        MAX_FILE_TYPES_FOR_MIXED_CONCERNS,
        MAX_FILE_TYPES_FOR_WARNING,
        MAX_LINES_CHANGED,
        MIXING_INDICATORS,
        SINGLE_PURPOSE_KEYWORDS,
    )


def run_command(cmd: List[str]) -> Tuple[str, str, int]:
    """Run command and return (stdout, stderr, returncode)."""
    # Security: Use shlex.escape for any user input, but since cmd is a list of strings
    # and we control all the commands, this is safe. The commands are hardcoded.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_git_stats(
    base: str = "HEAD~1",
    head: str = "HEAD",
) -> Tuple[int, int, List[str]]:
    """Get git statistics for a commit range."""
    # Use git range operator base...head for both name-only and stats
    range_spec = f"{base}...{head}"

    # Get number of files changed
    stdout, stderr, code = run_command(["git", "diff", "--name-only", range_spec])
    if code != 0:
        print(f"❌ Git diff error: {stderr}")
        return 0, 0, []

    files = stdout.split("\n") if stdout else []
    num_files = len([f for f in files if f.strip()])

    # Get lines changed using shortstat
    stdout, stderr, code = run_command(["git", "diff", "--shortstat", range_spec])
    lines_changed = 0
    if code != 0:
        print(f"❌ Git diff --shortstat error: {stderr}")
    elif stdout:
        # Parse shortstat format like "1 file changed, 10 insertions(+), 2 deletions(-)"
        shortstat = stdout.strip()
        if shortstat:
            insertions = 0
            deletions = 0
            ins_match = re.search(r"(\d+)\s+insertions?\(\+\)", shortstat)
            del_match = re.search(r"(\d+)\s+deletions?\(-\)", shortstat)
            if ins_match:
                insertions = int(ins_match.group(1))
            if del_match:
                deletions = int(del_match.group(1))
            lines_changed = insertions + deletions

    return num_files, lines_changed, files


def _determine_commit_range(base: Optional[str], head: Optional[str]) -> str:
    """Determine the commit range to check."""
    if base and head:
        return f"{base}..{head}"

    # Try to determine range from environment or git context
    # Check for common CI environment variables
    if os.environ.get("GITHUB_BASE_REF") and os.environ.get("GITHUB_HEAD_REF"):
        base_ref = os.environ["GITHUB_BASE_REF"]
        head_ref = os.environ["GITHUB_HEAD_REF"]
        return f"origin/{base_ref}..origin/{head_ref}"

    # Fallback to HEAD^..HEAD for single commit
    return "HEAD^..HEAD"


def _get_commit_shas(commit_range: str) -> List[str]:
    """Get all non-merge commit SHAs in the range."""
    stdout, stderr, code = run_command(["git", "rev-list", "--no-merges", commit_range])
    if code != 0:
        print(f"❌ Git rev-list error: {stderr}")
        return []

    return [sha.strip() for sha in stdout.split("\n") if sha.strip()]


def _check_single_commit(sha: str) -> bool:
    """Check a single commit message for quality."""
    # Get the oneline commit message
    stdout, stderr, code = run_command(["git", "log", "-1", "--format=%s", sha])
    if code != 0:
        print(f"❌ Failed to get commit message for {sha}: {stderr}")
        return False

    commit_msg = stdout.strip()
    if not commit_msg:
        print(f"❌ Empty commit message for {sha}")
        return False

    print(f"🔎 Checking commit {sha[:8]}: {commit_msg}")

    # Check for single-purpose keywords
    has_single_purpose = any(
        commit_msg.startswith(keyword) for keyword in SINGLE_PURPOSE_KEYWORDS
    )

    if not has_single_purpose:
        print(
            f"❌ Commit {sha[:8]} message must start with feat:, fix:, chore:, refactor:, docs:, or test:",
        )
        print(f"   Message: {commit_msg}")
        return False

    # Check for mixing concerns (contains 'and', 'also', 'plus')
    if any(indicator in commit_msg.lower() for indicator in MIXING_INDICATORS):
        print(
            f"❌ Commit {sha[:8]} message indicates multiple concerns (contains 'and', 'also', etc.)",
        )
        print(f"   Message: {commit_msg}")
        return False

    print(f"✅ Commit {sha[:8]} passed quality checks")
    return True


def check_commit_message_quality(
    base: Optional[str] = None,
    head: Optional[str] = None,
) -> bool:
    """Check if commit messages follow single-purpose rules for all commits in range."""
    commit_range = _determine_commit_range(base, head)
    print(f"🔍 Checking commit messages in range: {commit_range}")

    commit_shas = _get_commit_shas(commit_range)
    if not commit_shas:
        print("ℹ️  No non-merge commits found in range")
        return True

    print(f"📝 Found {len(commit_shas)} commit(s) to check")

    return all(_check_single_commit(sha) for sha in commit_shas)


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
    if not re.match(BRANCH_NAME_PATTERN, branch_name):
        print("❌ Branch name must follow pattern: type/short-description")
        print(f"   Current: {branch_name}")
        print("   Examples: feat/add-user-auth, fix/validate-input, chore/update-deps")
        return False

    return True


def _check_extensions(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches extension patterns."""
    return "extensions" in patterns and any(file_path.endswith(ext) for ext in patterns["extensions"])


def _check_directories(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches directory patterns."""
    return "directories" in patterns and any(file_path.startswith(dir_path) for dir_path in patterns["directories"])


def _check_suffixes(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches suffix patterns."""
    return "suffixes" in patterns and any(file_path.endswith(suffix) for suffix in patterns["suffixes"])


def _check_prefixes(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches prefix patterns."""
    return "prefixes" in patterns and any(file_path.startswith(prefix) for prefix in patterns["prefixes"])


def _check_patterns_regex(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches regex patterns."""
    if "patterns" not in patterns:
        return False
    return any(re.search(pattern, file_path, re.IGNORECASE) for pattern in patterns["patterns"])


def _check_exact_files(file_path: str, patterns: Dict[str, List[str]]) -> bool:
    """Check if file matches exact file patterns."""
    return "exact_files" in patterns and file_path in patterns["exact_files"]


def _check_keywords(file_path: str, patterns: Dict[str, List[str]], file_lower: str) -> bool:
    """Check if file matches keyword patterns."""
    return "keywords" in patterns and any(keyword in file_lower for keyword in patterns["keywords"])


def _check_patterns(file_path: str, patterns: Dict[str, List[str]], file_lower: str) -> bool:
    """Check if file matches any pattern in the given patterns dict."""
    return (
        _check_extensions(file_path, patterns) or
        _check_directories(file_path, patterns) or
        _check_suffixes(file_path, patterns) or
        _check_prefixes(file_path, patterns) or
        _check_patterns_regex(file_path, patterns) or
        _check_exact_files(file_path, patterns) or
        _check_keywords(file_path, patterns, file_lower)
    )


def detect_file_types(file_path: str) -> Set[str]:
    """Detect all applicable file types for a given file path."""
    file_lower = file_path.lower()
    detected_types = set()

    # First, check extension-based categories (mutually exclusive)
    if file_path.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c", ".h")):
        detected_types.add("code")
    elif file_path.endswith((".md", ".rst", ".txt", ".adoc")):
        detected_types.add("docs")

    # Then check other categories (not mutually exclusive)
    for file_type, patterns in FILE_TYPE_PATTERNS.items():
        if file_type in ["code", "docs"]:  # Skip already handled extension-based categories
            continue
        if _check_patterns(file_path, patterns, file_lower):
            detected_types.add(file_type)

    return detected_types


def check_mixed_concerns(files: List[str]) -> Tuple[bool, Set[str]]:
    """Check for mixed concerns in changed files."""
    if not files:
        return False, set()

    file_types = set()
    for file in files:
        detected_types = detect_file_types(file)
        file_types.update(detected_types)

    # Flag as mixed concerns only if we have more than MAX_FILE_TYPES_FOR_MIXED_CONCERNS types OR
    # if we have an unusual combination that's not in acceptable list
    is_mixed_concerns = len(file_types) > MAX_FILE_TYPES_FOR_MIXED_CONCERNS or (
        len(file_types) > MAX_FILE_TYPES_FOR_WARNING and file_types not in ACCEPTABLE_COMBINATIONS
    )

    return is_mixed_concerns, file_types


def check_size_limits(num_files: int, lines_changed: int, strict_mode: bool) -> bool:
    """Check if PR size is within limits."""
    all_passed = True

    if num_files > MAX_FILES_CHANGED:
        print(f"❌ Too many files changed! Max {MAX_FILES_CHANGED} allowed, got {num_files}")
        if strict_mode:
            all_passed = False

    if lines_changed > MAX_LINES_CHANGED:
        print(f"❌ Too many lines changed! Max {MAX_LINES_CHANGED} allowed, got {lines_changed}")
        if strict_mode:
            all_passed = False

    return all_passed


def display_changed_files(files: List[str]) -> None:
    """Display list of changed files."""
    if not files:
        return

    print("\n📁 Files changed:")
    for file in files[:MAX_FILES_TO_DISPLAY]:  # Show first MAX_FILES_TO_DISPLAY
        if file.strip():
            print(f"   - {file}")
    if len(files) > MAX_FILES_TO_DISPLAY:
        print(f"   ... and {len(files) - MAX_FILES_TO_DISPLAY} more")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check PR scope compliance")
    parser.add_argument(
        "--branch",
        default="HEAD",
        help="Branch to check (default: HEAD)",
    )
    parser.add_argument(
        "--base",
        help="Base branch/commit for range comparison (e.g., main, origin/main)",
    )
    parser.add_argument(
        "--head",
        help="Head branch/commit for range comparison (default: current branch)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode - fail on any warning",
    )
    return parser.parse_args()


def _check_branch_and_commits(args) -> bool:
    """Check branch name and commit messages."""
    print("📋 Checking branch name...")
    branch_ok = check_branch_name_quality()

    print("\n📝 Checking commit message...")
    commit_ok = check_commit_message_quality(args.base, args.head)

    return branch_ok and commit_ok


def _get_git_stats_for_check(args) -> Tuple[int, int, List[str]]:
    """Get git statistics for the given arguments."""
    print("\n📊 Checking size limits...")
    # Determine base for git stats - use or operator for cleaner code
    base_for_stats = args.base or f"{args.branch}~1"
    head_for_stats = args.head or args.branch
    num_files, lines_changed, files = get_git_stats(base_for_stats, head_for_stats)

    print(f"   Files changed: {num_files}")
    print(f"   Lines changed: {lines_changed}")

    return num_files, lines_changed, files


def _check_mixed_concerns_display(files: List[str], strict_mode: bool) -> bool:
    """Check and display mixed concerns information."""
    print("\n🎯 Checking for mixed concerns...")
    if not files:
        return True

    is_mixed_concerns, file_types = check_mixed_concerns(files)

    if is_mixed_concerns:
        print(f"⚠️  Mixed concerns detected: {', '.join(sorted(file_types))}")
        print("   Consider splitting into separate PRs")
        print(
            "   Acceptable combinations include: code+tests, code+tests+config, etc.",
        )
        return not strict_mode
    elif len(file_types) > MAX_FILE_TYPES_FOR_WARNING:
        # Show info about acceptable combinations but don't fail
        print(f"ℹ️  Multiple file types detected: {', '.join(sorted(file_types))}")
        print("   This combination is acceptable for a single PR")

    return True


def _check_size_and_concerns(args) -> bool:
    """Check file size limits and mixed concerns."""
    num_files, lines_changed, files = _get_git_stats_for_check(args)

    # Check size limits
    size_ok = check_size_limits(num_files, lines_changed, args.strict)

    # Display changed files
    display_changed_files(files)

    # Check for mixed concerns
    concerns_ok = _check_mixed_concerns_display(files, args.strict)

    return size_ok and concerns_ok


def run_scope_checks(args) -> bool:
    """Run all scope checks and return overall result."""
    branch_commit_ok = _check_branch_and_commits(args)
    size_concerns_ok = _check_size_and_concerns(args)

    return branch_commit_ok and size_concerns_ok


def print_final_result(all_passed: bool) -> int:
    """Print final result and return exit code."""
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ PR Scope Check PASSED")
        return 0
    print("❌ PR Scope Check FAILED")
    print("\n💡 Remember the rules:")
    print(f"   • Max {MAX_FILES_CHANGED} files changed")
    print(f"   • Max {MAX_LINES_CHANGED} lines changed")
    print("   • ONE purpose per PR")
    print("   • Single concern (no mixing API + tests + docs)")
    return 1


def main():
    """Main function."""
    print("🔍 Checking PR Scope Compliance")
    print("=" * 50)

    args = parse_arguments()
    all_passed = run_scope_checks(args)
    return print_final_result(all_passed)


if __name__ == "__main__":
    sys.exit(main())
