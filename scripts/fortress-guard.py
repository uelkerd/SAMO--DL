#!/usr/bin/env python3
"""Fortress Guard - Prevent monster PRs and quarantine violations"""

import sys
import subprocess
import os


def check_file_limit():
    """Kill PR if >5 files changed"""
    try:
        result = subprocess.run(
            ['/usr/bin/git', 'diff', '--cached', '--name-only'],
            capture_output=True, text=True,
            check=True)
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []

        if len(files) > 5:
            print(f"🚨 FORTRESS BREACH: {len(files)} files changed (max: 5)")
            print("🏰 Fortress Rule: Keep PRs micro-sized")
            print("📋 Split your changes into smaller commits")
            return False
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Error running git diff: {e}", file=sys.stderr)
        return False
    return True


def check_quarantine_violations():
    """Block modifications to quarantined files"""
    if not os.path.exists('LEGACY_TRACKING.md'):
        return True

    try:
        with open('LEGACY_TRACKING.md', 'r') as f:
            content = f.read()

        # Extract quarantined files (lines with #legacy-quarantined tag)
        quarantined = []
        for line in content.split('\n'):
            if line.startswith('- [ ]') and '#legacy-quarantined' in line:
                # Extract file path from markdown: "- [ ] path/to/file.py (...) #legacy-quarantined"
                parts = line.split(']', 1)
                if len(parts) > 1:
                    file_path = parts[1].split('(')[0].strip()
                    quarantined.append(file_path)

        # Check if any staged files are quarantined
        result = subprocess.run(
            ['/usr/bin/git', 'diff', '--cached', '--name-only'],
            capture_output=True, text=True,
            check=True)
        staged_files = (
            result.stdout.strip().split('\n') if result.stdout.strip() else []
        )

        # Normalize file paths for comparison
        normalized_quarantined = {os.path.normpath(path) for path in quarantined}
        if violations := [f for f in staged_files if os.path.normpath(f) in normalized_quarantined]:
            print("🚨 QUARANTINED FILE VIOLATION:")
            for f in violations:
                print(f"   🔴 {f}")
            print("🏰 These files are in quarantine until migration")
            return False

    except Exception as e:
        print(f"Warning: Could not check quarantine: {e}")

    return True


if __name__ == "__main__":
    success = True

    if not check_file_limit():
        success = False

    if not check_quarantine_violations():
        success = False

    sys.exit(0 if success else 1)
