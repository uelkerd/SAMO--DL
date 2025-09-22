#!/usr/bin/env python3
"""Fortress Guard - Prevent monster PRs and quarantine violations"""

import sys
import subprocess
import os
import argparse


def check_file_limit(base_sha=None, head_sha=None, max_files=None):
    """
    Check if the number of changed files exceeds the limit.
    
    Args:
        base_sha: Base commit SHA (optional)
        head_sha: Head commit SHA (optional) 
        max_files: Maximum allowed files (optional, defaults to env var or 5)
    
    Returns:
        bool: True if within limits, False if exceeded or on error
    """
    # Get max files from parameter, env var, or default
    if max_files is None:
        try:
            max_files = int(os.getenv('MAX_FILES', '5'))
        except ValueError:
            print("Error: MAX_FILES environment variable must be a valid integer", file=sys.stderr)
            return False
    
    # Validate max_files
    if max_files < 1:
        print("Error: MAX_FILES must be at least 1", file=sys.stderr)
        return False
    
    # Build git diff command
    if base_sha and head_sha:
        # Compare specific commits
        cmd = ['/usr/bin/git', 'diff', '--name-only', f'{base_sha}..{head_sha}']
    else:
        # Use staged files (default behavior)
        cmd = ['/usr/bin/git', 'diff', '--cached', '--name-only']
    
    # Run git diff command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, 
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        # Check for git errors
        if result.returncode != 0:
            print(f"Error: git diff failed with return code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"Git error: {result.stderr.strip()}", file=sys.stderr)
            return False
        
        # Parse file list
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        
        # Check file count
        if len(files) > max_files:
            print(f"🚨 FORTRESS BREACH: {len(files)} files changed (max: {max_files})", file=sys.stderr)
            print("🏰 Fortress Rule: Keep PRs micro-sized", file=sys.stderr)
            print("📋 Split your changes into smaller commits", file=sys.stderr)
            if base_sha and head_sha:
                print(f"📊 Comparing commits: {base_sha[:8]}..{head_sha[:8]}", file=sys.stderr)
            return False
        
        # Success
        print(f"✅ File count check passed: {len(files)}/{max_files} files", file=sys.stderr)
        return True
        
    except (OSError, subprocess.SubprocessError) as e:
        print(f"Error: Failed to run git diff: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: Unexpected error in file limit check: {e}", file=sys.stderr)
        return False


def check_quarantine_violations():
    """Block modifications to quarantined files"""
    if not os.path.exists('LEGACY_TRACKING.md'):
        return True

    try:
        with open('LEGACY_TRACKING.md', 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract quarantined files (lines with #legacy-quarantined tag)
        quarantined = []
        for line in content.split('\n'):
            if line.startswith('- [ ]') and '#legacy-quarantined' in line:
                # Extract file path from markdown:
                # "- [ ] path/to/file.py (...) #legacy-quarantined"
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
        normalized_quarantined = {
            os.path.normpath(path) for path in quarantined
        }
        normalized_staged = {os.path.normpath(f) for f in staged_files}
        violations = sorted(normalized_staged & normalized_quarantined)
        if violations:
            print("🚨 QUARANTINED FILE VIOLATION:", file=sys.stderr)
            for f in violations:
                print(f"   🔴 {f}", file=sys.stderr)
            print("🏰 These files are in quarantine until migration", file=sys.stderr)
            return False

    except Exception as e:
        print(f"Error: Could not check quarantine: {e}", file=sys.stderr)
        # Fail-closed by default, but allow fail-open via environment variable
        fail_open = os.getenv('FORTRESS_FAIL_OPEN', '').lower() in ('1', 'true', 'yes')
        return True if fail_open else False


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Fortress Guard - Prevent monster PRs and quarantine violations"
    )
    parser.add_argument(
        '--base', 
        help='Base commit SHA for comparison (requires --head)'
    )
    parser.add_argument(
        '--head', 
        help='Head commit SHA for comparison (requires --base)'
    )
    parser.add_argument(
        '--max-files', 
        type=int, 
        help='Maximum allowed files (overrides MAX_FILES env var)'
    )
    parser.add_argument(
        '--skip-quarantine', 
        action='store_true', 
        help='Skip quarantine file checks'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.base and not args.head) or (args.head and not args.base):
        print("Error: --base and --head must be provided together", file=sys.stderr)
        sys.exit(1)
    
    success = True
    
    # Check file limit
    if not check_file_limit(args.base, args.head, args.max_files):
        success = False
    
    # Check quarantine violations (unless skipped)
    if not args.skip_quarantine and not check_quarantine_violations():
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
