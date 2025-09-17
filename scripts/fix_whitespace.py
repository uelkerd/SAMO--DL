#!/usr/bin/env python3
"""
Script to fix whitespace issues in Python files
"""

import os
import re
from pathlib import Path

def fix_whitespace_issues(file_path):
    """Fix whitespace issues in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix 1: Remove trailing whitespace from lines
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Remove trailing whitespace
            fixed_line = line.rstrip()
            fixed_lines.append(fixed_line)

        content = '\n'.join(fixed_lines)

        # Fix 2: Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'

        # Fix 3: Remove blank lines that contain only whitespace
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            if line.strip() == '':
                # Empty line - keep it
                fixed_lines.append('')
            else:
                # Non-empty line - keep it
                fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed whitespace: {file_path}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix whitespace issues in all Python files."""
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    fixed_count = 0
    for file_path in python_files:
        if fix_whitespace_issues(file_path):
            fixed_count += 1

    print(f"\nFixed whitespace in {fixed_count} files")

if __name__ == "__main__":
    main()
