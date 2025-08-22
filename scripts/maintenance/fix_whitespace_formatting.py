#!/usr/bin/env python3
"""
Whitespace and Formatting Fixer
Fixes only formatting issues without changing code logic.
"""

import os
import re
from pathlib import Path


def fix_trailing_whitespace(content):
    """Remove trailing whitespace from lines."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Remove trailing whitespace but preserve empty lines
        if line.strip():
            fixed_lines.append(line.rstrip())
        else:
            fixed_lines.append('')

    return '\n'.join(fixed_lines)


def fix_missing_newlines(content):
    """Ensure file ends with newline."""
    if not content.endswith('\n'):
        content += '\n'
    return content


def fix_file(file_path):
    """Fix whitespace and formatting in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes in order
        content = fix_trailing_whitespace(content)
        content = fix_missing_newlines(content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix whitespace and formatting issues."""
    print("🧹 Whitespace and Formatting Fixer")
    print("=" * 40)

    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files")

    fixed_count = 0

    for py_file in python_files:
        try:
            if fix_file(py_file):
                fixed_count += 1
                print(f"  ✅ Fixed {py_file}")
        except Exception as e:
            print(f"  ❌ Error processing {py_file}: {e}")

    print(f"\n📋 SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_count}")

    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
        print("💡 Run the quality check again to verify fixes.")
    else:
        print(f"\n⚠️  No files were fixed. All files may already be clean.")


if __name__ == "__main__":
    main()
