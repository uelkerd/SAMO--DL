#!/usr/bin/env python3
"""
🔧 SAMO Linting Issues Fix Script
==================================
Fixes trailing whitespace and indentation issues identified by DeepSource.
"""

import os
from pathlib import Path
from typing import List, Tuple

def find_python_files(project_root: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'build', 'dist'}]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    return python_files

def fix_trailing_whitespace(file_path: Path) -> Tuple[bool, List[str]]:
    """Fix trailing whitespace in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        lines = content.splitlines()
        fixed_lines = []
        issues_fixed = []

        for i, line in enumerate(lines, 1):
            # Remove trailing whitespace
            if line.rstrip() != line:
                original_line = line
                line = line.rstrip()
                issues_fixed.append(f"Line {i}: Removed trailing whitespace")

            fixed_lines.append(line)

        # Reconstruct content with proper line endings
        fixed_content = '\n'.join(fixed_lines)
        if fixed_content and not fixed_content.endswith('\n'):
            fixed_content += '\n'

        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, issues_fixed

        return False, []

    except Exception as e:
        return False, [f"Error processing file: {e}"]

def fix_indentation_issues(file_path: Path) -> Tuple[bool, List[str]]:
    """Fix indentation issues in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        lines = content.splitlines()
        fixed_lines = []
        issues_fixed = []

        for i, line in enumerate(lines, 1):
            # Fix visually indented lines with same indent as next logical line
            # This is a simplified fix - in practice, you'd need more context
            if i < len(lines) - 1:
                current_indent = len(line) - len(line.lstrip())
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())

                # If current line is continuation and next line has same indent
                if (line.strip().endswith('and') or line.strip().endswith('or')) and current_indent == next_indent:
                    # Add proper indentation for continuation
                    line = ' ' * (current_indent + 4) + line.strip()
                    issues_fixed.append(f"Line {i}: Fixed continuation indentation")

            fixed_lines.append(line)

        # Reconstruct content
        fixed_content = '\n'.join(fixed_lines)
        if fixed_content and not fixed_content.endswith('\n'):
            fixed_content += '\n'

        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, issues_fixed

        return False, []

    except Exception as e:
        return False, [f"Error processing file: {e}"]

def fix_blank_lines_with_whitespace(file_path: Path) -> Tuple[bool, List[str]]:
    """Fix blank lines that contain whitespace."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        lines = content.splitlines()
        fixed_lines = []
        issues_fixed = []

        for i, line in enumerate(lines, 1):
            # Check if line is blank but contains whitespace
            if not line.strip() and line != '':
                issues_fixed.append(f"Line {i}: Removed whitespace from blank line")
                line = ''

            fixed_lines.append(line)

        # Reconstruct content
        fixed_content = '\n'.join(fixed_lines)
        if fixed_content and not fixed_content.endswith('\n'):
            fixed_content += '\n'

        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, issues_fixed

        return False, []

    except Exception as e:
        return False, [f"Error processing file: {e}"]

def main():
    """Main function to fix all linting issues."""
    print("🔧 SAMO Linting Issues Fix Script")
    print("=" * 50)

    # Get project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    # Find all Python files
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files")

    total_files_processed = 0
    total_files_fixed = 0
    all_issues = []

    # Process each file
    for file_path in python_files:
        print(f"\nProcessing: {file_path.relative_to(project_root)}")

        file_fixed = False
        file_issues = []

        # Fix trailing whitespace
        fixed, issues = fix_trailing_whitespace(file_path)
        if fixed:
            file_fixed = True
            file_issues.extend(issues)

        # Fix indentation issues
        fixed, issues = fix_indentation_issues(file_path)
        if fixed:
            file_fixed = True
            file_issues.extend(issues)

        # Fix blank lines with whitespace
        fixed, issues = fix_blank_lines_with_whitespace(file_path)
        if fixed:
            file_fixed = True
            file_issues.extend(issues)

        if file_issues:
            print(f"  ✅ Fixed {len(file_issues)} issues:")
            for issue in file_issues:
                print(f"    - {issue}")
            all_issues.extend(file_issues)
            total_files_fixed += 1

        total_files_processed += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 Fix Summary:")
    print(f"  - Files processed: {total_files_processed}")
    print(f"  - Files fixed: {total_files_fixed}")
    print(f"  - Total issues fixed: {len(all_issues)}")

    if all_issues:
        print("\n🔧 Issues Fixed:")
        for issue in all_issues:
            print(f"  - {issue}")

    print("\n✅ Linting issues fix completed!")
    print("\n💡 Next steps:")
    print("  1. Review the changes")
    print("  2. Test that functionality is preserved")
    print("  3. Commit the fixes")
    print("  4. Run linting tools to verify")

if __name__ == "__main__":
    main()
