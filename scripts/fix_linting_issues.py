#!/usr/bin/env python3
"""
🔧 SAMO Linting Issues Fix Script
==================================
Fixes trailing whitespace, stray blank-line whitespace, and simple continuation-indentation issues flagged by common linters (e.g., Ruff/Flake8). Use with care.
"""

import os
import argparse
import shutil
import tempfile
import contextlib
from pathlib import Path
from typing import Optional


def find_python_files(
    project_root: Path,
    excluded_dirs: Optional[set[str]] = None,
) -> list[Path]:
    """Find all Python files in the project, skipping excluded directories."""
    if excluded_dirs is None:
        excluded_dirs = {
            '.git', '__pycache__', '.venv', 'venv', 'node_modules', 'build', 'dist',
            '.mypy_cache', '.pytest_cache', '.cache', '.coverage', '.eggs', '.tox',
            '.idea', '.vscode', '.DS_Store'
        }

    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        python_files.extend(
            Path(root) / file for file in files if file.endswith('.py')
        )

    return python_files


def fix_trailing_whitespace(
    file_path: Path,
    backup: bool = False,
) -> tuple[bool, list[str]]:
    """Fix trailing whitespace in a file, processing line by line for efficiency."""
    changed = False
    issues_fixed: list[str] = []
    try:
        with open(file_path, encoding='utf-8') as src, tempfile.NamedTemporaryFile(
            'w', delete=False, encoding='utf-8'
        ) as tmp:
            for i, line in enumerate(src, 1):
                # Remove trailing whitespace (including tabs/spaces) and normalize newline
                stripped_line_no_nl = line.rstrip('\r\n')
                stripped_line = stripped_line_no_nl.rstrip()
                if stripped_line != stripped_line_no_nl:
                    changed = True
                    issues_fixed.append(f"Line {i}: Removed trailing whitespace")
                tmp.write(stripped_line + '\n')
        # If content changed, optionally back up and replace
        if changed:
            if backup:
                shutil.copyfile(file_path, f"{file_path}.bak")
            Path(tmp.name).replace(file_path)
        else:
            Path(tmp.name).unlink(missing_ok=True)
        return changed, issues_fixed
    except Exception as e:
        # Best-effort cleanup of temp file if it still exists
        if 'tmp' in locals():
            with contextlib.suppress(FileNotFoundError):
                Path(tmp.name).unlink()
        return False, [f"Error processing {file_path}: {e}"]


def fix_indentation_issues(file_path: Path) -> tuple[bool, list[str]]:
    """Detect indentation issues using AST; do not attempt automatic fixes."""
    try:
        with open(file_path, encoding='utf-8') as f:
            original_content = f.read()

        # Use ast to check for indentation/syntax issues without modifying the file
        import ast
        try:
            ast.parse(original_content)
            return False, []  # Parsed successfully; assume no indentation issues
        except IndentationError as ie:
            return False, [f"Indentation error: {ie}"]
        except SyntaxError as se:
            return False, [f"Syntax error (may be indentation related): {se}"]
    except Exception as e:
        return False, [f"Error processing {file_path}: {e}"]


def fix_blank_lines_with_whitespace(
    file_path: Path,
    backup: bool = False,
) -> tuple[bool, list[str]]:
    """Fix blank lines that contain whitespace."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        original_content = content
        lines = content.splitlines()
        fixed_lines: list[str] = []
        issues_fixed: list[str] = []

        for i, line in enumerate(lines, 1):
            # Check if line is blank but contains whitespace
            if not line.strip() and line != '':
                issues_fixed.append(
                    f"Line {i}: Removed whitespace from blank line"
                )
                fixed_lines.append('')
                continue

            fixed_lines.append(line)

        # Reconstruct content
        fixed_content = '\n'.join(fixed_lines)
        if fixed_content and not fixed_content.endswith('\n'):
            fixed_content += '\n'

        if fixed_content != original_content:
            if backup:
                shutil.copyfile(file_path, f"{file_path}.bak")
            with open(file_path, 'w', encoding='utf-8') as f_out:
                f_out.write(fixed_content)
            return True, issues_fixed

        return False, []

    except Exception as e:
        return False, [f"Error processing {file_path}: {e}"]


def main():
    """Main function to fix all linting issues."""
    parser = argparse.ArgumentParser(
        description="Fix linting issues in files."
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of files before modifying them.",
    )
    args = parser.parse_args()

    # Warn user if not backing up
    if not args.backup:
        print(
            "⚠️ WARNING: No backups will be created before modifying files. "
            "This may result in accidental data loss."
        )
        print(
            "   Use the --backup option to create .bak files before changes are made.\n"
        )

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
    all_issues: list[str] = []

    # Process each file
    for file_path in python_files:
        print(f"\nProcessing: {file_path.relative_to(project_root)}")

        fixed_issues: list[str] = []
        detected_issues: list[str] = []

        # Fix trailing whitespace
        fixed, issues = fix_trailing_whitespace(file_path, backup=args.backup)
        if issues:
            if fixed:
                fixed_issues.extend(issues)
            else:
                detected_issues.extend(issues)

        # Detect indentation issues (no auto-fix)
        fixed, issues = fix_indentation_issues(file_path)
        if issues:
            # These are detections only; no modifications performed here
            detected_issues.extend(issues)

        # Fix blank lines with whitespace
        fixed, issues = fix_blank_lines_with_whitespace(file_path, backup=args.backup)
        if issues:
            if fixed:
                fixed_issues.extend(issues)
            else:
                detected_issues.extend(issues)

        if fixed_issues:
            print(f"  ✅ Fixed {len(fixed_issues)} issues:")
            for issue in fixed_issues:
                print(f"    - {issue}")
            all_issues.extend(fixed_issues)
            total_files_fixed += 1

        if detected_issues:
            print(
                f"  ⚠️ Detected {len(detected_issues)} issues that may require "
                f"manual attention:"
            )
            for issue in detected_issues:
                print(f"    - {issue}")

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
    print("  5. If you used --backup, verify .bak files were created for safety.")


if __name__ == "__main__":
    main()
