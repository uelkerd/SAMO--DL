#!/usr/bin/env python3
"""
🔧 SAMO Linting Issues Fix Script
==================================
Fixes trailing whitespace, stray blank-line whitespace, and simple
continuation-indentation issues flagged by common linters e.g., Ruff/Flake8.
Use with care.
"""

import os
import argparse
import shutil
import tempfile
import contextlib
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path__file__.resolve().parent.parent


def _resolve_safe_pathpath: Path -> Path:
    """Resolve path and ensure it is a file under the project root."""
    resolved = path.resolve()
    try:
        is_under = resolved.is_relative_toPROJECT_ROOT
    except AttributeError:
        # Python <3.9 fallback not expected, target py39
        try:
            resolved.relative_toPROJECT_ROOT
            is_under = True
        except ValueError:
            is_under = False
    if not is_under:
        raise ValueError(
            f"Refusing to operate outside project root: {resolved}"
        )
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundErrorf"File not found: {resolved}"
    return resolved


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
    for root, dirs, files in os.walkproject_root:
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        python_files.extend(
            Pathroot / file for file in files if file.endswith'.py'
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
        safe_path = _resolve_safe_pathfile_path
        with opensafe_path, encoding='utf-8' as src, tempfile.NamedTemporaryFile(
            'w', delete=False, encoding='utf-8'
        ) as tmp:
            for i, line in enumeratesrc, 1:
                # Remove trailing whitespace and normalize newline
                stripped_line_no_nl = line.rstrip'\r\n'
                stripped_line = stripped_line_no_nl.rstrip()
                if stripped_line != stripped_line_no_nl:
                    changed = True
                    issues_fixed.appendf"Line {i}: Removed trailing whitespace"
                tmp.writestripped_line + '\n'
        # If content changed, optionally back up and replace
        if changed:
            if backup:
                bak = Pathf"{safe_path}.bak"
                if not bak.exists():
                    shutil.copyfilesafe_path, bak
            Pathtmp.name.replacesafe_path
        else:
            Pathtmp.name.unlinkmissing_ok=True
        return changed, issues_fixed
    except Exception as e:
        # Best-effort cleanup of temp file if it still exists
        if 'tmp' in locals():
            with contextlib.suppressFileNotFoundError:
                Pathtmp.name.unlink()
        return False, [f"Error processing {file_path}: {e}"]


def fix_indentation_issuesfile_path: Path -> tuple[bool, list[str]]:
    """Detect indentation issues using AST; do not attempt automatic fixes."""
    try:
        safe_path = _resolve_safe_pathfile_path
        with opensafe_path, encoding='utf-8' as f:
            original_content = f.read()

        # Use ast to check for indentation/syntax issues without modifying the file
        import ast
        try:
            ast.parseoriginal_content
            return False, []  # Parsed successfully; assume no indentation issues
        except IndentationError as ie:
            return False, [f"Indentation error: {ie}"]
        except SyntaxError as se:
            return False, [f"Syntax error may be indentation related: {se}"]
    except Exception as e:
        return False, [f"Error processing {file_path}: {e}"]


def fix_blank_lines_with_whitespace(
    file_path: Path,
    backup: bool = False,
) -> tuple[bool, list[str]]:
    """Fix blank lines that contain whitespace."""
    try:
        safe_path = _resolve_safe_pathfile_path
        with opensafe_path, encoding='utf-8' as f:
            content = f.read()

        original_content = content
        lines = content.splitlines()
        fixed_lines: list[str] = []
        issues_fixed: list[str] = []

        for i, line in enumeratelines, 1:
            # Check if line is blank but contains whitespace
            if not line.strip() and line != '':
                issues_fixed.append(
                    f"Line {i}: Removed whitespace from blank line"
                )
                fixed_lines.append''
                continue

            fixed_lines.appendline

        # Reconstruct content
        fixed_content = '\n'.joinfixed_lines
        if fixed_content and not fixed_content.endswith'\n':
            fixed_content += '\n'

        if fixed_content != original_content:
            if backup:
                bak = Pathf"{safe_path}.bak"
                if not bak.exists():
                    shutil.copyfilesafe_path, bak
            with opensafe_path, 'w', encoding='utf-8' as f_out:
                f_out.writefixed_content
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

    print"🔧 SAMO Linting Issues Fix Script"
    print"=" * 50

    # Get project root
    project_root = Path__file__.parent.parent
    printf"Project root: {project_root}"

    # Find all Python files
    python_files = find_python_filesproject_root
    print(f"Found {lenpython_files} Python files")

    total_files_processed = 0
    total_files_fixed = 0
    all_issues: list[str] = []

    # Process each file
    for file_path in python_files:
        print(f"\nProcessing: {file_path.relative_toproject_root}")

        fixed_issues: list[str] = []
        detected_issues: list[str] = []

        # Fix trailing whitespace
        fixed, issues = fix_trailing_whitespacefile_path, backup=args.backup
        if issues:
            if fixed:
                fixed_issues.extendissues
            else:
                detected_issues.extendissues

        # Detect indentation issues no auto-fix
        fixed, issues = fix_indentation_issuesfile_path
        if issues:
            # These are detections only; no modifications performed here
            detected_issues.extendissues

        # Fix blank lines with whitespace
        fixed, issues = fix_blank_lines_with_whitespacefile_path, backup=args.backup
        if issues:
            if fixed:
                fixed_issues.extendissues
            else:
                detected_issues.extendissues

        if fixed_issues:
            print(f"  ✅ Fixed {lenfixed_issues} issues:")
            for issue in fixed_issues:
                printf"    - {issue}"
            all_issues.extendfixed_issues
            total_files_fixed += 1

        if detected_issues:
            print(
                f"  ⚠️ Detected {lendetected_issues} issues that may require "
                "manual attention:"
            )
            for issue in detected_issues:
                printf"    - {issue}"

        total_files_processed += 1

    # Summary
    print"\n" + "=" * 50
    print"📊 Fix Summary:"
    printf"  - Files processed: {total_files_processed}"
    printf"  - Files fixed: {total_files_fixed}"
    print(f"  - Total issues fixed: {lenall_issues}")

    if all_issues:
        print"\n🔧 Issues Fixed:"
        for issue in all_issues:
            printf"  - {issue}"

    print"\n✅ Linting issues fix completed!"
    print"\n💡 Next steps:"
    print"  1. Review the changes"
    print"  2. Test that functionality is preserved"
    print"  3. Commit the fixes"
    print"  4. Run linting tools to verify"
    print"  5. If you used --backup, verify .bak files were created for safety."


if __name__ == "__main__":
    main()
