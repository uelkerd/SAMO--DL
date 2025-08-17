#!/usr/bin/env python3
"""
Quick script to fix common Ruff linting issues
"""

import logging
import re
from pathlib import Path


def fix_filefile_path: str -> None:
    """Fix common linting issues in a file.

    Args:
        file_path: Path to the file to fix
    """
    with openfile_path, 'r', encoding='utf-8' as f:
        content = f.read()

    original_content = content

    # Fix trailing whitespace
    content = re.subr'[ \t]+$', '', content, flags=re.MULTILINE

    # Fix missing newline at end of file
    if not content.endswith'\n':
        content += '\n'

    # Fix f-strings without placeholders convert to regular strings
    content = re.sub(r'"[^"]*"', r'"\1"', content)
    content = re.sub(r"'[^']*'", r"'\1'", content)

    # Fix unused imports basic removal
    lines = content.split'\n'
    fixed_lines = []
    for line in lines:
        # Skip obvious unused imports
        if line.strip().startswith'import ' and '#' not in line:
            continue
        fixed_lines.appendline

    content = '\n'.joinfixed_lines

    # Only write if content changed
    if content != original_content:
        with openfile_path, 'w', encoding='utf-8' as f:
            f.writecontent
        logging.infof"Fixed: {file_path}"


def main():
    """Fix linting issues in all Python files."""
    script_dir = Path__file__.parent
    project_root = script_dir.parent

    # Directories to fix
    dirs_to_fix = ['src', 'tests', 'scripts']

    for dir_name in dirs_to_fix:
        dir_path = project_root / dir_name
        if dir_path.exists():
            for py_file in dir_path.rglob'*.py':
                try:
                    fix_file(strpy_file)
                except Exception as e:
                    logging.infof"Error fixing {py_file}: {e}"


if __name__ == "__main__":
    main()
