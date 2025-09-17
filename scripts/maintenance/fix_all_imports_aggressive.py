            # Add all needed imports
        # Add imports right after the first line (usually shebang or docstring)
        # Skip obvious unused imports
    # Check for common undefined names
    # Check what imports are needed
    # Directories to fix
    # Find the first import line or add at the beginning
    # Fix f-strings without placeholders (convert to regular strings)
    # Fix missing newline at end of file
    # Fix trailing whitespace
    # Fix unused imports (remove obvious ones)
    # If no imports needed, return early
    # If we didn't add imports yet, add them at the very beginning
    # Only write if content changed
    # Only write if content changed
    # Split into lines
#!/usr/bin/env python3
from pathlib import Path
import logging
import re




"""
Aggressive script to fix ALL missing imports across the codebase.
This addresses the extensive linting errors causing CircleCI failures.
"""

def fix_file_imports_aggressive(file_path: str) -> bool:
    """Aggressively fix missing imports in a file."""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    original_content = content

    needed_imports = set()

    if 'sys.' in content or 'sys.path' in content or 'sys.exit' in content:
        needed_imports.add('import sys')

    if 'os.' in content or 'os.path' in content or 'os.environ' in content:
        needed_imports.add('import os')

    if 'np.' in content or 'np.ndarray' in content or 'np.array' in content:
        needed_imports.add('import numpy as np')

    if 'json.' in content or 'json.dumps' in content or 'json.loads' in content:
        needed_imports.add('import json')

    if 'traceback.' in content:
        needed_imports.add('import traceback')

    if 'time.' in content and 'import time' not in content:
        needed_imports.add('import time')

    if 'datetime.' in content and 'import datetime' not in content:
        needed_imports.add('import datetime')

    if not needed_imports:
        return False

    lines = content.split('\n')
    new_lines = []

    import_added = False

    for i, line in enumerate(lines):
        if i == 0 and not import_added:
            for imp in sorted(needed_imports):
                new_lines.append(imp)
            new_lines.append('')  # Empty line after imports
            import_added = True

        new_lines.append(line)

    if not import_added:
        new_lines = []
        for imp in sorted(needed_imports):
            new_lines.append(imp)
        new_lines.append('')  # Empty line after imports
        new_lines.extend(lines)

    content = '\n'.join(new_lines)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info("Fixed imports in: {file_path}")
        return True

    return False

def fix_common_issues_aggressive(file_path: str) -> bool:
    """Aggressively fix common linting issues."""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    original_content = content

    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

    if not content.endswith('\n'):
        content += '\n'

    content = re.sub(r'"([^"]*)"', r'"\1"', content)
    content = re.sub(r"'([^']*)'", r"'\1'", content)

    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if any(unused in line for unused in [
        ]):
            continue
        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info("Fixed common issues in: {file_path}")
        return True

    return False

def main():
    """Fix all import and linting issues aggressively."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    dirs_to_fix = ['src', 'tests', 'scripts']

    total_fixed = 0

    for dir_name in dirs_to_fix:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob('*.py'):
            try:
                fixed_imports = fix_file_imports_aggressive(str(py_file))
                fixed_common = fix_common_issues_aggressive(str(py_file))
                if fixed_imports or fixed_common:
                    total_fixed += 1
            except Exception:
                logging.info("Error fixing {py_file}: {e}")

    logging.info("\n✅ Fixed {total_fixed} files")

if __name__ == "__main__":
    main()
