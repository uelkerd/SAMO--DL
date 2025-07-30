import json
import numpy as np
import os
import sys
import traceback

#!/usr/bin/env python3
"""
Aggressive script to fix ALL missing imports across the codebase.
This addresses the extensive linting errors causing CircleCI failures.
"""

import os
import re
from pathlib import Path

def fix_file_imports_aggressive(file_path: str) -> bool:
    """Aggressively fix missing imports in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Check what imports are needed
    needed_imports = set()

    # Check for common undefined names
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

    # If no imports needed, return early
    if not needed_imports:
        return False

    # Split into lines
    lines = content.split('\n')
    new_lines = []

    # Find the first import line or add at the beginning
    import_added = False

    for i, line in enumerate(lines):
        # Add imports right after the first line (usually shebang or docstring)
        if i == 0 and not import_added:
            # Add all needed imports
            for imp in sorted(needed_imports):
                new_lines.append(imp)
            new_lines.append('')  # Empty line after imports
            import_added = True

        new_lines.append(line)

    # If we didn't add imports yet, add them at the very beginning
    if not import_added:
        new_lines = []
        for imp in sorted(needed_imports):
            new_lines.append(imp)
        new_lines.append('')  # Empty line after imports
        new_lines.extend(lines)

    content = '\n'.join(new_lines)

    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed imports in: {file_path}")
        return True

    return False

def fix_common_issues_aggressive(file_path: str) -> bool:
    """Aggressively fix common linting issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

    # Fix missing newline at end of file
    if not content.endswith('\n'):
        content += '\n'

    # Fix f-strings without placeholders (convert to regular strings)
    content = re.sub(r'"([^"]*)"', r'"\1"', content)
    content = re.sub(r"'([^']*)'", r"'\1'", content)

    # Fix unused imports (remove obvious ones)
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Skip obvious unused imports
        if any(unused in line for unused in [
        ]):
            continue
        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed common issues in: {file_path}")
        return True

    return False

def main():
    """Fix all import and linting issues aggressively."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Directories to fix
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
            except Exception as e:
                print("Error fixing {py_file}: {e}")

    print("\n✅ Fixed {total_fixed} files")

if __name__ == "__main__":
    main()
