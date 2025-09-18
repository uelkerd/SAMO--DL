#!/usr/bin/env python3
"""
Script to fix methods decorated with @staticmethod that incorrectly have 'self' parameters.
"""

import re
from pathlib import Path


def fix_staticmethod_parameters(file_path: Path):
    """Fix @staticmethod methods that incorrectly have 'self' parameters."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern to find @staticmethod followed by method definition with self parameter
        # This pattern matches:
        # @staticmethod
        # def method_name(self, other_params):
        pattern = r"(@staticmethod\s*\n\s*def\s+\w+\s*\(\s*)self\s*,?\s*"

        def fix_self_parameter(match):
            """Remove 'self' parameter from @staticmethod methods."""
            prefix = match.group(1)
            # Remove 'self,' or just 'self' if it's the only parameter
            return prefix

        # Fix methods with self as first parameter
        content = re.sub(pattern, fix_self_parameter, content, flags=re.MULTILINE)

        # Also handle case where self is the only parameter
        pattern2 = r"(@staticmethod\s*\n\s*def\s+\w+\s*\(\s*)self\s*(\)\s*:)"
        content = re.sub(pattern2, r"\1\2", content, flags=re.MULTILINE)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed @staticmethod parameters in {file_path}")
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False


def process_files(root_dir: Path):
    """Process Python files to fix @staticmethod parameters."""
    fixed_count = 0

    for pattern in ["**/*.py"]:
        for file_path in root_dir.glob(pattern):
            if file_path.name == "__init__.py":
                continue

            # Skip certain directories
            if any(part in str(file_path) for part in [".git", "__pycache__", ".pytest_cache"]):
                continue

            if fix_staticmethod_parameters(file_path):
                fixed_count += 1

    return fixed_count


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    print("Fixing @staticmethod methods with incorrect 'self' parameters...")

    fixed_count = process_files(root_dir)
    print(f"Fixed {fixed_count} files")
    print("Done!")
