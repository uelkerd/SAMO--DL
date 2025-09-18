#!/usr/bin/env python3
"""
Script to fix missing super().__init__() calls in test classes.
"""

import re
from pathlib import Path


def fix_missing_super_calls(root_dir: Path):
    """Fix missing super() calls in test files."""
    test_files = list(root_dir.glob("tests/**/*.py"))

    for file_path in test_files:
        if file_path.name == "__init__.py":
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Pattern to find test classes with setUp methods that don't call super()
            pattern = r'(class\s+\w+\([^)]*TestCase[^)]*\):[^}]*?def\s+setUp\s*\([^)]*\):\s*"""[^"]*"""\s*)(?!.*super\(\)\.setUp\(\))'

            def add_super_call(match):
                method_start = match.group(1)
                return method_start + "super().setUp()\n        "

            content = re.sub(pattern, add_super_call, content, flags=re.DOTALL)

            # Also handle tearDown methods
            pattern = r'(class\s+\w+\([^)]*TestCase[^)]*\):[^}]*?def\s+tearDown\s*\([^)]*\):\s*"""[^"]*"""\s*)(?!.*super\(\)\.tearDown\(\))'

            def add_super_teardown_call(match):
                method_start = match.group(1)
                return method_start + "super().tearDown()\n        "

            content = re.sub(pattern, add_super_teardown_call, content, flags=re.DOTALL)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed super() calls in {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def fix_specific_patterns():
    """Fix specific known patterns."""
    root_dir = Path(__file__).parent.parent.parent

    # Common test setUp patterns that need super() calls
    patterns = [
        {
            "file": "tests/unit/test_validation.py",
            "old": 'def setUp(self):\n        """Set up test fixtures."""',
            "new": 'def setUp(self):\n        """Set up test fixtures."""\n        super().setUp()',
        },
        {
            "file": "tests/unit/test_emotion_detection.py",
            "old": 'def setUp(self):\n        """Set up test fixtures for emotion detection tests."""',
            "new": 'def setUp(self):\n        """Set up test fixtures for emotion detection tests."""\n        super().setUp()',
        },
        {
            "file": "tests/unit/test_database.py",
            "old": 'def setUp(self):\n        """Set up test database."""',
            "new": 'def setUp(self):\n        """Set up test database."""\n        super().setUp()',
        },
    ]

    for pattern in patterns:
        file_path = root_dir / pattern["file"]
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if pattern["old"] in content and "super().setUp()" not in content:
                    content = content.replace(pattern["old"], pattern["new"])

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Fixed setUp in {file_path}")

            except Exception as e:
                print(f"Error fixing {file_path}: {e}")


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    print("Fixing missing super() calls in test files...")
    fix_missing_super_calls(root_dir)
    fix_specific_patterns()
    print("Done!")
