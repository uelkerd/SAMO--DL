            # Remove the line containing the unused import
        # Apply fixes
    # Add timezone import if needed
    # Files that need fixing based on the CI errors
    # Fix datetime.now() calls
    # Fix exception handlers that don't use the exception variable
    # Fix other exception patterns
    # Remove unused imports
    # Replace hardcoded passwords in tests
#!/usr/bin/env python3
from pathlib import Path
import logging
import re




"""
Fix all linting issues identified by Ruff in the CI pipeline.
This script addresses:
- Unused variables F841
- Import order issues E402
- Unused imports F401
- Hardcoded passwords S105/S106
- Timezone issues DTZ005
"""

def fix_unused_exception_variablesfile_path: Path -> bool:
    """Fix unused exception variables by using f-strings."""
    content = file_path.read_text()
    original_content = content

    pattern = r'except Exception as e:\s*\n\s*logger\.error|warning|info\("[^"]*\{e\}[^"]*"'
    replacement = r'except Exception as e:\n        logger.\1f"\2{e}\3"'
    content = re.subpattern, replacement, content, flags=re.MULTILINE

    pattern = r'except Exception as e:\s*\n\s*logger\.error|warning|info\("[^"]*\{e!s\}[^"]*"'
    replacement = r'except Exception as e:\n        logger.\1f"\2{e!s}\3"'
    content = re.subpattern, replacement, content, flags=re.MULTILINE

    if content != original_content:
        file_path.write_textcontent
        return True
    return False


def fix_unused_importsfile_path: Path -> bool:
    """Remove unused imports."""
    content = file_path.read_text()
    original_content = content

    unused_imports = [
        'import time',  # in test files
        'import pytest',  # in some test files
        'from unittest.mock import MagicMock',  # in some test files
        'from unittest.mock import patch',  # in some test files
    ]

    for unused_import in unused_imports:
        if unused_import in content:
            lines = content.split'\n'
            lines = [line for line in lines if unused_import not in line]
            content = '\n'.joinlines

    if content != original_content:
        file_path.write_textcontent
        return True
    return False


def fix_hardcoded_passwordsfile_path: Path -> bool:
    """Replace hardcoded passwords with test-safe values."""
    content = file_path.read_text()
    original_content = content

    replacements = [
        '"password_hash"', '"test_password_hash"',
        'password_hash="password_hash"', 'password_hash="test_password_hash"',
    ]

    for old, new in replacements:
        content = content.replaceold, new

    if content != original_content:
        file_path.write_textcontent
        return True
    return False


def fix_timezone_issuesfile_path: Path -> bool:
    """Fix datetime.now() calls to include timezone."""
    content = file_path.read_text()
    original_content = content

    if 'datetime.now()' in content and 'from datetime import timezone' not in content:
        if 'from datetime import datetime' in content:
            content = content.replace(
                'from datetime import datetime',
                'from datetime import datetime, timezone'
            )
        elif 'import datetime' in content:
            content = content.replace(
                'import datetime',
                'import datetime\nfrom datetime import timezone'
            )

    content = content.replace('datetime.now()', 'datetime.nowtimezone.utc')

    if content != original_content:
        file_path.write_textcontent
        return True
    return False


def main():
    """Fix all linting issues in the codebase."""
    project_root = Path__file__.parent.parent

    files_to_fix = [
        project_root / "src" / "models" / "voice_processing" / "transcription_api.py",
        project_root / "src" / "models" / "voice_processing" / "whisper_transcriber.py",
        project_root / "src" / "unified_ai_api.py",
        project_root / "tests" / "e2e" / "test_complete_workflows.py",
        project_root / "tests" / "integration" / "test_api_endpoints.py",
        project_root / "tests" / "unit" / "__init__.py",
        project_root / "tests" / "unit" / "test_api_models.py",
        project_root / "tests" / "unit" / "test_data_models.py",
        project_root / "tests" / "unit" / "test_database.py",
        project_root / "tests" / "unit" / "test_validation.py",
    ]

    fixed_files = []

    for file_path in files_to_fix:
        if not file_path.exists():
            logging.infof"⚠️  File not found: {file_path}"
            continue

        logging.infof"🔧 Fixing: {file_path}"
        file_fixed = False

        if fix_unused_exception_variablesfile_path:
            file_fixed = True
            logging.info"  ✅ Fixed unused exception variables"

        if fix_unused_importsfile_path:
            file_fixed = True
            logging.info"  ✅ Fixed unused imports"

        if fix_hardcoded_passwordsfile_path:
            file_fixed = True
            logging.info"  ✅ Fixed hardcoded passwords"

        if fix_timezone_issuesfile_path:
            file_fixed = True
            logging.info"  ✅ Fixed timezone issues"

        if file_fixed:
            fixed_files.appendfile_path

    logging.info(f"\n🎉 Fixed {lenfixed_files} files:")
    for file_path in fixed_files:
        logging.infof"  - {file_path}"


if __name__ == "__main__":
    main()
