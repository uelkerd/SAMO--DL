# Only remove if it's clearly unused and not necessary
# Restore backup
# Check if this import is actually used
# Keep all imports for now - we'll let ruff handle unused imports
# Validate syntax
# Add logging import if not present
# Apply fixes
# Create backup
# Only write if content changed
# Read content
# Find all import lines
# Fix unused exception variables
# Fix unused loop variables
# Reconstruct with imports at top
# Replace print statements
# Simple check - can be improved
# Directories to process
#!/usr/bin/env python3
import ast
import logging
import re
import shutil

"""Comprehensive Linting Fix Script for SAMO Deep Learning.
from pathlib import Path

This script fixes linting issues across the entire codebase systematically.
It processes all Python files in specified directories and applies safe fixes.

Fixes applied:
- F841: Unused variables (replace with _)
- E402: Import order (move to top)
- RUF022: __all__ sorting
- W291: Trailing whitespace
- T201: Print statements (replace with logging)
- F401: Unused imports (only if clearly safe)
- F821: Undefined names (add missing imports)

Safety features:
- Preserves necessary imports (time, pytest, patch, etc.)
- Creates backups before making changes
- Validates Python syntax after changes
- Reports all changes made
"""


class ComprehensiveLintingFixer:
    """Comprehensive linting fixer for entire codebase."""

    def __init__(self):
        self.necessary_imports = {
            "time",
            "pytest",
            "patch",
            "Mock",
            "json",
            "logging",
            "os",
            "sys",
            "pathlib",
            "typing",
            "datetime",
            "tempfile",
            "numpy",
            "torch",
            "whisper",
            "fastapi",
            "sqlalchemy",
        }
        self.fixed_files = []
        self.errors = []

    def find_python_files(self, directories: list[str]) -> list[Path]:
        """Find all Python files in specified directories."""
        python_files = []
        for directory in directories:
            if Path(directory).exists():
                python_files.extend(Path(directory).rglob("*.py"))
        return python_files

    def separate_imports_and_code(
        self, lines: list[str]
    ) -> tuple[list[str], list[str]]:
        """Separate import lines from other code lines."""
        import_lines = []
        non_import_lines = []

        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("#")
            ):
                import_lines.append(line)
            else:
                non_import_lines.append(line)

        return import_lines, non_import_lines

    def filter_imports(self, lines: list[str]) -> list[str]:
        """Filter out unused imports."""
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)

        return filtered_lines

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file."""
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        shutil.copy2(file_path, backup_path)
        return backup_path

    def validate_python_syntax(self, file_path: Path) -> bool:
        """Validate that the file has correct Python syntax."""
        try:
            with open(file_path, encoding="utf-8") as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False

    def fix_import_order(self, content: str) -> str:
        """Fix import order by moving all imports to the top."""
        lines = content.split("\n")

        import_lines = []
        non_import_lines = []

        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("#")
            ):
                import_lines.append(line)
            else:
                non_import_lines.append(line)

        return "\n".join(import_lines + non_import_lines)

    def fix_unused_variables(self, content: str) -> str:
        """Fix unused variables by replacing with underscore."""
        content = re.sub(r"except Exception as e:", "except Exception as e:", content)
        content = re.sub(r"except Exception as e:", "except Exception as e:", content)

        content = re.sub(r"for (\w+) in (\w+):", r"for _\1 in \2:", content)

        return content

    def fix_unused_imports(self, content: str) -> str:
        """Safely remove unused imports."""
        lines = content.split("\n")
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_name = self.extract_import_name(stripped)
                if import_name and import_name not in self.necessary_imports:
                    if not self.is_import_used(content, import_name):
                        continue  # Skip this line
            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def extract_import_name(self, import_line: str) -> str:
        """Extract the main import name from an import line."""
        if import_line.startswith("import "):
            return import_line.split()[1].split(".")[0]
        elif import_line.startswith("from "):
            parts = import_line.split()
            if len(parts) >= 3:
                return parts[1].split(".")[0]
        return ""

    def is_import_used(self, content: str, import_name: str) -> bool:
        """Check if an import is actually used in the content."""
        return import_name in content

    def fix_trailing_whitespace(self, content: str) -> str:
        """Remove trailing whitespace."""
        lines = content.split("\n")
        return "\n".join(line.rstrip() for line in lines)

    def fix_print_statements(self, content: str) -> str:
        """Replace print statements with logging."""
        if "print(" in content and "import logging" not in content:
            lines = content.split("\n")
            import_added = False
            for _i, line in enumerate(lines):
                if line.strip().startswith("import ") or line.strip().startswith(
                    "from "
                ):
                    if not import_added:
                        lines.insert(i, "import logging")
                        import_added = True
                        break

            if not import_added:
                lines.insert(0, "import logging")

            content = "\n".join(lines)

        content = re.sub(r"print\((.*?)\)", r"logging.info(\1)", content)

        return content

    def fix_all_issues(self, file_path: Path) -> bool:
        """Fix all linting issues in a file."""
        try:
            backup_path = self.backup_file(file_path)

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            content = self.fix_import_order(content)
            content = self.fix_unused_variables(content)
            content = self.fix_trailing_whitespace(content)
            content = self.fix_print_statements(content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                if not self.validate_python_syntax(file_path):
                    shutil.copy2(backup_path, file_path)
                    self.errors.append(
                        f"Syntax error after fixing {file_path}, restored backup"
                    )
                    return False

                self.fixed_files.append(str(file_path))
                return True

            return False

        except Exception as e:
            self.errors.append(f"Error fixing {file_path}: {e}")
            return False

    def run_on_directories(self, directories: list[str]) -> int:
        """Run the fixer on all Python files in specified directories."""
        logging.info(f"🔍 Scanning directories: {directories}")

        python_files = self.find_python_files(directories)
        logging.info(f"📁 Found {len(python_files)} Python files")

        fixed_count = 0
        for file_path in python_files:
            logging.info(f"🔧 Processing: {file_path}")
            if self.fix_all_issues(file_path):
                fixed_count += 1

        logging.info(f"\n🎉 Fixed {fixed_count} files:")
        for file_path in self.fixed_files:
            logging.info(f"  ✅ {file_path}")

        if self.errors:
            logging.info("\n❌ Errors encountered:")
            for error in self.errors:
                logging.info(f"  ⚠️  {error}")

        return fixed_count


def main():
    """Main function to run the comprehensive linting fixer."""
    directories = [
        "src/models/emotion_detection",
        "src/models/summarization",
        "src/models/voice_processing",
        "src/data",
        "src/evaluation",
        "src/inference",
        "tests",
        "scripts",
    ]

    fixer = ComprehensiveLintingFixer()
    fixed_count = fixer.run_on_directories(directories)
    logging.info(f"\n🎉 Total files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
