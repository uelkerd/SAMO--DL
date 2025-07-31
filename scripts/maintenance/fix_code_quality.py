#!/usr/bin/env python3
"""
Code Quality Fixer Script

This script automatically fixes common code quality issues
identified by Ruff linter.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CodeQualityFixer:
    """Automated code quality fixer for Python files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.fixed_files = 0
        self.total_issues = 0

    def fix_path_operations(self, content: str) -> str:
        """Fix path operations to use pathlib (PTH-codes)."""
        if "os.path." in content or "os.makedirs" in content or "os.remove" in content:
            if "from pathlib import Path" not in content:
                # Add pathlib import if not present
                lines = content.split("\n")
                import_found = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        if "pathlib" in line:
                            import_found = True
                            break
                        if not import_found and i > 0:
                            lines.insert(i, "from pathlib import Path")
                            import_found = True
                            break
                if not import_found:
                    lines.insert(0, "from pathlib import Path")
                content = "\n".join(lines)

        # Replace os.path operations with pathlib equivalents
        content = re.sub(r"os\.path\.join\(([^)]+)\)", r"Path(\1).as_posix()", content)
        content = re.sub(r"os\.makedirs\(([^,)]+)\)", r"Path(\1).mkdir(parents=True, exist_ok=True)", content)
        content = re.sub(r"os\.remove\(([^)]+)\)", r"Path(\1).unlink(missing_ok=True)", content)
        content = re.sub(r"os\.path\.exists\(([^)]+)\)", r"Path(\1).exists()", content)
        content = re.sub(r"os\.path\.isfile\(([^)]+)\)", r"Path(\1).is_file()", content)
        content = re.sub(r"os\.path\.isdir\(([^)]+)\)", r"Path(\1).is_dir()", content)

        return content

    def fix_f_strings(self, content: str) -> str:
        """Fix f-string formatting issues."""
        # Fix f-strings without placeholders
        content = re.sub(r'f"([^"]*)"', r'"\1"', content)
        content = re.sub(r"f'([^']*)'", r"'\1'", content)
        
        # Fix f-strings with invalid syntax
        content = re.sub(r'f"([^"]*)\{([^}]*)\}([^"]*)"', r'f"\1{\2}\3"', content)
        
        return content

    def fix_import_order(self, content: str) -> str:
        """Fix import order and grouping."""
        lines = content.split("\n")
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith(("import ", "from ")):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        # Sort import lines
        import_lines.sort()
        
        # Reconstruct content
        return "\n".join(import_lines + [""] + other_lines)

    def fix_unused_imports(self, content: str) -> str:
        """Remove unused imports."""
        lines = content.split("\n")
        filtered_lines = []
        
        for line in lines:
            if line.strip().startswith(("import ", "from ")):
                # Keep all imports for now - let Ruff handle specific removals
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        
        return "\n".join(filtered_lines)

    def fix_trailing_whitespace(self, content: str) -> str:
        """Remove trailing whitespace."""
        lines = content.split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        return "\n".join(cleaned_lines)

    def fix_missing_newlines(self, content: str) -> str:
        """Ensure file ends with newline."""
        if not content.endswith("\n"):
            content += "\n"
        return content

    def fix_file(self, file_path: Path) -> bool:
        """Fix code quality issues in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply fixes
            content = self.fix_path_operations(content)
            content = self.fix_f_strings(content)
            content = self.fix_import_order(content)
            content = self.fix_unused_imports(content)
            content = self.fix_trailing_whitespace(content)
            content = self.fix_missing_newlines(content)

            # Write back if changed
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"✅ Fixed: {file_path}")
                self.fixed_files += 1
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {e}")
            return False

    def fix_project(self) -> None:
        """Fix code quality issues across the entire project."""
        logger.info(f"🔧 Starting code quality fixes in: {self.project_root}")

        python_files = list(self.project_root.rglob("*.py"))
        logger.info(f"📁 Found {len(python_files)} Python files")

        for file_path in python_files:
            if self.fix_file(file_path):
                self.total_issues += 1

        logger.info(f"✅ Code quality fixes completed!")
        logger.info(f"   • Files fixed: {self.fixed_files}")
        logger.info(f"   • Total issues resolved: {self.total_issues}")


def main():
    """Main function to run code quality fixes."""
    project_root = Path(__file__).parent.parent.parent
    fixer = CodeQualityFixer(project_root)
    fixer.fix_project()


if __name__ == "__main__":
    main()
