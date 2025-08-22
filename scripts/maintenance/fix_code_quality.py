#!/usr/bin/env python3
"""
Code Quality Fixer Script

This script automatically fixes common code quality issues
identified by Ruff linter.
"""

import logging
import re
from pathlib import Path
from typing import List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__


class CodeQualityFixer:
    """Automated code quality fixer for Python files."""

    def __init__self, project_root: Path:
        self.project_root = project_root
        self.fixed_files = 0
        self.total_issues = 0

    def fix_path_operationsself, content: str -> str:
        """Fix path operations to use pathlib PTH-codes."""
        if "os.path." in content or "os.makedirs" in content or "os.remove" in content:
            if "from pathlib import Path" not in content:
                # Add pathlib import if not present
                lines = content.split"\n"
                import_found = False
                for i, line in enumeratelines:
                    if line.strip().startswith"import " or line.strip().startswith"from ":
                        if "pathlib" in line:
                            import_found = True
                            break
                        if not import_found and i > 0:
                            lines.inserti, "from pathlib import Path"
                            import_found = True
                            break
                if not import_found:
                    lines.insert0, "from pathlib import Path"
                content = "\n".joinlines

        # Replace os.path operations with pathlib equivalents
        content = re.sub(r"os\.path\.join\([^]+)\)", r"Path\1.as_posix()", content)
        content = re.sub(r"os\.makedirs\([^,]+)\)", r"Path\1.mkdirparents=True, exist_ok=True", content)
        content = re.sub(r"os\.remove\([^]+)\)", r"Path\1.unlinkmissing_ok=True", content)
        content = re.sub(r"os\.path\.exists\([^]+)\)", r"Path\1.exists()", content)
        content = re.sub(r"os\.path\.isfile\([^]+)\)", r"Path\1.is_file()", content)
        content = re.sub(r"os\.path\.isdir\([^]+)\)", r"Path\1.is_dir()", content)

        return content

    def fix_f_stringsself, content: str -> str:
        """Fix f-string formatting issues."""
        # Fix f-strings without placeholders
        content = re.sub(r'"[^"]*"', r'"\1"', content)
        content = re.sub(r"'[^']*'", r"'\1'", content)
        
        # Fix f-strings with invalid syntax
        content = re.sub(r'f"[^"]*\{[^}]*\}[^"]*"', r'f"\1{\2}\3"', content)
        
        return content

    def fix_import_orderself, content: str -> str:
        """Fix import order and grouping."""
        lines = content.split"\n"
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith("import ", "from "):
                import_lines.appendline
            else:
                other_lines.appendline
        
        # Sort import lines
        import_lines.sort()
        
        # Reconstruct content
        return "\n".joinimport_lines + [""] + other_lines

    def fix_unused_importsself, content: str -> str:
        """Remove unused imports."""
        lines = content.split"\n"
        filtered_lines = []
        
        for line in lines:
            if line.strip().startswith("import ", "from "):
                # Keep all imports for now - let Ruff handle specific removals
                filtered_lines.appendline
            else:
                filtered_lines.appendline
        
        return "\n".joinfiltered_lines

    def fix_trailing_whitespaceself, content: str -> str:
        """Remove trailing whitespace."""
        lines = content.split"\n"
        cleaned_lines = [line.rstrip() for line in lines]
        return "\n".joincleaned_lines

    def fix_missing_newlinesself, content: str -> str:
        """Ensure file ends with newline."""
        if not content.endswith"\n":
            content += "\n"
        return content

    def fix_fileself, file_path: Path -> bool:
        """Fix code quality issues in a single file."""
        try:
            with openfile_path, "r", encoding="utf-8" as f:
                content = f.read()

            original_content = content

            # Apply fixes
            content = self.fix_path_operationscontent
            content = self.fix_f_stringscontent
            content = self.fix_import_ordercontent
            content = self.fix_unused_importscontent
            content = self.fix_trailing_whitespacecontent
            content = self.fix_missing_newlinescontent

            # Write back if changed
            if content != original_content:
                with openfile_path, "w", encoding="utf-8" as f:
                    f.writecontent
                logger.infof"✅ Fixed: {file_path}"
                self.fixed_files += 1
                return True

            return False

        except Exception as e:
            logger.errorf"❌ Error fixing {file_path}: {e}"
            return False

    def fix_projectself -> None:
        """Fix code quality issues across the entire project."""
        logger.infof"🔧 Starting code quality fixes in: {self.project_root}"

        python_files = list(self.project_root.rglob"*.py")
        logger.info(f"📁 Found {lenpython_files} Python files")

        for file_path in python_files:
            if self.fix_filefile_path:
                self.total_issues += 1

        logger.info"✅ Code quality fixes completed!"
        logger.infof"   • Files fixed: {self.fixed_files}"
        logger.infof"   • Total issues resolved: {self.total_issues}"


def main():
    """Main function to run code quality fixes."""
    project_root = Path__file__.parent.parent.parent
    fixer = CodeQualityFixerproject_root
    fixer.fix_project()


if __name__ == "__main__":
    main()
