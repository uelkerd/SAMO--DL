#!/usr/bin/env python3
""""
Conservative Linting Issues Fixer

This script fixes common linting issues in Python files without being too aggressive.
It focuses on:
- Unused variables (F841)
- Import order (E402)
- __all__ sorting (RUF022)
- Trailing whitespace (W291)
- Print statements (T201)

Usage:
    python scripts/fix_linting_issues_conservative.py
""""

import re
from pathlib import Path
from typing import List


class ConservativeLintingFixer:
    """Conservative linting fixer that preserves functionality."""

    def __init__(self, project_root: str = "."):
        """Initialize the fixer."

        Args:
            project_root: Root directory of the project
        """"
        self.project_root = Path(project_root)
        self.fixed_files: List[str] = []
        self.skipped_files: List[str] = []

    def should_preserve_import(self, import_name: str, file_content: str) -> bool:
        """Check if an import should be preserved."

        Args:
            import_name: Name of the import to check
            file_content: Content of the file

        Returns:
            True if import should be preserved
        """"
        # Check if the import is actually used in the file
        # This is a simple check - could be improved
        return import_name in file_content

    def fix_f841_unused_variables(self, content: str) -> str:
        """Fix F841: Remove unused variables in exception handlers."

        Args:
            content: File content

        Returns:
            Fixed content
        """"
        def replace_exception(match):
            exception_var = match.group(1)
            if not self.should_preserve_import(exception_var, content):
                return "except:"
            return match.group(0)

        # Replace unused exception variables
        content = re.sub(r'except\s+(\w+):', replace_exception, content)
        return content

            def fix_e402_import_order(self, content: str) -> str:
        """Fix E402: Move imports to top of file."

        Args:
            content: File content

        Returns:
            Fixed content
        """"
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        in_import_section = True

            for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                import_lines.append(line)
                in_import_section = True
            elif stripped and in_import_section:
                # Found non-import line after imports
                in_import_section = False
                other_lines.append(line)
            else:
                other_lines.append(line)

        # Reconstruct with imports at top
        result = []
            if import_lines:
            result.extend(import_lines)
            result.append('')  # Add blank line after imports
        result.extend(other_lines)

        return '\n'.join(result)

            def fix_ruf022_all_sorting(self, content: str) -> str:
        """Fix RUF022: Sort __all__ lists."

        Args:
            content: File content

        Returns:
            Fixed content
        """"
            def sort_all_list(match):
            all_content = match.group(1)
            items = [item.strip().strip('"\'') for item in all_content.split(',')]"
            items = [item for item in items if item]  # Remove empty items
            items.sort()
            formatted_items = [""{item}"' for item in items]"
            return "__all__ = [\n    {",\n    ".join(formatted_items)},\n]'"

        pattern = r'__all__\s*=\s*\[(.*?)\]'
        return re.sub(pattern, sort_all_list, content, flags=re.DOTALL)

            def fix_w291_trailing_whitespace(self, content: str) -> str:
        """Fix W291: Remove trailing whitespace."

        Args:
            content: File content

        Returns:
            Fixed content
        """"
        lines = content.split('\n')
        fixed_lines = [line.rstrip() for line in lines]
        return '\n'.join(fixed_lines)

            def fix_t201_print_statements(self, content: str) -> str:
        """Fix T201: Replace print statements with logging."

        Args:
            content: File content

        Returns:
            Fixed content
        """"
            if 'print(' in content and 'logging' not in content:)
            if 'import logging' not in content:
                content = 'import logging\n\n' + content

            content = re.sub(r'print\((.*?)\)', r'logging.info(\1)', content)

        return content

            def fix_file(self, file_path: Path) -> bool:
        """Fix linting issues in a single file."

        Args:
            file_path: Path to the file to fix

        Returns:
            True if file was modified
        """"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            content = self.fix_f841_unused_variables(content)
            content = self.fix_e402_import_order(content)
            content = self.fix_ruf022_all_sorting(content)
            content = self.fix_w291_trailing_whitespace(content)
            content = self.fix_t201_print_statements(content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                return True

            return False

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            self.skipped_files.append(str(file_path))
            return False

            def process_directory(self, directory: str) -> None:
        """Process all Python files in a directory."

        Args:
            directory: Directory to process
        """"
        dir_path = self.project_root / directory

            if not dir_path.exists():
            print(f"Directory {directory} does not exist")
            return

        python_files = list(dir_path.rglob('*.py'))
        print(f"Processing {len(python_files)} Python files in {directory}/")

            for file_path in python_files:
            if self.fix_file(file_path):
                print(f" Fixed: {file_path}")
            else:
                print(f"⏭️  No changes: {file_path}")

            def run(self) -> None:
        """Run the conservative linting fixer."""
        print("🔧 Starting Conservative Linting Fixer...")
        print("=" * 50)

        directories = ['scripts', 'src', 'tests']

            for directory in directories:
            print(f"\n📁 Processing {directory}/ directory...")
            self.process_directory(directory)

        print("\n" + "=" * 50)
        print(" SUMMARY:")
        print(f" Files fixed: {len(self.fixed_files)}")
        print(f"⏭️  Files skipped: {len(self.skipped_files)}")

            if self.fixed_files:
            print("\n📝 Fixed files:")
            for file_path in self.fixed_files:
                print(f"  - {file_path}")

            if self.skipped_files:
            print("\n⚠️  Skipped files:")
            for file_path in self.skipped_files:
                print(f"  - {file_path}")


            def main():
    """Main function."""
    fixer = ConservativeLintingFixer()
    fixer.run()


            if __name__ == "__main__":
    main()
