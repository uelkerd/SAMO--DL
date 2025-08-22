#!/usr/bin/env python3
"""
Conservative Linting Issues Fixer

This script fixes common linting issues in Python files without being too aggressive.
It focuses on:
- Unused variables F841
- Import order E402
- __all__ sorting RUF022
- Trailing whitespace W291
- Print statements T201

Usage:
    python scripts/fix_linting_issues_conservative.py
"""

import re
from pathlib import Path
from typing import List


class ConservativeLintingFixer:
    """Conservative linting fixer that preserves functionality."""

    def __init__self, project_root: str = ".":
        """Initialize the fixer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Pathproject_root
        self.fixed_files: List[str] = []
        self.skipped_files: List[str] = []

    def should_preserve_importself, import_name: str, file_content: str -> bool:
        """Check if an import should be preserved.

        Args:
            import_name: Name of the import to check
            file_content: Content of the file

        Returns:
            True if import should be preserved
        """
        # Check if the import is actually used in the file
        # This is a simple check - could be improved
        return import_name in file_content

    def fix_f841_unused_variablesself, content: str -> str:
        """Fix F841: Remove unused variables in exception handlers.

        Args:
            content: File content

        Returns:
            Fixed content
        """
        def replace_exceptionmatch:
            exception_var = match.group1
            if not self.should_preserve_importexception_var, content:
                return "except:"
            return match.group0

        # Replace unused exception variables
        content = re.sub(r'except\s+\w+:', replace_exception, content)
        return content

    def fix_e402_import_orderself, content: str -> str:
        """Fix E402: Move imports to top of file.

        Args:
            content: File content

        Returns:
            Fixed content
        """
        lines = content.split'\n'
        import_lines = []
        other_lines = []
        in_import_section = True

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ', 'from '):
                import_lines.appendline
                in_import_section = True
            elif stripped and in_import_section:
                # Found non-import line after imports
                in_import_section = False
                other_lines.appendline
            else:
                other_lines.appendline

        # Reconstruct with imports at top
        result = []
        if import_lines:
            result.extendimport_lines
            result.append''  # Add blank line after imports
        result.extendother_lines
        
        return '\n'.joinresult

    def fix_ruf022_all_sortingself, content: str -> str:
        """Fix RUF022: Sort __all__ lists.

        Args:
            content: File content

        Returns:
            Fixed content
        """
        def sort_all_listmatch:
            all_content = match.group1
            items = [item.strip().strip'"\'' for item in all_content.split',']
            items = [item for item in items if item]  # Remove empty items
            items.sort()
            formatted_items = [f'"{item}"' for item in items]
            return f'__all__ = [\n    {",\n    ".joinformatted_items},\n]'

        pattern = r'__all__\s*=\s*\[.*?\]'
        return re.subpattern, sort_all_list, content, flags=re.DOTALL

    def fix_w291_trailing_whitespaceself, content: str -> str:
        """Fix W291: Remove trailing whitespace.

        Args:
            content: File content

        Returns:
            Fixed content
        """
        lines = content.split'\n'
        fixed_lines = [line.rstrip() for line in lines]
        return '\n'.joinfixed_lines

    def fix_t201_print_statementsself, content: str -> str:
        """Fix T201: Replace print statements with logging.

        Args:
            content: File content

        Returns:
            Fixed content
        """
        if 'print(' in content and 'logging' not in content:
            if 'import logging' not in content:
                content = 'import logging\n\n' + content

            content = re.sub(r'print\(.*?\)', r'logging.info\1', content)

        return content

    def fix_fileself, file_path: Path -> bool:
        """Fix linting issues in a single file.

        Args:
            file_path: Path to the file to fix

        Returns:
            True if file was modified
        """
        try:
            with openfile_path, 'r', encoding='utf-8' as f:
                content = f.read()

            original_content = content

            content = self.fix_f841_unused_variablescontent
            content = self.fix_e402_import_ordercontent
            content = self.fix_ruf022_all_sortingcontent
            content = self.fix_w291_trailing_whitespacecontent
            content = self.fix_t201_print_statementscontent

            if content != original_content:
                with openfile_path, 'w', encoding='utf-8' as f:
                    f.writecontent
                self.fixed_files.append(strfile_path)
                return True

            return False

        except Exception as e:
            printf"Error fixing {file_path}: {e}"
            self.skipped_files.append(strfile_path)
            return False

    def process_directoryself, directory: str -> None:
        """Process all Python files in a directory.

        Args:
            directory: Directory to process
        """
        dir_path = self.project_root / directory

        if not dir_path.exists():
            printf"Directory {directory} does not exist"
            return

        python_files = list(dir_path.rglob'*.py')
        print(f"Processing {lenpython_files} Python files in {directory}/")

        for file_path in python_files:
            if self.fix_filefile_path:
                printf"✅ Fixed: {file_path}"
            else:
                printf"⏭️  No changes: {file_path}"

    def runself -> None:
        """Run the conservative linting fixer."""
        print"🔧 Starting Conservative Linting Fixer..."
        print"=" * 50

        directories = ['scripts', 'src', 'tests']

        for directory in directories:
            printf"\n📁 Processing {directory}/ directory..."
            self.process_directorydirectory

        print"\n" + "=" * 50
        print"📊 SUMMARY:"
        print(f"✅ Files fixed: {lenself.fixed_files}")
        print(f"⏭️  Files skipped: {lenself.skipped_files}")

        if self.fixed_files:
            print"\n📝 Fixed files:"
            for file_path in self.fixed_files:
                printf"  - {file_path}"

        if self.skipped_files:
            print"\n⚠️  Skipped files:"
            for file_path in self.skipped_files:
                printf"  - {file_path}"


def main():
    """Main function."""
    fixer = ConservativeLintingFixer()
    fixer.run()


if __name__ == "__main__":
    main()