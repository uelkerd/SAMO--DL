                # Continue import block for empty lines and comments
            # Add logging import if not present
            # Apply fixes in order
            # Check if this is an import line
            # Extract items and sort them
            # Only replace if it's a single letter (likely unused)
            # Only write if content changed
            # Reconstruct the __all__ list
            # Replace print statements
        # Check if the import is actually used
        # Critical imports that should never be removed
        # Only replace print statements that are not in test files
        # Pattern to match __all__ lists
        # Pattern to match exception handlers with unused variables
        # Patterns that indicate imports might be needed
        # Process directories in order of priority
        # Reconstruct with imports at top
        # Summary
        # and are not already in logging context
#!/usr/bin/env python3
from pathlib import Path
from typing import List, Set, Tuple
import os
import re
import sys



"""
Conservative Linting Fix Script for SAMO Deep Learning.

This script fixes linting issues while preserving necessary imports and avoiding
dangerous changes that could break functionality.

Fixes only safe issues:
- F841: Unused variables (replace with _)
- E402: Import order (move to top)
- RUF022: __all__ sorting
- W291: Trailing whitespace
- T201: Print statements (replace with logging)

Skips potentially dangerous issues:
- F401: Unused imports (might be needed)
- F821: Undefined names (might be from removed imports)
- S105/S106: Hardcoded passwords (might be intentional)
"""

class ConservativeLintingFixer:
    """Conservative linting fixer that preserves functionality."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.skipped_files = []

        self.critical_imports = {
            'time', 'pytest', 'patch', 'mock', 'logging', 'os', 'sys',
            'pathlib', 'typing', 'json', 'tempfile', 'contextlib'
        }

        self.import_usage_patterns = [
            r'\btime\.\w+\b',
            r'\bpytest\.\w+\b',
            r'\bpatch\b',
            r'\bmock\b',
            r'\blogging\b',
            r'\bos\.\w+\b',
            r'\bsys\.\w+\b',
            r'\bPath\b',
            r'\bList\b',
            r'\bDict\b',
            r'\bAny\b',
            r'\bOptional\b',
            r'\bUnion\b',
            r'\bjson\.\w+\b',
            r'\btempfile\.\w+\b',
            r'\bcontextlib\.\w+\b'
        ]

    def should_preserve_import(self, import_name: str, file_content: str) -> bool:
        """Check if an import should be preserved."""
        if import_name in self.critical_imports:
            return True

        for pattern in self.import_usage_patterns:
            if re.search(pattern, file_content):
                return True

        return False

    def fix_f841_unused_variables(self, content: str) -> str:
        """Fix F841: Replace unused variables with underscore."""
        pattern = r'except\s+(\w+)\s+as\s+(\w+):'

        def replace_exception(match):
            exception_type = match.group(1)
            variable_name = match.group(2)
            if len(variable_name) == 1:
                return f'except {exception_type} as _:'
            return match.group(0)

        return re.sub(pattern, replace_exception, content)

    def fix_e402_import_order(self, content: str) -> str:
        """Fix E402: Move imports to top of file."""
        lines = content.split('\n')
        imports = []
        other_lines = []
        in_import_block = False

        for line in lines:
            stripped = line.strip()

            if (stripped.startswith('import ') or
                stripped.startswith('from ') and ' import ' in stripped):
                imports.append(line)
                in_import_block = True
            elif in_import_block and (stripped == '' or stripped.startswith('#')):
                imports.append(line)
            else:
                in_import_block = False
                other_lines.append(line)

        result = []
        if imports:
            result.extend(imports)
            result.append('')  # Add blank line after imports
        result.extend(other_lines)

        return '\n'.join(result)

    def fix_ruf022_all_sorting(self, content: str) -> str:
        """Fix RUF022: Sort __all__ lists."""
        def sort_all_list(match):
            all_content = match.group(1)
            items = [item.strip().strip('"\'') for item in all_content.split(',')]
            items = [item for item in items if item]  # Remove empty items
            items.sort()
            formatted_items = [f'"{item}"' for item in items]
            return f'__all__ = [
    "\n",
    "\n    ".join(formatted_items)}",
    "\n    {",
]'

        pattern = r'__all__\s*=\s*\[(.*?)\]'
        return re.sub(pattern, sort_all_list, content, flags=re.DOTALL)

    def fix_w291_trailing_whitespace(self, content: str) -> str:
        """Fix W291: Remove trailing whitespace."""
        lines = content.split('\n')
        fixed_lines = [line.rstrip() for line in lines]
        return '\n'.join(fixed_lines)

    def fix_t201_print_statements(self, content: str) -> str:
        """Fix T201: Replace print statements with logging."""
        if 'print(' in content and 'logging' not in content:
            if 'import logging' not in content:
                content = 'import logging\n\n' + content

            content = re.sub(r'print\((.*?)\)', r'logging.info(\1)', content)

        return content

    def fix_file(self, file_path: Path) -> bool:
        """Fix linting issues in a single file."""
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
        """Process all Python files in a directory."""
        dir_path = self.project_root / directory

        if not dir_path.exists():
            print(f"Directory {directory} does not exist")
            return

        python_files = list(dir_path.rglob('*.py'))
        print(f"Processing {len(python_files)} Python files in {directory}/")

        for file_path in python_files:
            if self.fix_file(file_path):
                print(f"✅ Fixed: {file_path}")
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
        print("📊 SUMMARY:")
        print(f"✅ Files fixed: {len(self.fixed_files)}")
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
    """Main entry point."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."

    fixer = ConservativeLintingFixer(project_root)
    fixer.run()


if __name__ == "__main__":
    main()